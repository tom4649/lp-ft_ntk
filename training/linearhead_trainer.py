########## The following part was originally copied from https://github.com/princeton-nlp/MeZO/blob/main/medium_models/src/linearhead_trainer.py and then changed.  ##########

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import copy
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
)
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler
from transformers import Trainer
from transformers.trainer_utils import TrainOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)

def get_token_prediction_layer(model):
    # if isinstance(model, transformers.RobertaForSequenceClassificationLinear):
    if hasattr(model, "classifier"):
        return model.classifier.out_proj
    elif hasattr(model, "score"):
        return model.score
    else:
        raise NotImplementedError(model.__class__)

def extract_features(model, *args, **kwargs):
    """some magic for getting features pre last layer"""
    features = {}
    if hasattr(model, "classifier"): # RobertaForSequenceClassificationLinear

        def hook(model_, input_, output_):
            features["features"] = input_[0].detach()

        get_token_prediction_layer(model).register_forward_hook(hook)
    elif hasattr(model, "score"): # GPT2ForSequenceClassification

        def hook(model_, input_, output_):
            if len(input_) == 0:
                return
            input_ids = input_[0]
            batch_size, sequence_length = input_ids.shape[:2]
            sequence_lengths = (
                torch.eq(input_ids, model.config.pad_token_id).int().argmax(-1) - 1
            )
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(input_ids.device)
            features["features"] = output_[0][
                torch.arange(batch_size, device=output_[0].device), sequence_lengths
            ].detach()

        model.transformer.register_forward_hook(hook)
    model.forward(*args, **kwargs)
    return features["features"]


def get_features_tensor(model, train_dataloader, device):
    model.eval()

    targets = []
    features = []

    arg_names = model.forward.__code__.co_varnames[: model.forward.__code__.co_argcount]
    with torch.no_grad():
        for step, inputs in enumerate(train_dataloader):
            inputs_ = {}
            for k, v in inputs.items():
                if k not in arg_names:
                    continue
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                inputs_[k] = v
            feature = extract_features(model, **inputs_)
            features.append(feature.cpu())  # GPUからCPUにデータを移動
            targets.append(inputs["labels"].cpu())  # 同様にターゲットもCPUに移動

            del feature  # GPUメモリから削除
            torch.cuda.empty_cache()  # キャッシュをクリア
    features = torch.cat(features, dim=0).to(device)
    targets = torch.cat(targets, dim=0).to(device)
    return features, targets

def tensor_all_gather(tensor: torch.Tensor, distributed_world_size: int):
    tensor_list = [torch.zeros_like(tensor) for _ in range(distributed_world_size)]
    torch.distributed.all_gather(tensor_list=tensor_list, tensor=tensor)
    return torch.cat(tensor_list, dim=0)


def varsize_tensor_all_gather(tensor: torch.Tensor, distributed_world_size: int):
    tensor = tensor.contiguous()

    dim_tensor = torch.tensor([tensor.size(0)], dtype=torch.int64, device=tensor.device)
    dim_tensor = tensor_all_gather(dim_tensor, distributed_world_size).cpu()
    max_size = dim_tensor.max()

    padded = torch.empty(
        max_size, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device
    )
    padded[: tensor.shape[0]] = tensor

    ag = tensor_all_gather(padded, distributed_world_size)
    slices = []
    for i, sz in enumerate(dim_tensor):
        start_idx = i * max_size
        end_idx = start_idx + sz.item()

        if end_idx > start_idx:
            slices.append(ag[start_idx:end_idx])

    return torch.cat(slices, dim=0)

def train_linear_head(
    model,
    train_dataloader,
    device,
    max_iter=5000,
    many_workers=False,
    use_cv=True,
    no_reg=False,
    random_state=0,
):
    features, targets = get_features_tensor(model, train_dataloader, device)

    if many_workers:
        logger.info("Starting to gather features across workers")
        features = varsize_tensor_all_gather(
            features, torch.distributed.get_world_size()
        )
        targets = varsize_tensor_all_gather(targets, torch.distributed.get_world_size())
        logger.info("Finished gathering features across workers")

    features = features.cpu()
    targets = targets.cpu()

    if model.num_labels == 1:  # Regression
        targets_coords = targets.squeeze().unsqueeze(-1).float()
        reg = LinearRegression().fit(features.numpy(), targets_coords.numpy())
    else:
        use_bias = hasattr(model, "classifier")
        tol = 1e-4  # 1e-4 is scipy default
        print(
            f"Fitting logistic regression with max_iter: {max_iter}, random_state: {random_state}, regularization: {not no_reg}, cv: {use_cv}"
        )
        if use_cv:
            reg = LogisticRegressionCV(
                penalty="none" if no_reg else "l2",
                max_iter=max_iter,
                fit_intercept=use_bias,
                multi_class="multinomial",
                random_state=random_state,
                tol=tol,
            )
        else:
            reg = LogisticRegression(
                penalty="none" if no_reg else "l2",
                max_iter=max_iter,
                fit_intercept=use_bias,
                multi_class="multinomial",
                random_state=random_state,
                tol=tol,
            )
        logger.info("Fitting linear regression")
        reg = reg.fit(features.numpy(), targets.numpy())

    logger.info("Assigning weights to model")
    decoder = get_token_prediction_layer(model)
    coef_torch, bias_torch = coef_to_tensor(reg, model, use_bias)

    decoder.weight.data = coef_torch
    if use_bias:
        decoder.bias.data = bias_torch

    logits = torch.tensor(reg.predict_log_proba(features.numpy()))
    train_loss = torch.nn.functional.cross_entropy(
        logits, targets.squeeze(), reduction="none"
    )
    print("Finished assigning weights to linear probing layer")
    return train_loss, reg, features, targets


def coef_to_tensor(reg, model, use_bias):
    decoder = get_token_prediction_layer(model)
    coef_torch = torch.tensor(
        reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype
    )
    bias_torch = None
    if use_bias:
        bias_torch = torch.tensor(
            reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype
        )
    if model.num_labels == 2 and coef_torch.size(0) == 1:
        print("Binary classification")
        coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
        if use_bias:
            bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

    return coef_torch, bias_torch

class LinearHeadTrainer(Trainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def train(
        self, max_iter=5000, model_path=None, dev_objective=None, *args, **kwargs
    ):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        _ でstageを分割
        Loss:["normal", "lp", "lp_ft"]
        """
        self.best_dir = None
        self.objective = -float("inf")

        if self.args.loss_option == "normal":
            train_result = super().train(*args, **kwargs)
            return train_result

        print(f"Start linear probing: {self.args.loss_option}")
        dataloader = self.get_train_dataloader()
        use_cv = True
        no_reg = False
        max_iter = 5000 if max_iter is None else max_iter

        train_loss, reg, features, targets = train_linear_head(
            self.model,
            dataloader,
            self.args.device,
            max_iter=max_iter,
            use_cv=use_cv,
            many_workers=self.args.local_rank != -1,
            no_reg=no_reg,
            random_state=self.args.seed,
        )

        if self.args.loss_option == "lp":
            return TrainOutput(0, train_loss, {})

        assert self.args.loss_option == "lp_ft"
        # Fine-tune the model
        logger.info("FT stage of LP-FT")
        self.model.train()
        return super().train(*args, **kwargs)

    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, *args, **kwargs
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(
            eval_dataset, collections.abc.Sized
        ):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        return output.metrics

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels", None)

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

    def get_unshuffled_dataloader(
        self,
        dataset=None,
        sharded=False,
        batch_size=-1,
    ):
        if dataset is None:
            dataset = self.train_dataset
        sampler = SequentialSampler(dataset)

        bs = self.args.per_device_eval_batch_size if batch_size == -1 else batch_size
        data_loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=bs,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader
