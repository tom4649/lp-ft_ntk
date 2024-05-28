########## The following part was originally copied from Transformers' trainer (3.4.0) and then changed heavily to compute eNTKs.  ##########

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
"""
The trainer for computing eNTKs
"""

import gc
import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from functorch import jacrev, jvp, make_functional_with_buffers, vmap
from torch.autograd.functional import jacobian
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.sampler import SequentialSampler
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, set_seed
from transformers.data.data_collator import DataCollator
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer import SequentialDistributedSampler
from transformers.trainer_utils import EvalPrediction, PredictionOutput, TrainOutput
from transformers.training_args import TrainingArguments

from model.utils import TaskType, get_model
from tasks.glue.dataset import GlueDataset
from tasks.ood.dataset import OODDataset
from tasks.pubmed.dataset import PubMedDataset
from tasks.superglue.dataset import SuperGlueDataset
from training.linearhead_trainer import (
    LinearHeadTrainer,
    get_token_prediction_layer,
    varsize_tensor_all_gather,
)
from utils.kernel_solvers import solve_kernel

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm

logger = logging.getLogger(__name__)
_default_log_level = logging.INFO
logger.setLevel(_default_log_level)
import json
import pathlib


def calculate_large_einsum(jac_sub_params_eval, jac_sub_params_train, chunk_size):
    n, c, p = jac_sub_params_eval.shape
    m, d, _ = jac_sub_params_train.shape
    start = 0
    end = min(start + chunk_size, p)
    chunk_eval = jac_sub_params_eval[:, :, start:end]  # [n, c, chunk_size]
    chunk_train = jac_sub_params_train[:, :, start:end]  # [m, d, chunk_size]
    full_result = torch.einsum("ncp,mdp->ncmd", chunk_eval, chunk_train)

    for start in range(end, p, chunk_size):
        end = min(start + chunk_size, p)
        chunk_eval = jac_sub_params_eval[:, :, start:end]
        chunk_train = jac_sub_params_train[:, :, start:end]
        chunk_result = torch.einsum("ncp,mdp->ncmd", chunk_eval, chunk_train)
        full_result += chunk_result

    return full_result


class LogitModelWrapper(nn.Module):
    def __init__(self, model, pad_token_id=None):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        self.model_type = model.config.model_type

    def forward(self, input_ids, attention_mask):
        if "roberta" in self.model_type:  # for roberta
            logits = self.model(input_ids, attention_mask)[0]  # don't provide labels
            return logits[:, 0, :]  # only take the first token's logits
        else:  # for gpt2
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
            batch_size = logits.shape[0]
            return logits[
                torch.arange(batch_size, device=logits.device), sequence_lengths
            ]


def predicate_without_one_layer(layer_id, sub_param="attention", model_type="roberta"):
    # sub_param: attention or intermediate or output
    if "roberta" in model_type:
        if sub_param == "no_attention":
            return lambda name: not (
                name.startswith(f"encoder.layer.{layer_id}.")
                and "attention" not in name
            )
        return lambda name: not (
            name.startswith(f"encoder.layer.{layer_id}.{sub_param}")
        )
    elif "gpt2" in model_type:
        if sub_param == "attention":
            return lambda name: not name.startswith(f"h.{layer_id}.attn.")
        elif sub_param == "no_attention":
            return lambda name: name.startswith(
                f"h.{layer_id}.attn."
            ) or not name.startswith(f"h.{layer_id}.")
        elif sub_param == "intermediate":
            return lambda name: not name.startswith(f"h.{layer_id}.mlp.c_fc.")
        elif sub_param == "output":
            return (
                lambda name: name.startswith(f"h.{layer_id}.attn.")
                or name.startswith(f"h.{layer_id}.mlp.c_fc.")
                or not name.startswith(f"h.{layer_id}.")
            )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def predicate_without_embedding(is_word_embedding=True, model_type="roberta"):
    if "roberta" in model_type:
        if is_word_embedding:
            return lambda name: not name.startswith("embeddings.word_embeddings")
        else:
            return lambda name: not (
                name.startswith("embeddings.") and "word_embeddings" not in name
            )
    elif "gpt2" in model_type:
        if is_word_embedding:
            return lambda name: not name.startswith("wte")
        else:
            return lambda name: not name.startswith("wpe")


def flatten_params(params_tuple):
    flattened_params = torch.cat([p.flatten() for p in params_tuple])
    shape = [p.shape for p in params_tuple]
    offsets = [0] + [p.numel() for p in params_tuple]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    return flattened_params, shape, offsets


def unflatten_params(flattened_params, shapes, offsets):
    tensors = []
    for offset, shape in zip(offsets, shapes):
        numel = torch.prod(torch.tensor(shape)).item()
        tensor = flattened_params[offset : offset + numel].view(shape)
        tensors.append(tensor)
    return tuple(tensors)


def param_to_buffer(module, module_name, predicate, return_name_list=False):
    """Turns all parameters of a module into buffers."""
    modules = module.named_modules(prefix=str(module_name))
    next(modules)  # Skip itself
    params = []
    name_list = []
    for name, param in module.named_parameters(recurse=False, prefix=str(module_name)):
        if predicate(name):
            # print(f"Delete {name} from {module_name} in param to buffer")
            params.append((name.split(".")[-1], param))
            name_list.append(name)

    for name, param in params:
        delattr(module, name)  # Unregister parameter
        module.register_buffer(name, param)
    for name, module in modules:
        name_list_sub = param_to_buffer(
            module, name, predicate, return_name_list=return_name_list
        )
        if return_name_list:
            name_list.extend(name_list_sub)
    if return_name_list:
        return name_list
    else:
        return None


def buffer_to_param_from_name_list(module, module_name, name_list):
    modules = module.named_modules(prefix=str(module_name))
    next(modules)  # Skip itself

    buffers = []
    for name, buf in module.named_buffers(recurse=False, prefix=str(module_name)):
        if name in name_list:
            # print(f"Delete {name} from {module_name} in buffer to param")
            buffers.append((name.split(".")[-1], buf))
    for name, buf in buffers:
        delattr(module, name)
        module.register_parameter(name, torch.nn.Parameter(buf))
    for name, module in modules:
        buffer_to_param_from_name_list(module, name, name_list)


class KernelTrainerFunc(LinearHeadTrainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        *posargs,
        **kwargs,
    ):
        super().__init__(
            model, args, data_collator, train_dataset, eval_dataset, *posargs, **kwargs
        )

        self.grad_dim = None
        self.train_train_kernel = None
        self.train_targets = None
        self.num_labels = None

        def convert_to_buffer(name):
            if model.config.lora:
                if (
                    name.startswith("roberta") or name.startswith("transformer")
                ) and "lora" not in name:
                    logger.info("Excluding {}".format(name))
                    return True
            return False

        param_to_buffer(self.model, "", convert_to_buffer)
        if self.args.from_linearhead:
            loss_option_org = self.args.loss_option
            self.args.loss_option = "lp"  # use regularization in LinearProbing
            super().train()  # Train output layer using LinearHeadTrainer
            self.args.loss_option = loss_option_org
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        if len(self.train_dataset) > 250:
            logger.info(f"train_dataset length is too long: {len(self.train_dataset)}")
            index_path = f"{self.args.output_dir}/selected_train_dataset.json"
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    index = json.load(f)
            else:
                index = np.random.choice(len(self.train_dataset), 250, replace=False)
            self.train_dataset = self.train_dataset.select(index)
            with open(index_path, "w") as f:
                json.dump(index.tolist(), f)

    def profile_memory(self):
        import gc

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(obj, "data") and torch.is_tensor(obj.data)
                ):
                    print(type(obj), obj.size())
            except:
                pass

    def compute_kernel_inner(
        self,
        curried_fn,
        curried_jacobian_fn,
        grads_outer,
        dataset_inner,
        batch_size=None,
    ):
        dataloader_inner = self.get_unshuffled_dataloader(
            dataset_inner,
            sharded=False,
            batch_size=self.args.per_device_eval_batch_size
            if batch_size is None
            else batch_size,
        )
        kernel_blocks = []
        targets_inner = []

        for inputs_inner in dataloader_inner:
            for k, v in inputs_inner.items():
                if isinstance(v, torch.Tensor):
                    inputs_inner[k] = v.to(self.args.device)

            def get_ntk_slice(tangents):
                _, jvps = curried_fn(
                    inputs_inner.get("input_ids"),
                    inputs_inner.get("attention_mask"),
                    tangents,
                )
                return jvps

            block = (
                    vmap(vmap(get_ntk_slice))(grads_outer).to(torch.float64).cpu()
                )  # N_outer x C_outer x N_inner x C_inner

            kernel_blocks.append(block.detach())
            label = inputs_inner.get("labels")
            targets_inner.append(label)

            # del grads_inner
            del block
            del inputs_inner

            torch.cuda.empty_cache()
            gc.collect()

        return (
            torch.cat(kernel_blocks, dim=2) if kernel_blocks else torch.tensor([]),
            torch.cat(targets_inner, dim=0) if targets_inner else torch.tensor([]),
        )

    def get_kernel_blocks(self, model_wrapper, dataloader_outer, dataset_inner):
        for param in model_wrapper.parameters():
            param.requires_grad_(True)
        for buf in model_wrapper.buffers():
            buf.requires_grad_(False)

        model_fn, params, buffers = make_functional_with_buffers(model_wrapper)
        jacobian_fn = jacrev(model_fn)

        def curried_jacobian_fn(input_ids, attention_mask):
            return jacobian_fn(params, buffers, input_ids, attention_mask)

        def curried_fn(input_ids, attention_mask, tangent):
            def curried_model_fn(params_):
                return model_fn(params_, buffers, input_ids, attention_mask)

            return jvp(curried_model_fn, (params,), (tangent,))

        inner_targets = None
        kernel_blocks = []
        for inputs_outer in dataloader_outer:
            for k, v in inputs_outer.items():
                if isinstance(v, torch.Tensor):
                    inputs_outer[k] = v.to(self.args.device)

            grads_outer = curried_jacobian_fn(
                inputs_outer.get("input_ids"),
                inputs_outer.get("attention_mask"),
            )

            if self.grad_dim is None:
                self.grad_dim = sum(np.prod(x.shape[2:]) for x in grads_outer)

            # Starting to compute kernel inner
            kernel_blocks_per_param, inner_targets = self.compute_kernel_inner(
                curried_fn,
                curried_jacobian_fn,
                grads_outer,
                dataset_inner,
                batch_size=self.args.per_device_eval_batch_size,
            )

            kernel_blocks.append(kernel_blocks_per_param)

            del grads_outer
            del inputs_outer
            del kernel_blocks_per_param

            torch.cuda.empty_cache()
            gc.collect()
        kernel_blocks = torch.cat(
            kernel_blocks, dim=0
        )  # N_outer x C_outer x N_inner x C_inner
        return kernel_blocks, inner_targets

    def get_jacobian_sub_params(
        self, model_wrapper, dataloader_outer
    ):
        for param in model_wrapper.parameters():
            param.requires_grad_(True)
        for buf in model_wrapper.buffers():
            buf.requires_grad_(False)

        model_fn, params, buffers = make_functional_with_buffers(model_wrapper)
        if len(params) == 0:
            return None, None

        inner_targets = []
        jac_blocks = []
        classifier_weight = (
            get_token_prediction_layer(self.model).weight.detach().to("cpu")
        )
        for i, inputs_outer in enumerate(dataloader_outer):
            for k, v in inputs_outer.items():
                if isinstance(v, torch.Tensor):
                    inputs_outer[k] = v.to(self.args.device)
            flattened_params, shapes, offsets = flatten_params(params)

            def model_fn_buffers(flattened_params_):
                unflattened_params = unflatten_params(
                    flattened_params_, shapes, offsets
                )
                return model_fn(
                    unflattened_params,
                    buffers,
                    input_ids=inputs_outer.get("input_ids"),
                    attention_mask=inputs_outer.get("attention_mask"),
                )

            jac = jacobian(model_fn_buffers, flattened_params).detach().to("cpu")
            jac = torch.einsum("ch,nhp->ncp", classifier_weight, jac)
            jac_blocks.append(
                jac,
            )
            inner_targets.append(inputs_outer.get("labels"))
            del flattened_params
            del shapes
            del offsets
            del inputs_outer
            del jac
            del model_fn_buffers

            torch.cuda.empty_cache()
            gc.collect()
        jac_sub_params = torch.cat(jac_blocks, dim=0)  # N_outer x C_outer x P
        inner_targets = torch.cat(inner_targets, dim=0)
        return jac_sub_params.to(self.args.device), inner_targets

    def get_jacobian_word_embedding(
        self, model_wrapper, dataloader, sample_ratio=0.1
    ):
        set_seed(self.args.seed)
        model_fn, params, buffers = make_functional_with_buffers(model_wrapper)
        if len(params) == 0:
            return None, None
        num_samples_of_param = int(
            self.model.config.vocab_size * self.model.config.hidden_size * sample_ratio
        )
        num_sample_of_output = int(self.model.config.hidden_size * sample_ratio)

        param_sample_index = random.sample(
            range(self.model.config.vocab_size * self.model.config.hidden_size),
            num_samples_of_param,
        )

        targets = None
        jac_blocks = None
        classifier_weight_all = (
            get_token_prediction_layer(self.model).weight.detach().to("cpu")
        )
        flattened_params, shapes, offsets = flatten_params(params)
        logger.info("Start to compute jacobian for word embedding")
        for i, inputs in enumerate(dataloader):
            # print("i", i)
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
            output_sample_index = random.sample(
                range(self.model.config.hidden_size), num_sample_of_output
            )
            classifier_weight = classifier_weight_all[:, output_sample_index]

            def model_fn_buffers(flattened_params_):
                unflattened_params = unflatten_params(
                    flattened_params_, shapes, offsets
                )
                return model_fn(
                    unflattened_params,
                    buffers,
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask"),
                )[:, output_sample_index]

            jac = jacobian(model_fn_buffers, flattened_params)
            jac_block = jac[:, :, param_sample_index].detach().to("cpu")
            jac_block = torch.einsum("ch,nhp->ncp", classifier_weight, jac_block)
            actual_batch_size = jac_block.shape[0]
            start_index = i * self.args.per_device_eval_batch_size
            end_index = start_index + actual_batch_size
            if jac_blocks is None:
                jac_blocks = torch.zeros(
                    len(dataloader.dataset), jac_block.shape[1], jac_block.shape[2]
                )
                targets = torch.zeros(len(dataloader.dataset))
            jac_blocks[start_index:end_index] = jac_block
            targets[start_index:end_index] = inputs.get("labels")
            del inputs
            del jac
            del model_fn_buffers

            torch.cuda.empty_cache()
            gc.collect()
        jac_sub_params = jac_blocks / sample_ratio
        return jac_sub_params.to(self.args.device), targets

    def get_kernel_word_embedding(
        self,
        model_wrapper,
        dataloader_train,
        dataloader_eval,
        train_train=True,
        sample_ratio=0.1,
    ):
        predicate = predicate_without_embedding(
            is_word_embedding=True, model_type=model_wrapper.model_type
        )
        logger.info("Start to compute kernel for word embedding")
        name_list = param_to_buffer(
            model_wrapper.model, "", predicate, return_name_list=True
        )
        if os.path.exists(f"{self.args.output_dir}/jacobian_word_embedding_train.pt"):
            logger.info("Load jacobian_word_embedding_train")
            with open(
                f"{self.args.output_dir}/jacobian_word_embedding_train.pt", "rb"
            ) as f:
                jacobian_word_embedding_train = torch.load(f)
        else:
            logger.info("Compute jacobian_word_embedding_train")
            jacobian_word_embedding_train, _ = self.get_jacobian_word_embedding(
                model_wrapper, dataloader_train, sample_ratio=sample_ratio
            )
        if jacobian_word_embedding_train is None:
            return None
        with open(
            f"{self.args.output_dir}/jacobian_word_embedding_train.pt", "wb"
        ) as f:
            torch.save(jacobian_word_embedding_train, f)
        if not train_train:
            if os.path.exists(
                f"{self.args.output_dir}/jacobian_word_embedding_eval.pt"
            ):
                logger.info("Load jacobian_word_embedding_eval")
                with open(
                    f"{self.args.output_dir}/jacobian_word_embedding_eval.pt", "rb"
                ) as f:
                    jacobian_word_embedding_eval = torch.load(f)
            else:
                logger.info("Compute jacobian_word_embedding_eval")
                del jacobian_word_embedding_train
                jacobian_word_embedding_eval, _ = self.get_jacobian_word_embedding(
                    model_wrapper,
                    dataloader_eval,
                    sample_ratio=sample_ratio,
                )
                logger.info(
                    f"jacobian_word_embedding_eval: {jacobian_word_embedding_eval.shape}"
                )
                if jacobian_word_embedding_eval is None:
                    return None
                with open(
                    f"{self.args.output_dir}/jacobian_word_embedding_eval.pt", "wb"
                ) as f:
                    torch.save(jacobian_word_embedding_eval, f)
                with open(
                    f"{self.args.output_dir}/jacobian_word_embedding_train.pt", "rb"
                ) as f:
                    jacobian_word_embedding_train = torch.load(f)
        else:
            jacobian_word_embedding_eval = jacobian_word_embedding_train

        kernel_word_embedding = torch.einsum(
            "ncp, mdp->ncmd",
            jacobian_word_embedding_eval,
            jacobian_word_embedding_train,
        ).to(self.args.device) / (sample_ratio)

        buffer_to_param_from_name_list(model_wrapper.model, "", name_list=name_list)

        del jacobian_word_embedding_train
        del jacobian_word_embedding_eval
        return kernel_word_embedding

    def get_kernel_layers(
        self,
        model_wrapper,
        dataloader_train,
        dataloader_eval,
        predicate,
        train_train=True,
    ):
        name_list = param_to_buffer(
            model_wrapper.model, "", predicate, return_name_list=True
        )
        logger.info(
            f"Subparam to buffer done: parameters: {len(list(model_wrapper.model.named_parameters()))} buffer: {len(list(model_wrapper.model.named_buffers()))}",
        )

        jac_sub_params_train, train_targets = self.get_jacobian_sub_params(
            model_wrapper, dataloader_train
        )
        if jac_sub_params_train is None:
            return None, None
        jac_sub_params_eval, eval_targets = (
            self.get_jacobian_sub_params(
                model_wrapper, dataloader_eval
            )
            if not train_train
            else (jac_sub_params_train, train_targets)
        )
        if jac_sub_params_eval is None:
            return None, None


        kernel = torch.einsum(
            "ncp, mdp->ncmd", jac_sub_params_eval, jac_sub_params_train
        ).to(self.args.device)
        logger.info(f"kernel {kernel.shape}")  # N x C x M x D

        del jac_sub_params_eval
        del jac_sub_params_train
        del train_targets

        buffer_to_param_from_name_list(model_wrapper.model, "", name_list=name_list)
        return kernel, eval_targets

    def compute_ntk_ftE(
        self,
        eval_dataset,
        batch_size=None,
        train_train=True,
        sample_ratio=0.1,
    ):
        device = self.args.device
        train_dataset = self.train_dataset
        dataloader_train = self.get_unshuffled_dataloader(
            train_dataset,
            sharded=True,
            batch_size=batch_size
            if batch_size is not None
            else self.args.per_device_train_batch_size,
        )
        eval_dataset = self.train_dataset if train_train else eval_dataset
        dataloader_eval = (
            self.get_unshuffled_dataloader(
                eval_dataset,
                sharded=True,
                batch_size=batch_size
                if batch_size is not None
                else self.args.per_device_train_batch_size,
            )
            if not train_train
            else None
        )
        if "roberta" in self.model.config.model_type:
            model_wrapper = LogitModelWrapper(self.model.roberta)
        elif "gpt2" in self.model.config.model_type:
            model_wrapper = LogitModelWrapper(
                self.model.transformer, pad_token_id=self.model.config.pad_token_id
            )
        else:
            raise NotImplementedError
        model_wrapper.eval()
        logger.info(
            f"Parameters: {len(list(model_wrapper.model.named_parameters()))} Buffers: {len(list(model_wrapper.model.named_buffers()))}"
        ),
        suffix = "" if train_train else "_train_eval"
        kernel_layers_path = pathlib.Path(self.args.output_dir) / (
            "kernel_layers" + suffix
        )
        kernel_layers_path.mkdir(exist_ok=True)
        computed_kernel_layers_path = kernel_layers_path.rglob("*.pt")
        computed_kernel_layers_path = list(computed_kernel_layers_path)
        computed_kernel_layers_path = sorted(
            computed_kernel_layers_path, key=lambda x: int(x.stem.split("_")[-1])
        )
        last_computed_kernel_layers_path = (
            computed_kernel_layers_path[-1]
            if len(computed_kernel_layers_path) > 0
            else None
        )
        last_computed_kernel_layers_id = (
            int(last_computed_kernel_layers_path.stem.split("_")[-1])
            if last_computed_kernel_layers_path is not None
            else -1
        )
        kernel = (
            torch.load(last_computed_kernel_layers_path)
            if last_computed_kernel_layers_path is not None
            else torch.zeros(
                len(eval_dataset),
                self.model.config.num_labels,
                len(train_dataset),
                self.model.config.num_labels,
            )
        )
        kernel = kernel.to(device)

        if not self.model.config.lora and last_computed_kernel_layers_path is None:
            kernel_word_embedding = self.get_kernel_word_embedding(
                model_wrapper,
                dataloader_train,
                dataloader_eval,
                train_train=train_train,
                sample_ratio=sample_ratio,
            )
            if kernel_word_embedding is not None:
                kernel = kernel + kernel_word_embedding
            del kernel_word_embedding
            torch.cuda.empty_cache()
        layers_length = (
            len(model_wrapper.model.encoder.layer)
            if model_wrapper.model_type == "roberta"
            else len(model_wrapper.model.h)
        )
        if not self.model.config.lora:
            if train_train:
                predicate_list = [
                    predicate_without_embedding(
                        is_word_embedding=False, model_type=model_wrapper.model_type
                    )
                ] + [
                    predicate_without_one_layer(
                        i, sub_param=sub_param, model_type=model_wrapper.model_type
                    )
                    for sub_param in ["attention", "no_attention"]
                    for i in range(layers_length)
                ] # devide the layers into attention and no_attention
            else:
                predicate_list = [
                    predicate_without_embedding(
                        is_word_embedding=False, model_type=model_wrapper.model_type
                    )
                ] + [
                    predicate_without_one_layer(
                        i, sub_param=sub_param, model_type=model_wrapper.model_type
                    )
                    for sub_param in ["attention", "intermediate", "output"]
                    for i in range(layers_length)
                ]  # divide the layers into attention, intermediate, output to avoid memory error
        else:
            predicate_list = [
                predicate_without_one_layer(
                    i, sub_param="attention", model_type=model_wrapper.model_type
                )
                for i in range(layers_length)
            ]
        for id_predicate, predicate in enumerate(predicate_list):
            if id_predicate <= last_computed_kernel_layers_id:
                continue
            print(
                f"Starting to compute kernel for param {id_predicate + 1} / {len(predicate_list)}"
            )

            kernel_layers, eval_targets = self.get_kernel_layers(
                model_wrapper,
                dataloader_train,
                dataloader_eval,
                predicate,
                train_train=train_train,
            )
            if kernel_layers is None:
                continue
            kernel_layers = kernel_layers.to(device)
            kernel = kernel + kernel_layers
            del kernel_layers
            with open(
                kernel_layers_path / f"kernel_layers_{id_predicate}.pt", "wb"
            ) as f:
                torch.save(kernel, f)
            if (kernel_layers_path / f"kernel_layers_{id_predicate - 1}.pt").exists():
                os.remove(kernel_layers_path / f"kernel_layers_{id_predicate - 1}.pt")
            torch.cuda.empty_cache()
            gc.collect()

        return (kernel, eval_targets)

    def get_phi(self, model_wrapper, dataloader):
        model_wrapper.eval()
        all_phi = []
        targets = []
        for input in dataloader:
            for k, v in input.items():
                if isinstance(v, torch.Tensor):
                    input[k] = v.to(self.args.device)
            phi = model_wrapper(
                input.get("input_ids"),
                input.get("attention_mask"),
            )
            all_phi.append(phi)
            targets.append(input.get("labels"))
            del phi
            del input
            torch.cuda.empty_cache()
            gc.collect()
        all_phi = torch.cat(all_phi, dim=0)
        targets = torch.cat(targets, dim=0)
        return all_phi, targets

    def compute_ntk_ptE(self, eval_dataset, batch_size=None, train_train=True):
        # compute pre-train-effective component of NTK
        train_dataset = self.train_dataset
        dataloader_train = self.get_unshuffled_dataloader(
            train_dataset,
            sharded=True,
            batch_size=batch_size
            if batch_size is not None
            else self.args.per_device_train_batch_size,
        )
        dataloader_eval = (
            self.get_unshuffled_dataloader(
                eval_dataset,
                sharded=True,
                batch_size=batch_size
                if batch_size is not None
                else self.args.per_device_train_batch_size,
            )
            if not train_train
            else None
        )

        if "roberta" in self.model.config.model_type:
            model_wrapper = LogitModelWrapper(self.model.roberta)
        elif "gpt2" in self.model.config.model_type:
            model_wrapper = LogitModelWrapper(
                self.model.transformer, pad_token_id=self.model.config.pad_token_id
            )
        else:
            raise NotImplementedError
        model_wrapper.eval()
        all_phi_train, targets_train = self.get_phi(model_wrapper, dataloader_train)
        all_phi_eval, targets_eval = (
            self.get_phi(model_wrapper, dataloader_eval)
            if not train_train
            else (all_phi_train, targets_train)
        )
        kernel = all_phi_eval @ all_phi_train.T
        kernel_half = kernel + 1.0
        num_labels = self.model.config.num_labels
        N, M = kernel.shape
        ntk = torch.zeros(N, num_labels, M, num_labels)
        for i in range(num_labels):
            ntk[:, i, :, i] = kernel_half
        return (ntk, targets_eval)

    def compute_kernel_sharded(self, dataset, term="ptE", train_train=True):
        with torch.no_grad():
            if term == "ptE":
                kernel, targets_eval = self.compute_ntk_ptE(
                    dataset, batch_size=1, train_train=train_train
                )
            elif term == "ftE":
                kernel, targets_eval = self.compute_ntk_ftE(
                    dataset, batch_size=1, train_train=train_train
                )
            else:
                raise NotImplementedError

        if self.args.local_rank != -1:
            logger.info("Starting to gather kernel across GPUs")
            kernel = varsize_tensor_all_gather(
                kernel.to(self.args.device), torch.distributed.get_world_size()
            )
            logger.info("Finished gathering kernel across GPUs")

        return kernel, targets_eval

    def compute_model_logits_cached(self, eval_dataset):
        if self.args.load_kernels is not None:
            output_dir = self.args.load_kernels
        else:
            output_dir = self.args.output_dir
        logit_file_name = "logits.pt"
        logit_path = os.path.join(output_dir, logit_file_name)

        if os.path.exists(logit_path) and not self.args.overwrite_kernels:
            logger.info(f"Starting to load logits from {logit_path}.")
            logits, targets = torch.load(logit_path)
            logger.info(f"Finished loading logits from {logit_path}.")
        else:
            logger.info(f"Starting to compute the logits.")
            dataloader = self.get_unshuffled_dataloader(eval_dataset)

            if "roberta" in self.model.config.model_type:
                model_wrapper = LogitModelWrapper(self.model.roberta)
            elif "gpt2" in self.model.config.model_type:
                model_wrapper = LogitModelWrapper(
                    self.model.transformer, pad_token_id=self.model.config.pad_token_id
                )
            else:
                raise NotImplementedError
            model_wrapper.eval()

            logits = []
            targets = []
            with torch.no_grad():
                for inputs in dataloader:
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.args.device)

                    label = inputs.get("labels")

                    preds = model_wrapper(
                        inputs.get("input_ids"),
                        inputs.get("attention_mask"),
                        inputs.get("mask_pos"),
                    )
                    logits.append(preds.detach().cpu())
                    targets.append(label.cpu())

            logits = torch.cat(logits, dim=0)
            targets = torch.cat(targets, dim=0)

            logger.info(f"Finished computing the logits.")

            if self.is_world_process_zero():
                torch.save((logits, targets), logit_path)
        return logits, targets

    def reshape_kernel_and_targets(self, kernel, targets):
        # reshape kernel to previous format
        if self.num_labels is None:
            self.num_labels = kernel.shape[1]
        assert (
            self.num_labels == kernel.shape[1]
        ), "label dim not constant: {} and {}".format(self.num_labels, kernel.shape[1])
        assert (
            self.num_labels == kernel.shape[3]
        ), "label dim not constant: {} and {}".format(self.num_labels, kernel.shape[3])

        if self.num_labels > 1:  # multi logit
            targets = torch.nn.functional.one_hot(targets.squeeze(), self.num_labels)

        size1 = kernel.shape[0] * kernel.shape[1]
        size2 = kernel.shape[2] * kernel.shape[3]
        return kernel.reshape(1, size1, size2), targets.reshape(-1)

    def compute_kernel_cached(
        self, eval_dataset, kernel_stem_name="kernels", train_train=True
    ):
        ptE, ftE, targets = None, None, None
        for term in ["ptE", "ftE"]:
            kernel_path = os.path.join(
                self.args.output_dir, f"{kernel_stem_name}_{term}.pt"
            )
            if os.path.exists(kernel_path) and not self.args.overwrite_kernels:
                logger.info(f"Starting to load kernels from {kernel_path}.")
                kernel, targets = torch.load(kernel_path)
                logger.info(f"Finished loading kernels from {kernel_path}.")
            else:
                logger.info(f"Starting to compute the kernel.")
                kernel, targets = self.compute_kernel_sharded(
                    eval_dataset, term=term, train_train=train_train
                )
                logger.info(f"Finished computing the kernel.")

                kernel, targets = kernel.cpu(), targets.cpu()

                if self.is_world_process_zero():
                    torch.save((kernel, targets), kernel_path)
            if kernel is None:
                raise ValueError(f"Kernel is None for {term}")
            if term == "ptE":
                ptE = kernel
            elif term == "ftE":
                ftE = kernel
        return ptE, ftE, targets

    def train(self, model_path=None, dev_objective=None):
        eval_dataset = self.train_dataset
        logger.info("Starting to compute the kernel for training set.")
        self.train_ptE, self.train_ftE, self.train_targets = self.compute_kernel_cached(
            eval_dataset, kernel_stem_name="kernels_train_train"
        )
        return TrainOutput(0, 0.0, {}), None

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        kernel_file_name="kernels.pt",
        **kwargs,
    ) -> Dict:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if len(eval_dataset) > 250:
            logger.info(f"eval_dataset length is too long: {len(eval_dataset)}")
            index_path = f"{self.args.output_dir}/selected_eval_dataset.json"
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    index = json.load(f)
            else:
                index = np.random.choice(len(eval_dataset), 250, replace=False)
            eval_dataset = eval_dataset.select(index)
            with open(index_path, "w") as f:
                json.dump(index.tolist(), f)
        logger.info("Starting to compute the kernel for evaluation set.")
        eval_ptE, eval_ftE, eval_targets = self.compute_kernel_cached(
            eval_dataset, kernel_stem_name="kernels_eval", train_train=False
        )
        assert (
            hasattr(self, "train_ptE") and hasattr(self, "train_ftE") and self.train_ptE is not None and self.train_ftE is not None
        ), "Train kernel is None, did you forget to call train()?"
        kernel_dict = {
            "train_ptE": self.train_ptE,
            "train_ftE": self.train_ftE,
            "eval_ptE": eval_ptE,
            "eval_ftE": eval_ftE,
            "train_targets": self.train_targets,
            "eval_targets": eval_targets,
        }
        for key in ["train_ptE", "train_ftE", "eval_ptE", "eval_ftE"]:
            kernel = kernel_dict[key]
            targets_key = "train_targets" if "train" in key else "eval_targets"
            kernel, targets = self.reshape_kernel_and_targets(kernel.cpu(), kernel_dict[targets_key].cpu())
            kernel_dict[key] = kernel
            if "ftE" in key:
                kernel_dict[targets_key] = targets

        # get train and test logits
        if self.args.adjust_for_init:
            train_logits, _ = self.compute_model_logits_cached(self.train_dataset)
            eval_logits, _ = self.compute_model_logits_cached(eval_dataset)
            train_logits = train_logits.reshape(-1, 1)
            eval_logits = eval_logits.reshape(-1, 1)
        else:
            train_logits, eval_logits = None, None

        solvers, scores = solve_kernel(self, kernel_dict, train_logits, eval_logits)
        return scores


def get_trainer(args):
    model_args, data_args, training_args = args
    logger.info(f"set model randome seed {model_args.model_seed}")
    set_seed(model_args.model_seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    if model_args.model_name_or_path == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token

    if data_args.task_name.lower() == "superglue":
        dataset = SuperGlueDataset(tokenizer, data_args, training_args)
    elif data_args.task_name.lower() == "glue":
        dataset = GlueDataset(tokenizer, data_args, training_args)
    elif data_args.task_name.lower() == "ood":
        dataset = OODDataset(tokenizer, data_args, training_args)
    elif data_args.task_name.lower() == "pubmed":
        dataset = PubMedDataset(tokenizer, data_args, training_args)
    else:
        raise ValueError(f"Task {data_args.task_name} not supported.")
    assert not (hasattr(dataset, "multiple_choice") and dataset.multiple_choice), "Multiple choice not supported"
    assert not (hasattr(dataset, "is_regression") and dataset.is_regression), "Regression not supported"
    config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )

    assert not hasattr(config, "adapter_config"), "Adapter not supported"
    config.lora = False
    task_type = TaskType.SEQUENCE_CLASSIFICATION

    model = get_model(
        model_args,
        task_type,
        config,
        fix_bert=False,
    )

    set_seed(training_args.seed)
    logger.info(f"set data randome seed {model_args.model_seed}")

    callbacks = []
    trainer_cls = KernelTrainerFunc

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.eval_dataset,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        callbacks=callbacks,
    )

    return trainer, dataset.predict_dataset
