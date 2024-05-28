import logging
import os

import datasets
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from datasets.load import load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)

from utils.custom_metric import custom_metric

logger = logging.getLogger(__name__)


OOD_DATASET_DICT = {
    "amazon":["dynasent", "semeval", "sst5"],
}

class OODDataset:
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        super().__init__()
        datasets.set_progress_bar_enabled(False)
        data_dir = (
            f"tasks/OOD_NLP/datasets/process/SentimentAnalysis/{data_args.dataset_name}"
        )
        if data_args.dataset_name == "amazon":
            raw_datasets = {
                "train": self.get_examples(data_dir, "train"),
                "test": self.get_examples(data_dir, "test"),
            }
        else:
            raw_datasets = {
                "test": self.get_examples(data_dir, "test"),
            }
        raw_datasets = DatasetDict(raw_datasets)

        self.tokenizer = tokenizer
        self.data_args = data_args

        self.multiple_choice = False

        self.label_list = ["negative", "positive", "neutral"]
        self.num_labels = 3

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        self.label2id = {"negative": 0, "positive": 1, "neutral": 2}
        self.id2label = {id: label for label, id in self.label2id.items()}

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        self.train_dataset, self.eval_dataset = None, None
        if data_args.dataset_name == "amazon":
            logger.info("Split the train dataset!")
            train_dataset = raw_datasets["train"].train_test_split(
                test_size=0.1, shuffle=False
            )

            self.train_dataset, self.eval_dataset = train_dataset = (
                train_dataset["train"],
                train_dataset["test"],
            )
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(
                    range(data_args.max_train_samples)
                )

        self.predict_dataset = None
        if (
            training_args.do_predict
            or data_args.dataset_name is not None
            or data_args.test_file is not None
        ):
            self.predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(
                    range(data_args.max_predict_samples)
                )
        # offline
        self.metric = load_metric("./tasks/ood/ood_metric.py", data_args.dataset_name)

        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(
                tokenizer, pad_to_multiple_of=8
            )

        self.test_key = "accuracy"

    def get_examples(self, data_dir, split):
        examples = []
        lines = pd.read_csv(
            os.path.join(data_dir, f"{split}.tsv"), sep="\t", header=0
        ).values
        for idx, line in enumerate(lines):
            text_a = line[0]
            label = line[1]
            guid = "%s-%s" % (split, idx)

            if not isinstance(text_a, str):
                # print(line)
                continue

            try:
                example = {"guid": guid, "text_a": text_a, "label": int(label)}
            except:
                # print(line)
                continue
            examples.append(example)
        examples = Dataset.from_pandas(pd.DataFrame(examples))
        return examples

    def preprocess_function(self, examples):
        args = examples["text_a"]
        result = self.tokenizer(
            args,
            padding=self.padding,
            max_length=self.max_seq_length,
            truncation=True,
        )

        return result

    def compute_metrics(self, p: EvalPrediction):
        probs = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = np.array(p.label_ids)
        custom_metrics = custom_metric(probs, labels, self.num_labels)
        preds = np.argmax(probs, axis=1)

        if self.data_args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            for key, value in custom_metrics.items():
                result[key] = value
            return result
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
