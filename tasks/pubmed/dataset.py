import logging

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


class PubMedDataset:
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        super().__init__()
        datasets.set_progress_bar_enabled(False)

        self.tokenizer = tokenizer
        self.data_args = data_args

        self.multiple_choice = False

        self.label_list = [
            "BACKGROUND",
            "CONCLUSIONS",
            "METHODS",
            "OBJECTIVE",
            "RESULTS",
        ]
        self.num_labels = 5

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {id: label for label, id in self.label2id.items()}

        dir_name = (
            "tasks/pubmed/pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign"
        )
        raw_datasets = {
            "train": self.get_examples(dir_name, "train"),
            "test": self.get_examples(dir_name, "test"),
            "validation": self.get_examples(dir_name, "dev"),
        }
        raw_datasets = DatasetDict(raw_datasets)

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

        self.train_dataset = raw_datasets["train"]

        self.train_samples = len(raw_datasets["train"])
        if data_args.max_train_samples is not None:
            self.train_dataset = self.train_dataset.select(
                range(data_args.max_train_samples)
            )

        if training_args.do_eval:
            self.eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(
                    range(data_args.max_eval_samples)
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
        self.metric = load_metric(
            "./tasks/pubmed/pubmed_metric.py", data_args.dataset_name
        )

        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(
                tokenizer, pad_to_multiple_of=8
            )

        self.test_key = "accuracy"

    def get_examples(self, data_dir, split):
        examples = []
        with open(f"{data_dir}/{split}.txt", "r") as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            if line.startswith("###"):
                abstract_lines = ""
            elif line.isspace():
                abstract_lines_split = abstract_lines.splitlines()

                for abstrct_line_number, abstact_line in enumerate(
                    abstract_lines_split
                ):
                    line_data = {}
                    target_text_split = abstact_line.split("\t")
                    line_data["label"] = target_text_split[0]
                    line_data["label"] = int(self.label2id[line_data["label"]])
                    line_data["text"] = target_text_split[1].lower()
                    line_data["line_number"] = abstrct_line_number
                    line_data["total_lines"] = len(abstract_lines_split) - 1
                    examples.append(line_data)
            else:
                abstract_lines += line

        examples = pd.DataFrame(examples)
        examples = Dataset.from_pandas(pd.DataFrame(examples))
        return examples

    def preprocess_function(self, examples):
        args = examples["text"]
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
