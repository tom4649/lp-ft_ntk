from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments

from tasks.utils import DATASETS, TASKS


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.training_args
    """

    task_name: str = field(
        metadata={
            "help": "The name of the task to train on: " + ", ".join(TASKS),
            "choices": TASKS
        },
    )
    dataset_name: str = field(
        metadata={
            "help": "The name of the dataset to use: " + ", ".join(DATASETS),
            "choices": DATASETS
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."}
    )
    template_id: Optional[int] = field(
        default=0,
        metadata={
            "help": "The specific prompt string to use"
        }
    )
    pilot: Optional[str] = field(
        default=None,
        metadata={"help": "do the pilot experiments."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    lora: bool = field(
        default=False,
        metadata={
            "help": "Will use lora during training"
        }
    )
    lora_r: int = field(
        default=8,
        metadata={
            "help": "The rank of lora"
        }
    )
    lora_alpha: int = field(
        default=16,
        metadata={
            "help": "The length of prompt"
        }
    )
    model_seed: int = field(
        default=1111,
        metadata={
            "help": "The random seed of model initialization."
        }
    )
    patient: int = field(
        default=10, metadata={"help": "The patient of early stopping."}
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Custom TrainingArguments class to include new arguments.
    """

    # Training specification
    loss_option: Optional[str] = field(
        default=None, metadata={"help": "The loss_option method to use"}
    )
    no_early_stopping: bool = field(default=True, metadata={"help": "Whether to use early stopping or not"})
    # Arguments for Kernal Trainer
    kernel_formula: str = field(
        default="sgd",
        metadata={
            "help": "choose kernel formula from {sgd, signgd, asymmetric_signgd}"
        },
    )
    kernel_solver: str = field(
        default="logistic",
        metadata={
            "help": "choose kernel solver from {lstsq, logistic, svr, svc, asym (only for asymmetric_signgd)}"
        },
    )
    load_kernels: str = field(
        default=None,
        metadata={
            "help": "when specified, loads the kernels from the folder given here"
        },
    )
    overwrite_kernels: bool = field(
        default=False,
        metadata={
            "help": "when specified, overwrites the kernels in the output_dir and computes them from scratch"
        },
    )
    from_linearhead: bool = field(
        default=False,
        metadata={
            "help": "when specified, trains the linear head before the kernel training"
        },
    )
    kernel_regularization: float = field(
        default=0.0, metadata={"help": "Regularization constant for kernel"}
    )
    kernel_gamma: float = field(
        default=1.0, metadata={"help": "Gamma for asymmetric kernel solver"}
    )
    binary_classification: bool = field(
        default=False,
        metadata={
            "help": "If num_classes=2, convert two softmax logits to single sigmoid logit"
        },
    )
    adjust_for_init: bool = field(
        default=False,
        metadata={"help": "when on, trains kernel on y-f0 and adds f0 at test time"},
    )
    f0_scaling: float = field(
        default=1.0,
        metadata={
            "help": "adjust label scaling, might help with --adjust_for_init perf"
        },
    )

def get_args():
    """Parse all the args."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))

    args = parser.parse_args_into_dataclasses()

    return args

def main():
    args = get_args()
    print(args)

if __name__ == "__main__":
    main()
