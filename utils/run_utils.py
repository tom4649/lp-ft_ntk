import logging
import os
import pathlib

import datasets
import torch
import transformers
from transformers import AutoTokenizer

from tasks.ood.dataset import OOD_DATASET_DICT, OODDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def setup_library(training_args):
    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    datasets.set_progress_bar_enabled(False)
    training_args.disable_tqdm = True
    return training_args

def train(trainer):
    train_result = trainer.train()
    if train_result is None:
        return
    metrics = train_result.metrics
    if len(metrics) == 0:
        return
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

def evaluate(trainer):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def predict(trainer, predict_dataset=None):
    logger.info("*** Predict ***")
    predictions = trainer.predict(predict_dataset)

    trainer.log_metrics("test", predictions.metrics)
    trainer.save_metrics("test", predictions.metrics)
    trainer.log(predictions.metrics)
    return

def predict_ood(args, trainer):
    original_output_dir = trainer.args.output_dir
    _, data_args, training_args = args
    id_dataset_name = data_args.dataset_name
    for ood_dataset_name in OOD_DATASET_DICT[id_dataset_name]:
        trainer.args.output_dir = original_output_dir.replace(
            id_dataset_name, ood_dataset_name
        )
        os.makedirs(trainer.args.output_dir, exist_ok=True)
        data_args.dataset_name = ood_dataset_name
        dataset = OODDataset(trainer.tokenizer, data_args, training_args)
        logger.info(f"*** Predict OOD on {ood_dataset_name} ***")
        predictions = trainer.predict(dataset.predict_dataset)

        trainer.log_metrics("test", predictions.metrics)
        trainer.save_metrics("test", predictions.metrics)
        trainer.log(predictions.metrics)
    trainer.args.output_dir = original_output_dir
    return

def check_output_dir(training_args):
    to_continue = True
    if training_args.do_train:
        output_dir_path = pathlib.Path(training_args.output_dir)
        if (
            output_dir_path.exists()
            and any(output_dir_path.iterdir())
        ):
            if not training_args.overwrite_output_dir:
                logger.info(
                    f" `file exists in ({training_args.output_dir}) and --overwrite_output_dir` is not True, so exit."
                )
                to_continue = False
            else:
                logger.info(
                    f" file exists in ({training_args.output_dir}), but `--overwrite_output_dir` is True, so continue."
                )
    return to_continue


