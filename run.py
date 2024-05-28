import logging
import os
import sys
from functools import partialmethod

import tqdm

from arguments import get_args
from tasks.get_trainer import get_trainer
from utils.run_utils import check_output_dir, setup_library

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from utils.run_utils import evaluate, predict, predict_ood, train

logger = logging.getLogger(__name__)

def main():
    args = get_args()
    model_args, data_args, training_args = args
    training_args = setup_library(training_args)
    to_continue = check_output_dir(training_args)
    if not to_continue:
        return
    args = model_args, data_args, training_args
    trainer, predict_dataset = get_trainer(args)

    if training_args.do_train:
        train(trainer)
    if training_args.do_eval:
        evaluate(trainer)
    if training_args.do_predict:
        predict(trainer, predict_dataset)
    if data_args.task_name == "ood":
        predict_ood(args, trainer)
    logger.info(f"All done!")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    _default_log_level = logging.INFO
    logger.setLevel(_default_log_level)

    main()
