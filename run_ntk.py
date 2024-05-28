import logging
import os
import sys
from functools import partialmethod

import tqdm

from arguments import get_args
from training.kernel_trainer import get_trainer
from utils.run_utils import setup_library

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

logger = logging.getLogger(__name__)

def main():
    args = get_args()
    model_args, data_args, training_args = args
    training_args = setup_library(training_args)
    args = model_args, data_args, training_args
    trainer, _ = get_trainer(args)

    if training_args.do_train:
        trainer.train()
    if training_args.do_eval:
        evaluation_metrics = trainer.evaluate()
        trainer.log_metrics("eval", evaluation_metrics)
        trainer.save_metrics("eval", evaluation_metrics)
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
