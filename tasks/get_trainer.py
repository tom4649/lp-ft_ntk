import logging

from transformers import AutoConfig, AutoTokenizer, EarlyStoppingCallback, set_seed

from model.utils import TaskType, get_model
from tasks.glue.dataset import GlueDataset
from tasks.ood.dataset import OODDataset
from tasks.pubmed.dataset import PubMedDataset
from tasks.superglue.dataset import SuperGlueDataset
from training.linearhead_trainer import LinearHeadTrainer

logger = logging.getLogger(__name__)


def get_trainer(args):
    model_args, data_args, training_args = args
    logger.info(f"set model randome seed {model_args.model_seed}")
    set_seed(model_args.model_seed)
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

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

    trainer_cls = LinearHeadTrainer
    callbacks = []
    if (not training_args.no_early_stopping) and training_args.load_best_model_at_end:
        callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=model_args.patient)
            )
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


if __name__ == "__main__":
    from arguments import get_base_args

    args = get_base_args("amazon", "ft", "normal")
    trainer = get_trainer(args)
    logger.info("success")
