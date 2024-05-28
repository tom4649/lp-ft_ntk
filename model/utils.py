from enum import Enum

import torch
import torch.nn.functional as F
from transformers import (  # AutoModelForSequenceClassification,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
)

from model.gpt2.modeling_gpt2 import GPT2ForSequenceClassification
from model.roberta import RobertaForSequenceClassificationLinear
from model.sequence_classification import RobertaLoraForSequenceClassificationLinear


class TaskType(Enum):
    TOKEN_CLASSIFICATION = (1,)
    SEQUENCE_CLASSIFICATION = (2,)
    QUESTION_ANSWERING = (3,)
    MULTIPLE_CHOICE = 4

AUTO_MODELS = {
    TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
    TaskType.SEQUENCE_CLASSIFICATION: RobertaForSequenceClassificationLinear,
    TaskType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice,
}

LORA_MODELS = {
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaLoraForSequenceClassificationLinear,
    },
    "gpt2": {
        TaskType.SEQUENCE_CLASSIFICATION: GPT2ForSequenceClassification,
    },
}


def get_model(
    model_args,
    task_type: TaskType,
    config: AutoConfig,
    fix_bert: bool = False,
):
    if config.model_type == "gpt2":
        if model_args.lora:
            config.lora = True
            config.lora_r = model_args.lora_r
            config.lora_alpha = model_args.lora_alpha
        model_class = GPT2ForSequenceClassification  # TODO: Add LoRA
        config._attn_implementation = "eager"
        config.scale_attn_by_inverse_layer_idx = True
        config.reorder_and_upcast_attn = False
        config.pad_token_id = config.eos_token_id
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    elif model_args.lora:
        config.lora = True
        config.lora_r = model_args.lora_r
        config.lora_alpha = model_args.lora_alpha
        model_class = LORA_MODELS[config.model_type][task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
    else:
        model_class = AUTO_MODELS[task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )

        bert_param = 0
        if fix_bert:
            if config.model_type == "bert":
                for param in model.bert.parameters():
                    param.requires_grad = False
                for _, param in model.bert.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "deberta":
                for param in model.deberta.parameters():
                    param.requires_grad = False
                for _, param in model.deberta.named_parameters():
                    bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print("***** total param is {} *****".format(total_param))
    return model
