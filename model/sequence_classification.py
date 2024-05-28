import torch
import torch.nn as nn

# import torch.nn.functional as F
# from torch import Tensor
# from torch._C import NoopLogger
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# from transformers import BertModel, BertPreTrainedModel
from transformers.adapters.model_mixin import ModelWithHeadsAdaptersMixin

# from transformers.adapters.models.bert import (
#     BertModelAdaptersMixin,
#     BertModelHeadsMixin,
# )
from transformers.modeling_outputs import (  # BaseModelOutput,; Seq2SeqLMOutput,
    SequenceClassifierOutput,
)

# from transformers import RobertaModel, RobertaPreTrainedModel
from model.roberta import (
    RobertaClassificationHead,
    RobertaClassificationHeadLinear,
    RobertaModel,
    RobertaPreTrainedModel,
)


class RobertaLoraForSequenceClassification(
    ModelWithHeadsAdaptersMixin, RobertaPreTrainedModel
):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()

        for name, param in self.roberta.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print("total param is {}".format(total_param))  # 9860105

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        adapter_names=None,
        head=None,
        **kwargs
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            adapter_names=adapter_names,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaLoraForSequenceClassificationLinear(
    RobertaLoraForSequenceClassification,
    ModelWithHeadsAdaptersMixin,
    RobertaPreTrainedModel,
):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.classifier = RobertaClassificationHeadLinear(config)

        self.init_weights()
