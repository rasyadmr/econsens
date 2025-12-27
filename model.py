# model_definition.py
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

class IndoBERTCustom(BertPreTrainedModel):
    def __init__(self, config, dropout_rate=0.3, hidden_dim=256):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(config.hidden_size, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_dim, config.num_labels)

        self.loss_fn = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        pooled_output = outputs.pooler_output

        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        logits = self.out(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )