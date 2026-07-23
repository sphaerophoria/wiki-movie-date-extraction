import torch
import torch.nn as nn
from transformers import AutoModel

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KNOWN_MODELS = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "google-bert/bert-base-uncased",
    "answerdotai/ModernBERT-base",
    "sentence-transformers/all-MiniLM-L6-v2",
]
MEDIUM_LABELS = ["home", "theatrical"]
BIO_LABELS = ["O", "B", "I"]


class MyNetwork(nn.Module):
    def __init__(self, bert_model_name, num_token_labels, num_medium_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden = self.bert.config.hidden_size
        self.token_head = nn.Linear(hidden, num_token_labels)
        self.medium_head = nn.Linear(hidden, num_medium_classes)

    def forward(self, input_ids, attention_mask, labels=None, medium_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = outputs.last_hidden_state
        token_logits = self.token_head(seq)
        medium_logits = self.medium_head(seq)

        loss = None
        if labels is not None or medium_labels is not None:
            loss_terms = []
            if labels is not None:
                bio_weights = torch.tensor([1.0, 10.0, 10.0], device=token_logits.device)
                loss_token = nn.CrossEntropyLoss(weight=bio_weights, ignore_index=-100)(
                    token_logits.view(-1, token_logits.size(-1)),
                    labels.view(-1)
                )
                loss_terms.append(loss_token)
            if medium_labels is not None and labels is not None:
                b_mask = (labels == 1).float().unsqueeze(-1)
                loss_medium = nn.BCEWithLogitsLoss(reduction="none")(medium_logits, medium_labels)
                loss_medium = (loss_medium * b_mask).sum() / (b_mask.sum() * len(MEDIUM_LABELS) + 1e-8)
                loss_terms.append(loss_medium)
            loss = sum(loss_terms)

        return {"loss": loss, "token_logits": token_logits, "medium_logits": medium_logits}
