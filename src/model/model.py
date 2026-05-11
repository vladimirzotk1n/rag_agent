import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel, AutoTokenizer

from src.config import settings

from .device import DEVICE


class E5Vectorizer(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.e5 = AutoModel.from_pretrained(settings.e5_model, token=settings.hf_token)

    def forward(self, input_ids, attention_mask):
        embs = self.e5(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # [b, s, e]
        emb_size = embs.shape[-1]

        mask = attention_mask.unsqueeze(-1).expand(-1, -1, emb_size)  # [B, S, E]

        pooled_embs = (embs * mask).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

        embs_normalized = F.normalize(pooled_embs, p=2, dim=1)
        return embs_normalized


tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer, token=settings.hf_token)
model = E5Vectorizer.from_pretrained(
    "VladimirFireBall/tk_rf_e5-small-v3", token=settings.hf_token
)
model.to(device=DEVICE)
model.eval()
