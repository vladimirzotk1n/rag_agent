import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.config import settings

from .device import DEVICE


class E5Vectorizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.e5 = AutoModel.from_pretrained(
            "intfloat/multilingual-e5-small", token=settings.hf_token
        )

    def forward(self, input_ids, attention_mask):
        embs = self.e5(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # [b, s, e]
        emb_size = embs.shape[-1]

        mask = attention_mask.unsqueeze(-1).expand(-1, -1, emb_size)  # [B, S, E]

        pooled_embs = (embs * mask).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

        embs_normalized = F.normalize(pooled_embs, p=2, dim=1)
        return embs_normalized


tokenizer = AutoTokenizer.from_pretrained(
    "intfloat/multilingual-e5-small", token=settings.hf_token
)
model = E5Vectorizer()
model.to(device=DEVICE)
model.eval()

# Загрузить веса
