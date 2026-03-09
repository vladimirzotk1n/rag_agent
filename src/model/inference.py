import torch

from src.logger_config import logger

from .device import DEVICE
from .model import model, tokenizer


@torch.no_grad()
def dense_embed(text: str) -> list[float]:
    try:
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        # model(**inputs)[0] -> tensor of shape [batch, dim]
        embedding = model(**inputs)[0]
    except torch.cuda.OutOfMemoryError as e:
        logger.critical(f"Cuda out of memory when creating embedding: {e}")
        raise
    except Exception as e:
        logger.exception(f"Error creating embedding {e}")
        raise

    return embedding.cpu().tolist()

