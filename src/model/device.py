import torch

from src.config import settings


def get_device() -> torch.device:
    if settings.use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
