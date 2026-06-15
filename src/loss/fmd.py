import torch
from torch import nn


class FmdLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._fmd = None

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._fmd(x, y)
