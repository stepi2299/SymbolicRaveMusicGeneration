import torch
from torch import nn
from abc import ABC, abstractmethod

class Discriminator(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass