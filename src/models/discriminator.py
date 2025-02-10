from __future__ import annotations
import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Type


class Discriminator(nn.Module, ABC):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass


class DummyDiscriminator(Discriminator):
    def __init__(self, name: str = "dummy") -> None:
        super().__init__(name=name)
        self.model = nn.Sequential(
            nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 1), nn.Sigmoid()
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)


def discriminator_mapper(discriminator_name: str) -> Type[Discriminator]:
    if discriminator_name == "dummy":
        return DummyDiscriminator
    else:
        raise ValueError(f"Discriminator named '{discriminator_name}' not found")
