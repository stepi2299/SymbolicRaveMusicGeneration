from __future__ import annotations
import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Type, Dict

from transformers import BertConfig

from src.models.clamp import PatchEncoder


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


class ClampDiscriminator(Discriminator):
    def __init__(self, patchilizer_config: Dict, patch_size: int) -> None:
        super().__init__(name="clamp")
        patchilizer_config = BertConfig(**patchilizer_config)
        self.patch_encoder = PatchEncoder(
            config=patchilizer_config,
            patch_size=patch_size,
            projection_layer=patchilizer_config.hidden_size,
            token_size=self.token_size,
        )
        self.classification_layer = nn.Linear(self.patch_encoder.projection_layer, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        masks = (data != 0).any(dim=-1).long()

        hidden_states = self.patch_encoder(music_inputs=data, music_masks=masks)
        out = self.classification_layer(hidden_states)
        return self.sigmoid(out)


def discriminator_mapper(discriminator_name: str) -> Type[Discriminator]:
    if discriminator_name == "dummy":
        return DummyDiscriminator
    elif discriminator_name == "clamp":
        return PatchEncoder
    else:
        raise ValueError(f"Discriminator named '{discriminator_name}' not found")
