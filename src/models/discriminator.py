from __future__ import annotations
import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Type, Dict

from transformers import BertConfig

from src.models.clamp import PatchEncoder


class Discriminator(nn.Module, ABC):
    def __init__(self, name: str,
                 patch_len: int,
                 patch_size: int,
                 token_size: int,
                 ) -> None:
        super().__init__()
        self.name = name
        self.patch_size = patch_size
        self.patch_len = patch_len
        self.token_size = token_size

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass


class DummyDiscriminator(Discriminator):
    def __init__(self, name: str = "dummy") -> None:
        super().__init__(name=name, patch_len=64, patch_size=64, token_size=128)
        self.model = nn.Sequential(
            nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 1), nn.Sigmoid()
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)


class ClampDiscriminator(Discriminator):
    def __init__(self, patchilizer_config: Dict, patch_size: int, patch_len: int, token_size) -> None:
        super().__init__(name="clamp", patch_len=patch_len, patch_size=patch_size, token_size=token_size)
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

        hidden_states = self.patch_encoder(input_patches=data, input_masks=masks)
        out = self.classification_layer(hidden_states)
        return self.sigmoid(out)

class Clamp2Discriminator(Discriminator):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        patch_len: int,
        patch_size: int,
        token_size: int,
    ) -> None:
        super(Clamp2Discriminator, self).__init__(name="clamp2", patch_size=patch_size, patch_len=patch_len, token_size=token_size)
        patchilizer_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=patch_len,
            num_attention_heads=hidden_size // 64,
            intermediate_size=hidden_size * 4,
        )
        self.patch_encoder = PatchEncoder(
            config=patchilizer_config,
            patch_size=patch_size,
            projection_layer=hidden_size,
            token_size=self.token_size,
        )
        self.classification_layer = nn.Linear(self.patch_encoder.projection_layer, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        masks = (data != 0).any(dim=-1).long()

        hidden_states = self.patch_encoder(input_patches=data, input_masks=masks)
        out = self.classification_layer(hidden_states)
        return self.sigmoid(out)


def discriminator_mapper(discriminator_name: str) -> Type[Discriminator]:
    if discriminator_name == "dummy":
        return DummyDiscriminator
    elif discriminator_name == "clamp":
        return ClampDiscriminator
    elif discriminator_name == "clamp2":
        return Clamp2Discriminator
    else:
        raise ValueError(f"Discriminator named '{discriminator_name}' not found")
