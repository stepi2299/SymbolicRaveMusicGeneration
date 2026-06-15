import logging as lg
from abc import ABC, abstractmethod

import torch
from torch import nn

from src.models.sequence_decoder import BaseDecoder

logger = lg.getLogger(__name__)


class BaseWrapper(nn.Module, ABC):
    def __init__(self, decoder: BaseDecoder, input_dropout_rate: float) -> None:
        super().__init__()
        self._decoder = decoder
        self._input_dropout_rate = input_dropout_rate
        logger.info(
            f"Initialized {self.__class__.__name__} with decoder: {decoder.__class__.__name__} and input dropout rate: {input_dropout_rate}"
        )

    @abstractmethod
    def forward(
        self,
        batch_data: tuple[torch.Tensor, torch.Tensor],
        *,
        latent_vector: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        pass

    @abstractmethod
    def generate(
        self,
        *,
        latent_vector: torch.Tensor | None = None,
        class_idx: torch.Tensor | None = None,
        strict_sampling: bool = True,
        use_cache: bool = False,
        temperature: float = 1.0,
    ) -> list[list[int]] | list[int]:
        pass

    @abstractmethod
    def input_dropout(self, embeddings: torch.Tensor, input_sequence: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def sample_next_item(
        self,
        logits: torch.Tensor | dict[str, torch.Tensor],
        strict_sampling: bool,
        temperature: float,
    ) -> list[int] | int:
        pass

    @staticmethod
    @abstractmethod
    def reconstruct(reconstruction: torch.Tensor, target_data: torch.Tensor) -> tuple[float, dict]:
        pass
