from typing import Any
from omegaconf import DictConfig

from torch import nn
import torch
from abc import ABC, abstractmethod
import logging as lg
from pathlib import Path

from src.adapter import MusicAdapter
from src.constants import ExportFormat
from src.loss.base_loss import BaseLoss

logger = lg.getLogger(__name__)


class BaseHighLevelModel(nn.Module, ABC):
    def __init__(
        self,
        adapter: MusicAdapter,
        loss_function: BaseLoss,
        use_cache_generation: bool = False,
        strict_sampling: bool = True,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._use_cache = use_cache_generation
        self._strict_sampling = strict_sampling
        self._temperature = 1.0
        self._loss_function = loss_function
        self._original_sample = (
            None  # Useful for single data training to compute reconstruction loss
        )

    def set_generation_params(
        self,
        use_cache: bool = False,
        strict_sampling: bool = True,
        temperature: float = 1,
    ) -> None:
        logger.info(
            f"Setting generation params: use_cache={use_cache}, strict_sampling={strict_sampling}, temperature={temperature}"
        )
        self._use_cache = use_cache
        self._strict_sampling = strict_sampling
        self._temperature = temperature

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def training_step(
        self, data: torch.Tensor, global_step: int
    ) -> tuple[torch.Tensor, float, dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def validation_step(
        self, data: torch.Tensor, global_step: int
    ) -> tuple[torch.Tensor, float, dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    @torch.no_grad()
    def generate(
        self,
        folder_path: Path,
        file_name: str = "generated_song",
        export_format: ExportFormat = ExportFormat.WAV,
        *args,
        **kwargs,
    ):
        pass

    @abstractmethod
    @torch.no_grad()
    def reconstruct(
        self, reconstruction: torch.Tensor, original_data: torch.Tensor, *args, **kwargs
    ) -> tuple[float, dict[str, Any]]:
        pass

    @abstractmethod
    def get_optimizer(self, params: DictConfig) -> torch.optim.Optimizer:
        pass

    def layers_to_track(self) -> list[nn.Module]:
        return []

    def load_pretrained(self) -> None:
        pass
