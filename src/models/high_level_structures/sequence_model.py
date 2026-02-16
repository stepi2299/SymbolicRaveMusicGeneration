import torch
from torch import nn
import logging as lg
from omegaconf import DictConfig
from src.adapter import MusicAdapter
from src.loss.base_loss import BaseLoss
from typing import Any
from pathlib import Path
from src.constants import ExportFormat
from src.loss.reconstruction_loss import ReconstructionLoss

from src.models.high_level_structures.base_high_level_model import BaseHighLevelModel

logger = lg.getLogger(__name__)


class SequenceModel(BaseHighLevelModel):
    def __init__(
        self,
        model_cfg: DictConfig,
        adapter: MusicAdapter,
        loss_function: BaseLoss,
        use_cache_generation: bool = False,
        strict_sampling: bool = True,
    ) -> None:
        super().__init__(
            adapter=adapter,
            loss_function=loss_function,
            use_cache_generation=use_cache_generation,
            strict_sampling=strict_sampling,
        )
        self.decoder = self._get_decoder(model_cfg, adapter)
        if not isinstance(self._loss_function, ReconstructionLoss):
            raise ValueError(
                f"Loss function for SequenceModel must be an instance of ReconstructionLoss, got {type(self._loss_function)}"
            )

    def forward(
        self, input_music: torch.Tensor, recv_in_music_form: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._original_sample = input_music
        tensor_reconstruction = self.decoder(
            input_sequence=input_music,
            recv_in_music_form=recv_in_music_form,
            use_cache=False,
        )
        return tensor_reconstruction

    def _lightning_step(
        self, data: torch.Tensor, global_step: int, train: bool
    ) -> tuple[torch.Tensor, float, dict[str, torch.Tensor]]:
        x_input = data[:, :-1]  # Input sequence excluding the last token
        x_target = data[:, 1:]  # Target sequence excluding the first token
        reconstruction = self(x_input)
        loss, loss_metadata = self._loss_function(
            org_data=x_target,
            reconstruction=reconstruction,
            global_step=global_step,
            train=train,
        )
        return reconstruction, loss, loss_metadata

    @torch.no_grad()
    def reconstruct(
        self, reconstruction: torch.Tensor, original_data: torch.Tensor
    ) -> tuple[float, dict[str, Any]]:
        target_data = original_data[:, 1:]  # Target sequence excluding the first token
        return self.decoder.reconstruct(reconstruction, target_data)

    @torch.no_grad()
    def generate(
        self,
        folder_path: Path,
        file_name: str = "generated_song",
        export_format: ExportFormat = ExportFormat.WAV,
        number_of_songs: int = 1,
        *args,
        **kwargs,
    ):
        folder_path.resolve(True)
        if not folder_path.is_dir():
            raise ValueError(
                f"{folder_path} is not a valid directory when trying to generate songs."
            )
        generated_songs = []
        for i in range(number_of_songs):
            try:
                output_path = folder_path / f"{file_name}_{i}"
                coded_song = self.decoder.generate(
                    use_cache=self._use_cache,
                    strict_sampling=self._strict_sampling,
                    temperature=self._temperature,
                )
                song_path = self._adapter.decode(coded_song, output_path, export_format)
                logger.info(f"Generated song for index: {i} at path: {song_path}")
                generated_songs.append(song_path)
            except Exception as e:
                logger.error(f"Error generating song for index: {i}: {e}")
        return generated_songs

    def get_optimizer(self, params: DictConfig) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.decoder.parameters(),
            lr=params.max_decoder_learning_rate,
            weight_decay=params.decoder_weight_decay,
        )
