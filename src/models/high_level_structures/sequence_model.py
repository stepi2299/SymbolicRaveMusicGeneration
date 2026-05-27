import random

import torch
import logging as lg
from omegaconf import DictConfig
from src.adapter import MusicAdapter
from src.loss.base_loss import BaseLoss
from typing import Any
from pathlib import Path
from src.constants import ExportFormat

from src.models.high_level_structures.base_high_level_model import BaseHighLevelModel
from src.models.sequence_decoder import get_decoder
from src.models.decoder_wrappers import get_decoder_wrapper

logger = lg.getLogger(__name__)


class SequenceModel(BaseHighLevelModel):
    def __init__(
        self,
        model_cfg: DictConfig,
        adapter: MusicAdapter,
        loss_function: BaseLoss,
        num_classes: int,
        use_cache_generation: bool = False,
        strict_sampling: bool = True,
    ) -> None:
        super().__init__(
            adapter=adapter,
            loss_function=loss_function,
            use_cache_generation=use_cache_generation,
            strict_sampling=strict_sampling,
        )
        self._seq_decoder = get_decoder(
            model_cfg.decoder, adapter, latent_dim=None, num_classes=num_classes
        )
        self._decoder_wrapper = get_decoder_wrapper(
            model_cfg.decoder_wrapper.type,
            self._seq_decoder,
            model_cfg.decoder_wrapper.params,
        )

    def forward(
        self, batch_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._original_sample = batch_data
        tensor_reconstruction = self._decoder_wrapper(
            batch_data=batch_data,
            use_cache=False,
        )
        return tensor_reconstruction

    def _lightning_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], global_step: int, train: bool
    ) -> tuple[torch.Tensor, float, dict[str, torch.Tensor]]:
        data, class_idx = batch
        x_input = data[:, :-1]  # Input sequence excluding the last token
        x_target = data[:, 1:]  # Target sequence excluding the first token
        reconstruction = self([x_input, class_idx])
        loss, loss_metadata = self._loss_function(
            org_data=x_target,
            reconstruction=reconstruction,
            global_step=global_step,
            train=train,
        )
        return reconstruction, loss, loss_metadata

    @torch.no_grad()
    def generate(
        self,
        folder_path: Path,
        number_of_songs: int,
        device: torch.device,
        file_name: str = "generated_song",
        export_format: ExportFormat = ExportFormat.WAV,
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
                logger.info(f"Generating song for index: {i}")
                class_idx = random.randint(0, self._seq_decoder.num_classes - 1)
                class_idx_tensor = torch.tensor([class_idx], dtype=torch.long).to(
                    device
                )
                output_path = folder_path / f"{file_name}_{i}"
                logger.debug(
                    "Output path: %s and class embedding is ready", output_path
                )
                coded_song = self._decoder_wrapper.generate(
                    use_cache=self._use_cache,
                    strict_sampling=self._strict_sampling,
                    temperature=self._temperature,
                    class_idx=class_idx_tensor,
                )
                song_path = self._adapter.decode(coded_song, output_path, export_format)
                logger.info(
                    f"Successfully generated song for index: {i} at path: {song_path}"
                )
                generated_songs.append(song_path)
            except Exception as e:
                logger.error(f"Error generating song for index: {i}: {e}")
        logger.info(
            "Successfully generated songs: %d out of planned %s",
            len(generated_songs),
            number_of_songs,
        )
        return generated_songs

    def get_optimizer(self, params: DictConfig) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self._decoder_wrapper.parameters(),
            lr=params.max_decoder_learning_rate,
            weight_decay=params.decoder_weight_decay,
        )
