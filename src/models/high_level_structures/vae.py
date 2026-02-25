from typing import Any

import torch
import logging as lg
from pathlib import Path

from omegaconf import DictConfig
from torch import nn

from src.adapter import MusicAdapter
from src.constants import ExportFormat
from src.models.high_level_structures.base_high_level_model import BaseHighLevelModel
from src.loss.base_loss import BaseLoss
from src.models.encoder import get_encoder
from src.models.sequence_decoder import get_decoder
from src.models.decoder_wrappers import get_decoder_wrapper

logger = lg.getLogger(__name__)


class VAE(BaseHighLevelModel):
    def __init__(
        self,
        model_cfg: DictConfig,
        adapter: MusicAdapter,
        loss_function: BaseLoss,
        num_classes: int,
        use_cache_generation: bool = False,
        strict_sampling: bool = True,
    ) -> None:
        super(VAE, self).__init__(
            adapter=adapter,
            loss_function=loss_function,
            use_cache_generation=use_cache_generation,
            strict_sampling=strict_sampling,
        )
        self._seq_encoder = get_encoder(
            model_cfg.encoder, adapter, latent_dim=model_cfg.latent_dim
        )
        self._seq_decoder = get_decoder(
            model_cfg.decoder,
            adapter,
            latent_dim=model_cfg.latent_dim,
            num_classes=num_classes,
        )
        self._decoder_wrapper = get_decoder_wrapper(
            model_cfg.decoder_wrapper.type,
            self._seq_decoder,
            model_cfg.decoder_wrapper.params,
        )
        self.latent_dim = self._seq_encoder.latent_dim
        self._validate_if_latent_dimensions_match()

    def _validate_if_latent_dimensions_match(self) -> None:
        if self._seq_encoder.latent_dim != self._seq_decoder.latent_dim:
            raise ValueError(
                "Encoder and Decoder latent dimensions must match, "
                f"got {self._seq_encoder.latent_dim} for encoder and {self._seq_decoder.latent_dim} for decoder"
            )

    def forward(
        self, batch_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._original_sample = batch_data
        mu, log_variance = self._seq_encoder(batch_data)
        latent_vector = self._reparameterize(mu, log_variance)
        tensor_reconstruction = self._decoder_wrapper(
            batch_data=batch_data,
            latent_vector=latent_vector,
            use_cache=False,
        )
        return tensor_reconstruction, mu, log_variance

    def _lightning_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], global_step: int, train: bool
    ) -> tuple[torch.Tensor, float, dict[str, Any]]:
        data, class_idx = batch
        x_input = data[:, :-1]  # Input sequence excluding the last token
        x_target = data[:, 1:]  # Target sequence excluding the first token
        reconstruction, mu, log_variance = self([x_input, class_idx])
        loss, loss_metadata = self._loss_function(
            org_data=x_target,
            reconstruction=reconstruction,
            global_step=global_step,
            train=train,
            mu=mu,
            logvar=log_variance,
        )
        return reconstruction, loss, loss_metadata

    def get_optimizer(self, params: DictConfig) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [
                {
                    "params": self._seq_encoder.parameters(),
                    "lr": params.max_encoder_learning_rate,  # Base LR for the encoder
                    "weight_decay": params.encoder_weight_decay,
                },
                {
                    "params": self._decoder_wrapper.parameters(),
                    "lr": params.max_decoder_learning_rate,  # Base LR for the decoder
                    "weight_decay": params.decoder_weight_decay,
                },
            ]
        )

    @staticmethod
    def _reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _freeze_encoder(self) -> None:
        for param in self._seq_encoder.parameters():
            param.requires_grad = False

    def _unfreeze_encoder(self) -> None:
        for param in self._seq_encoder.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def generate(
        self,
        folder_path: Path,
        number_of_songs: int,
        device: torch.device,
        file_name: str = "generated_song",
        export_format: ExportFormat = ExportFormat.WAV,
        use_cache: bool = True,
        **kwargs,
    ) -> list[Path]:
        self._decoder_wrapper.eval()
        latent_vector = torch.randn(number_of_songs, self.latent_dim).to(device)
        folder_path.resolve(True)
        if not folder_path.is_dir():
            raise ValueError(
                f"{folder_path} is not a valid directory when trying to generate songs."
            )
        generated_songs = []
        for i, single_condition_latent in enumerate(latent_vector):
            try:
                output_path = folder_path / f"{file_name}_{i}"
                coded_song = self._decoder_wrapper.generate(
                    latent_vector=single_condition_latent.unsqueeze(0),
                    use_cache=self._use_cache,
                    strict_sampling=self._strict_sampling,
                    temperature=self._temperature,
                )
                song_path = self._adapter.decode(coded_song, output_path, export_format)
                logger.info(
                    f"Generated song for latent vector for index: {i} at path: {song_path}"
                )
                generated_songs.append(song_path)
            except Exception as e:
                logger.error(
                    f"Error generating song for latent vector for index: {i}: {e}"
                )
        return generated_songs

    def load_pretrained(self) -> None:
        self._seq_encoder.load_pretrained()

    def layers_to_track(self) -> list[nn.Module]:
        return self._seq_decoder.get_layers_to_track()
