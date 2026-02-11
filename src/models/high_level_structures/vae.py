from typing import Any

import torch
import logging as lg
from pathlib import Path

from omegaconf import DictConfig
from torch import nn

from src.models.encoder import Encoder, encoder_mapper
from src.models.decoder import Decoder, decoder_mapper
from src.adapter import MusicAdapter
from src.constants import ExportFormat
from src.models.high_level_structures.base_high_level_model import BaseHighLevelModel
from src.loss.base_loss import BaseLoss

logger = lg.getLogger(__name__)


class VAE(BaseHighLevelModel):
    def __init__(
        self,
        model_cfg: DictConfig,
        adapter: MusicAdapter,
        loss_function: BaseLoss,
        use_cache_generation: bool = False,
        strict_sampling: bool = True,
    ) -> None:
        super(VAE, self).__init__(
            adapter=adapter,
            loss_function=loss_function,
            use_cache_generation=use_cache_generation,
            strict_sampling=strict_sampling,
        )
        self.encoder = self._get_encoder(model_cfg, adapter)
        self.decoder = self._get_decoder(model_cfg, adapter)
        self.latent_dim = self.encoder.latent_dim
        self._validate_if_latent_dimensions_match()

    def _validate_if_latent_dimensions_match(self) -> None:
        if self.encoder.latent_dim != self.decoder.latent_dim:
            raise ValueError(
                "Encoder and Decoder latent dimensions must match, "
                f"got {self.encoder.latent_dim} for encoder and {self.decoder.latent_dim} for decoder"
            )

    @staticmethod
    def _get_encoder(model_cfg: DictConfig, adapter: MusicAdapter) -> Encoder:
        return encoder_mapper(model_cfg.encoder.name)(
            latent_dim=model_cfg.latent_dim,
            vocab_size=adapter.vocab_size,
            sequence_length=adapter.sequence_length,
            patch_size=adapter.patch_size,
            **model_cfg.encoder.params,
        )

    @staticmethod
    def _get_decoder(model_cfg: DictConfig, adapter: MusicAdapter) -> Decoder:
        return decoder_mapper(model_cfg.decoder.name)(
            latent_dim=model_cfg.latent_dim,
            vocab_size=adapter.vocab_size,
            sequence_length=adapter.sequence_length,
            patch_size=adapter.patch_size,
            pad_token=adapter.pad_token_id,
            bos_token=adapter.bos_token_id,
            eos_token=adapter.eos_token_id,
            **model_cfg.decoder.params,
        )

    def forward(
        self, input_music: torch.Tensor, recv_in_music_form: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._original_sample = input_music
        mu, log_variance = self.encoder(input_music)
        latent_vector = self._reparameterize(mu, log_variance)
        tensor_reconstruction = self.decoder(
            latent_vector,
            input_music,
            recv_in_music_form=recv_in_music_form,
            use_cache=False,
        )
        return tensor_reconstruction, mu, log_variance

    def _lightning_step(
        self, data: torch.Tensor, global_step: int, train: bool
    ) -> tuple[torch.Tensor, float, dict[str, Any]]:
        x_input = data[:, :-1]  # Input sequence excluding the last token
        x_target = data[:, 1:]  # Target sequence excluding the first token
        reconstruction, mu, log_variance = self(x_input)
        loss, loss_metadata = self._loss_function(
            org_data=x_target,
            reconstruction=reconstruction,
            global_step=global_step,
            train=train,
            mu=mu,
            logvar=log_variance,
        )
        return reconstruction, loss, loss_metadata

    def training_step(
        self, data: torch.Tensor, global_step: int
    ) -> tuple[torch.Tensor, float, dict[str, torch.Tensor]]:
        return self._lightning_step(data, global_step, train=True)

    def validation_step(
        self, data: torch.Tensor, global_step: int
    ) -> tuple[torch.Tensor, float, dict[str, torch.Tensor]]:
        return self._lightning_step(data, global_step, train=False)

    def reconstruct(
        self, reconstruction: torch.Tensor, original_data: torch.Tensor, *args, **kwargs
    ) -> tuple[float, dict[str, Any]]:
        target_data = original_data[:, 1:]  # Target sequence excluding the first token
        return self.decoder.reconstruct(reconstruction, target_data)

    def get_optimizer(self, params: DictConfig) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [
                {
                    "params": self.encoder.parameters(),
                    "lr": params.max_encoder_learning_rate,  # Base LR for the encoder
                    "weight_decay": params.encoder_weight_decay,
                },
                {
                    "params": self.decoder.parameters(),
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
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def generate(
        self,
        latent_vector: torch.Tensor,
        folder_path: Path,
        file_name: str = "generated_song",
        export_format: ExportFormat = ExportFormat.WAV,
    ) -> list[Path]:
        self.decoder.eval()
        if latent_vector.shape[1] != self.latent_dim:
            raise ValueError(
                f"Latent vector must have shape (batch_size, {self.latent_dim}), got {latent_vector.shape}"
            )
        folder_path.resolve(True)
        if not folder_path.is_dir():
            raise ValueError(
                f"{folder_path} is not a valid directory when trying to generate songs."
            )
        generated_songs = []
        for i, single_condition_latent in enumerate(latent_vector):
            try:
                output_path = folder_path / f"{file_name}_{i}"
                coded_song = self.decoder.generate(
                    single_condition_latent.unsqueeze(0),
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
        self.encoder.load_pretrained()

    def layers_to_track(self) -> list[nn.Module]:
        return self.decoder.get_layers_to_track()
