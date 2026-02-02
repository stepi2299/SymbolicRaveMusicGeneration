import torch
from torch import nn
import logging as lg
from pathlib import Path

from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.adapter import MusicAdapter
from src.constants import ExportFormat

logger = lg.getLogger(__name__)


class VAE(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        adapter: MusicAdapter,
        use_cache_generation: bool = False,
        strict_sampling: bool = True,
    ) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._adapter = adapter
        self._validate_if_latent_dimensions_match()
        self._orig_sample = None
        self._use_cache = use_cache_generation
        self._strict_sampling = strict_sampling
        self._temperature = 1.0

    def _validate_if_latent_dimensions_match(self) -> None:
        if self.encoder.latent_dim != self.decoder.latent_dim:
            raise ValueError(
                "Encoder and Decoder latent dimensions must match, "
                f"got {self.encoder.latent_dim} for encoder and {self.decoder.latent_dim} for decoder"
            )

    def forward(
        self, input_music: torch.Tensor, recv_in_music_form: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._orig_sample = input_music
        mu, log_variance = self.encoder(input_music)
        latent_vector = self.reparameterize(mu, log_variance)
        tensor_reconstruction = self.decoder(
            latent_vector,
            input_music,
            recv_in_music_form=recv_in_music_form,
            use_cache=False,
        )
        return tensor_reconstruction, mu, log_variance

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def set_generation_params(
        self,
        use_cache: bool = False,
        strict_sampling: bool = True,
        temperature: float = 1,
    ) -> None:
        logger.error(
            f"Setting generation params: use_cache={use_cache}, strict_sampling={strict_sampling}, temperature={temperature}"
        )
        self._use_cache = use_cache
        self._strict_sampling = strict_sampling
        self._temperature = temperature

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

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    @property
    def vocab_size(self) -> int:
        return self.encoder.vocab_size
