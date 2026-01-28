import torch
from torch import nn
import logging as lg

from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.adapter import MusicAdapter

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

    @torch.no_grad()
    def generate(self, latent_vector: torch.Tensor) -> list[list[int]]:
        self.decoder.eval()
        if latent_vector.shape[1] != self.latent_dim:
            raise ValueError(
                f"Latent vector must have shape (batch_size, {self.latent_dim}), got {latent_vector.shape}"
            )
        songs = []
        for i, single_condition_latent in enumerate(latent_vector):
            try:
                song = self.decoder.generate(
                    single_condition_latent.unsqueeze(0), self._strict_sampling
                )
                songs.append(song)
            except Exception as e:
                logger.error(
                    f"Error generating song for latent vector for index: {i}: {e}"
                )
        return songs

    def load_pretrained(self) -> None:
        self.encoder.load_pretrained()

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    @property
    def vocab_size(self) -> int:
        return self.encoder.vocab_size
