import torch
from torch import nn
from symusic import Score

from src.models.encoder import Encoder
from src.models.decoder import Decoder


class VAE(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._validate_if_latent_dimensions_match()

    def _validate_if_latent_dimensions_match(self) -> None:
        if self.encoder.latent_dim != self.decoder.latent_dim:
            raise ValueError(
                "Encoder and Decoder latent dimensions must match, "
                f"got {self.encoder.latent_dim} for encoder and {self.decoder.latent_dim} for decoder"
            )

    def forward(
        self, input_music: list[list[int]] | list[int]
    ) -> tuple[Score, torch.Tensor, torch.Tensor]:
        mu, log_variance = self.encoder(input_music)
        latent_vector = self.reparameterize(mu, log_variance)
        tensor_reconstruction = self.decoder(latent_vector)
        reconstruction = self.tokenizer.detokenize(tensor_reconstruction)
        return reconstruction, mu, log_variance

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    @torch.no_grad()
    def generate(self, latent_vector: torch.Tensor) -> torch.Tensor:
        self.decoder.eval()
        return self.decoder(latent_vector)
