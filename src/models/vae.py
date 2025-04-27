import torch
from torch import nn

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
        self, input_music: torch.Tensor, recv_in_music_form: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_variance = self.encoder(input_music)
        latent_vector = self.reparameterize(mu, log_variance)
        tensor_reconstruction = self.decoder(latent_vector, recv_in_music_form)
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
    def generate(self, latent_vector: torch.Tensor) -> list[list[list[int]]]:
        self.decoder.eval()
        if latent_vector.shape[1] != self.latent_dim:
            raise ValueError(
                f"Latent vector must have shape (batch_size, {self.latent_dim}), got {latent_vector.shape}"
            )
        reconstruction = self.decoder(latent_vector)
        reconstruction = reconstruction.argmax(dim=-1)
        reconstruction = reconstruction.cpu().numpy().tolist()
        return reconstruction

    def load_pretrained(self) -> None:
        self.encoder.load_pretrained()

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    @property
    def vocab_size(self) -> int:
        return self.encoder.vocab_size
