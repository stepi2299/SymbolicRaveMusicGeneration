import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, log_variance = self.encoder(x)
        latent_vector = self.reparameterize(mu, log_variance)
        reconstruction = self.decoder(latent_vector)
        return reconstruction, mu, log_variance

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate(self, latent_vector: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_vector)