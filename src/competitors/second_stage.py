from __future__ import annotations
from typing import Any, Tuple

import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from src.models.vae import VAE
from src.models.discriminator import Discriminator
from src.dataset.symbolic_music_dataset import SymbolicMusicDataset
from src.dataset.dataloader import get_dataloader

class SecondStage(pl.LightningModule):
    def __init__(
        self,
        vae: VAE,
        discriminator: Discriminator,
        symbolic_music_dataset: SymbolicMusicDataset,
        batch_size: int,
        num_workers: int = 0,
        discriminator_learning_rate: float = 1e-3,
        decoder_learning_rate: float = 1e-3,
    ) -> None:
        super(SecondStage, self).__init__()
        self.vae = vae
        self.vae.freeze_encoder()
        self.discriminator = discriminator
        self.symbolic_music_dataset = symbolic_music_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.discriminator_learning_rate = discriminator_learning_rate
        self.decoder_learning_rate = decoder_learning_rate
        self.adapter = symbolic_music_dataset.music_adapter
        self.save_hyperparameters()
        self.automatic_optimization = False  # Disable automatic optimization

    def _compute_loss(
        self,
        data: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        recon_loss = F.cross_entropy(
            reconstruction.view(-1, self.vae.token_size),  # Reshape to (batch*seq_len)
            data.view(-1),  # Reshape to (batch*seq_len)
        )
        kl_div = -0.5 * torch.sum(
            1 + log_variance - mu.pow(2) - log_variance.exp(), dim=1
        )
        kl_div = torch.mean(kl_div)
        return recon_loss, kl_div

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.vae(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        data = batch

        # Discriminator step
        opt_discriminator = self.optimizers()[0]
        opt_discriminator.zero_grad()
        z = self.vae.reparameterize(*self.vae.encoder(data))
        real = self.discriminator(z)
        fake = self.discriminator(torch.randn_like(z))
        fake_loss = F.binary_cross_entropy(fake, torch.zeros_like(fake))
        real_loss = F.binary_cross_entropy(real, torch.ones_like(real))
        loss_discriminator = fake_loss + real_loss
        self.manual_backward(loss_discriminator)
        opt_discriminator.step()
        self.log(
            "train_discriminator_loss",
            loss_discriminator,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Decoder step
        opt_decoder = self.optimizers()[1]
        opt_decoder.zero_grad()
        reconstruction, mu, log_variance = self.vae(data)
        recon_loss, kl_div = self._compute_loss(data, reconstruction, mu, log_variance)
        loss_decoder = recon_loss + kl_div
        self.manual_backward(loss_decoder)
        opt_decoder.step()
        self.log(
            "train_recon_loss",
            recon_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train_kl_div", kl_div, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss_decoder, on_step=True, on_epoch=True, prog_bar=True)

        return loss_discriminator + loss_decoder

    def train_dataloader(self):
        return get_dataloader(
            self.symbolic_music_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def configure_optimizers(self):
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.discriminator_learning_rate
        )
        decoder_optimizer = torch.optim.Adam(
            self.vae.decoder.parameters(), lr=self.decoder_learning_rate
        )
        return [discriminator_optimizer, decoder_optimizer], []

    def generate(self, z: torch.Tensor) -> list[str]:
        tokenized_songs = self.vae.generate(z)
        songs = []
        for tokenized_song in tokenized_songs:
            songs.append(self.adapter.decode(tokenized_song))
        return songs
