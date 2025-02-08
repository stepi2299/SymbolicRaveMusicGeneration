from __future__ import annotations
from typing import Any, Tuple

import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from src.models.vae import VAE
from src.dataset.symbolic_music_dataset import SymbolicMusicDataset
from src.utils import split_dataset
from src.dataset.dataloader import get_dataloader


class FirstStage(pl.LightningModule):
    def __init__(
        self,
        vae: VAE,
        symbolic_music_dataset: SymbolicMusicDataset,
        batch_size: int,
        num_workers: int = 0,
        train_split: float = 0.8,
        learning_rate: float = 1e-3,
        learning_patience: int = 5,
    ) -> None:
        super(FirstStage, self).__init__()
        self.vae = vae
        self.train_dataset, self.val_dataset = split_dataset(
            symbolic_music_dataset, train_split
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.learning_patience = learning_patience
        self.adapter = symbolic_music_dataset.music_adapter
        self.save_hyperparameters()

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
        reconstruction, mu, log_variance = self.vae(data)
        recon_loss, kl_div = self._compute_loss(data, reconstruction, mu, log_variance)
        loss = recon_loss + kl_div
        self.log(
            "train_recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train_kl_div", kl_div, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Any:
        data = batch
        reconstruction, mu, log_variance = self.vae(data)
        recon_loss, kl_div = self._compute_loss(data, reconstruction, mu, log_variance)
        loss = recon_loss + kl_div
        self.log(
            "val_recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val_kl_div", kl_div, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def train_dataloader(self):
        return get_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return get_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=self.learning_patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def generate(self, z: torch.Tensor) -> list[str]:
        tokenized_songs = self.vae.generate(z)
        songs = []
        for tokenized_song in tokenized_songs:
            songs.append(self.adapter.decode(tokenized_song))
        return songs
