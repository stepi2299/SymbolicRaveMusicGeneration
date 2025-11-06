from __future__ import annotations
from typing import Any

import torch
import pytorch_lightning as pl
from pathlib import Path

from src.models.vae import VAE
from src.dataset.symbolic_music_dataset import SymbolicMusicDataset
from src.utils import split_dataset
from src.dataset.dataloader import get_dataloader
from src.loss import BaseVaeLoss
from src.constants import ExportFormat


class FirstStage(pl.LightningModule):
    def __init__(
        self,
        vae: VAE,
        symbolic_music_dataset: SymbolicMusicDataset,
        loss_function: BaseVaeLoss,
        batch_size: int,
        num_workers: int = 0,
        train_split: float = 0.8,
        encoder_learning_rate: float = 1e-3,
        decoder_learning_rate: float = 1e-3,
        learning_patience: int = 5,
    ) -> None:
        super(FirstStage, self).__init__()
        self.vae = vae
        if len(symbolic_music_dataset) >=5:
            self.train_dataset, self.val_dataset = split_dataset(
                symbolic_music_dataset, train_split
            )
        else:
            self.train_dataset, self.val_dataset = symbolic_music_dataset, symbolic_music_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.encoder_learning_rate = encoder_learning_rate
        self.decoder_learning_rate = decoder_learning_rate
        self.learning_patience = learning_patience
        self.adapter = symbolic_music_dataset.music_adapter
        self.loss_function = loss_function
        self.save_hyperparameters()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.vae(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        data = batch
        reconstruction, mu, log_variance = self.vae(data)
        loss, loss_metadata = self.loss_function(
            org_data=data,
            reconstruction=reconstruction,
            mu=mu,
            logvar=log_variance,
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(loss_metadata, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Any:
        data = batch
        reconstruction, mu, log_variance = self.vae(data)
        loss, loss_metadata = self.loss_function(
            org_data=data,
            reconstruction=reconstruction,
            mu=mu,
            logvar=log_variance,
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(loss_metadata, on_step=False, on_epoch=True, prog_bar=True)
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
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.vae.encoder.parameters(),
                    "lr": self.encoder_learning_rate,
                },
                {
                    "params": self.vae.decoder.parameters(),
                    "lr": self.decoder_learning_rate,
                },
            ]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=self.learning_patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def generate(
        self,
        z: torch.Tensor,
        file_path: Path,
        export_format: ExportFormat = ExportFormat.WAV,
    ) -> list[str]:
        tokenized_songs = self.vae.generate(z)
        songs = []
        for tokenized_song in tokenized_songs:
            songs.append(
                self.adapter.decode(
                    tokenized_song, output_path=file_path, export_format=export_format
                )
            )
        return songs
