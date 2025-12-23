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
        single_data_training: bool = False,
    ) -> None:
        super(FirstStage, self).__init__()
        self.vae = vae
        if len(symbolic_music_dataset) >= 5:
            self.train_dataset, self.val_dataset = split_dataset(
                symbolic_music_dataset, train_split
            )
        else:
            self.train_dataset, self.val_dataset = (
                symbolic_music_dataset,
                symbolic_music_dataset,
            )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.encoder_learning_rate = encoder_learning_rate
        self.decoder_learning_rate = decoder_learning_rate
        self.learning_patience = learning_patience
        self.adapter = symbolic_music_dataset.music_adapter
        self.loss_function = loss_function
        self._single_data_training = single_data_training
        self.save_hyperparameters()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.vae(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        data = batch
        x_input = data[:, :-1]  # Input sequence excluding the last token
        x_target = data[:, 1:]  # Target sequence excluding the first token
        reconstruction, mu, log_variance = self.vae(x_input)

        loss, loss_metadata = self.loss_function(
            org_data=x_target,
            reconstruction=reconstruction,
            mu=mu,
            logvar=log_variance,
            global_step=self.global_step,
        )
        if self._single_data_training:
            token_reconstruction = reconstruction.argmax(dim=-1)
            correctness_reconstruction = (token_reconstruction == x_target).float().mean().item() * 100
            self.log(
                "reconstruction_success",
                correctness_reconstruction,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(loss_metadata, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Any:
        data = batch
        x_input = data[:, :-1]  # Input sequence excluding the last token
        x_target = data[:, 1:]  # Target sequence excluding the first token
        reconstruction, mu, log_variance = self.vae(x_input)
        loss, loss_metadata = self.loss_function(
            org_data=x_target,
            reconstruction=reconstruction,
            mu=mu,
            logvar=log_variance,
            global_step=self.global_step,
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
        optimizer = torch.optim.AdamW(
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
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, "min", patience=self.learning_patience
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10000,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def generate(
        self,
        z: torch.Tensor,
        folder_path: Path,
        file_name: str = "generated_song",
        export_format: ExportFormat = ExportFormat.WAV,
    ) -> list[str]:
        tokenized_songs = self.vae.generate(z)
        folder_path.resolve(True)
        if not folder_path.is_dir():
            raise ValueError(
                f"{folder_path} is not a valid directory when trying to generate songs."
            )
        songs = []
        for i, tokenized_song in enumerate(tokenized_songs):
            output_path = folder_path / f"{file_name}_{i}"
            songs.append(
                self.adapter.decode(
                    tokenized_song, output_path=output_path, export_format=export_format
                )
            )
        return songs
