from __future__ import annotations
from typing import Any

import torch
import math
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path

from src.models.vae import VAE
from src.dataset.symbolic_music_dataset import SymbolicMusicDataset
from src.utils import split_dataset
from src.dataset.dataloader import get_dataloader
from src.loss import BaseVaeLoss
from src.constants import ExportFormat
from src.constants import CP_WORD_INDEXES


class FirstStage(pl.LightningModule):
    def __init__(
        self,
        vae: VAE,
        symbolic_music_dataset: SymbolicMusicDataset,
        loss_function: BaseVaeLoss,
        batch_size: int,
        num_workers: int = 0,
        train_split: float = 0.8,
        max_encoder_learning_rate: float = 1e-3,
        max_decoder_learning_rate: float = 1e-3,
        single_data_training: bool = False,
        monitor_metric: str = "val_loss",
        warmup_steps=100,
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
        self.max_encoder_learning_rate = max_encoder_learning_rate
        self.max_decoder_learning_rate = max_decoder_learning_rate
        self.adapter = symbolic_music_dataset.music_adapter
        self.loss_function = loss_function
        self._single_data_training = single_data_training
        self._monitor_metric = monitor_metric
        self._warmup_steps = warmup_steps
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
            train=True,
        )
        if self._single_data_training:
            if isinstance(reconstruction, torch.Tensor):
                self._reconstruct_single_dim_token(
                    reconstruction=reconstruction, x_target=x_target
                )
            else:
                self._reconstruct_cp_tensor(
                    reconstruction=reconstruction, x_target=x_target
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
            train=False,
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
        # 1. Define the optimizer with separate parameter groups for Encoder and Decoder
        # This allows setting different maximum learning rates for each part of the VAE.
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.vae.encoder.parameters(),
                    "lr": self.max_encoder_learning_rate,  # Base LR for the encoder
                },
                {
                    "params": self.vae.decoder.parameters(),
                    "lr": self.max_decoder_learning_rate,  # Base LR for the decoder
                },
            ]
        )

        # 2. Calculate total training steps
        # Lightning automatically calculates this based on max_epochs, batch_size, and accumulation.
        total_steps = self.trainer.estimated_stepping_batches

        # 3. Define warmup steps
        # Standard practice is to warm up for a fraction of training (e.g., 5-10%) or a fixed number (e.g., 2000).
        # Since you are doing a sanity check on one song, keep it small (e.g., 50-100 steps).
        warmup_steps = self._warmup_steps

        # 4. Define the Scheduler Lambda function (Linear Warmup + Cosine Decay)
        # This function returns a *multiplier* for the base LR (from 0.0 to 1.0).
        def lr_lambda(current_step: int):
            # Phase 1: Linear Warmup
            # Linearly increase LR from 0 to the base LR defined in the optimizer.
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            # Phase 2: Cosine Decay
            # After warmup, follow a cosine curve down to 0.
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )

            # Ensure we don't go below 0.0
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        # 5. Initialize the scheduler using the lambda function
        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # CRITICAL: Scheduler must step every batch to create the smooth cosine curve.
                "interval": "step",
                "frequency": 1,
            },
            "monitor": self._monitor_metric,
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

    def on_before_optimizer_step(self, optimizer):
        global_norms = grad_norm(self, norm_type=2)
        if "grad_2.0_norm_total" in global_norms:
            self.log("grad_global/total_norm", global_norms["grad_2.0_norm_total"])

        layers_to_track = self.vae.decoder.get_layers_to_track()

        for i, layer in enumerate(layers_to_track):
            grads = []
            for param in layer.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))

            if grads:
                layer_grads = torch.cat(grads)

                l_norm = torch.norm(layer_grads, p=2)
                l_max = layer_grads.max()
                l_min = layer_grads.min()
                l_std = layer_grads.std()

                self.log(
                    f"grad_layer_{i}/norm",
                    l_norm,
                    prog_bar=True,
                    on_epoch=True,
                    on_step=False,
                )
                self.log(
                    f"grad_layer_{i}/max",
                    l_max,
                    prog_bar=True,
                    on_epoch=True,
                    on_step=False,
                )
                self.log(
                    f"grad_layer_{i}/min",
                    l_min,
                    prog_bar=True,
                    on_epoch=True,
                    on_step=False,
                )
                self.log(
                    f"grad_layer_{i}/std",
                    l_std,
                    prog_bar=True,
                    on_epoch=True,
                    on_step=False,
                )

    def _reconstruct_single_dim_token(
        self, reconstruction: torch.Tensor, x_target: torch.Tensor
    ) -> None:
        token_reconstruction = reconstruction.argmax(dim=-1)
        correctness_reconstruction = (
            token_reconstruction == x_target
        ).float().mean().item() * 100
        self.log(
            "reconstruction_success",
            correctness_reconstruction,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def _reconstruct_cp_tensor(
        self, reconstruction: dict[str, torch.Tensor], x_target: torch.Tensor
    ) -> None:
        total_accuracy = 0.0
        for attr_name, logits in reconstruction.items():
            attr_index = CP_WORD_INDEXES[attr_name]
            token_reconstruction = logits.argmax(dim=-1)
            correctness_reconstruction = (
                token_reconstruction == x_target[:, :, attr_index]
            ).float().mean().item() * 100
            total_accuracy += correctness_reconstruction
            self.log(
                f"reconstruction_success_{attr_name}",
                correctness_reconstruction,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        avg_accuracy = total_accuracy / len(reconstruction)
        self.log(
            "reconstruction_success_avg",
            avg_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
