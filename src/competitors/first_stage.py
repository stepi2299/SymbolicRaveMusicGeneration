from __future__ import annotations
from typing import Any
import copy
from omegaconf import DictConfig
import logging as lg

import torch
import math
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset
from pathlib import Path

from src.models.high_level_structures import BaseHighLevelModel, VAE
from src.dataset.symbolic_music_dataset import SymbolicMusicDataset
from src.utils import split_dataset
from src.dataset.dataloader import get_dataloader
from src.constants import ExportFormat

logger = lg.getLogger(__name__)


class FirstStage(pl.LightningModule):
    def __init__(
        self,
        main_model: BaseHighLevelModel,
        symbolic_music_dataset: SymbolicMusicDataset,
        batch_size: int,
        optimizer_params: DictConfig,
        num_workers: int = 0,
        train_split: float = 0.8,
        train_augment: bool = False,
        val_augment: bool = False,
        single_data_training: bool = False,
        monitor_metric: str = "val_loss",
        warmup_steps=100,
    ) -> None:
        super(FirstStage, self).__init__()
        self.main_model = main_model
        if len(symbolic_music_dataset) >= 5:
            train_subset, val_subset = split_dataset(
                symbolic_music_dataset, train_split
            )
            train_dataset_copy = copy.deepcopy(train_subset.dataset)
            train_dataset_copy.turn_augmentation(augment=train_augment)
            val_subset.dataset.turn_augmentation(augment=val_augment)
            self.train_dataset = Subset(train_dataset_copy, train_subset.indices)
            self.val_dataset = val_subset
        else:
            self.train_dataset, self.val_dataset = (
                symbolic_music_dataset,
                copy.deepcopy(symbolic_music_dataset),
            )
            self.val_dataset.turn_augmentation(augment=val_augment)
            self.train_dataset.turn_augmentation(augment=train_augment)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.adapter = symbolic_music_dataset.music_adapter
        self._single_data_training = single_data_training
        self._monitor_metric = monitor_metric
        self._warmup_steps = warmup_steps
        self._optimizer_params = optimizer_params
        self._train_augment = train_augment
        self._val_augment = val_augment
        self.save_hyperparameters()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        reconstruction, loss, loss_metadata = self.main_model.training_step(
            batch, self.global_step
        )
        if self._single_data_training:
            reconstruction_success, metadata = self.main_model.reconstruct(
                reconstruction=reconstruction, original_data=batch
            )
            if metadata:
                self.log_dict(metadata, on_step=False, on_epoch=True, prog_bar=True)
            self.log(
                "reconstruct_success",
                reconstruction_success,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(loss_metadata, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        reconstruction, loss, loss_metadata = self.main_model.validation_step(
            batch, self.global_step
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
        optimizer = self.main_model.get_optimizer(self._optimizer_params)

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
        num_samples: int,
        folder_path: Path,
        file_name: str,
        export_format: ExportFormat,
        generation_params: dict | None = None,
    ) -> list[Path]:
        generation_params = generation_params or {}
        self.main_model.set_generation_params(**generation_params)
        if isinstance(self.main_model, VAE):
            z = torch.randn(num_samples, self.main_model.latent_dim).to(self.device)
        else:
            z = None
        return self.main_model.generate(
            latent_vector=z,
            folder_path=folder_path,
            file_name=file_name,
            export_format=export_format,
        )

    def on_before_optimizer_step(self, optimizer):
        if logger.getEffectiveLevel() <= lg.DEBUG:
            global_norms = grad_norm(self, norm_type=2)
            if "grad_2.0_norm_total" in global_norms:
                self.log("grad_global/total_norm", global_norms["grad_2.0_norm_total"])

            layers_to_track = self.main_model.layers_to_track()

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
