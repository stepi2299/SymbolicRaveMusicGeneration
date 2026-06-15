from __future__ import annotations

import copy
import math
import logging as lg
from typing import Any

from omegaconf import DictConfig
from pathlib import Path

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset

from src.constants import ExportFormat
from src.models.high_level_structures import BaseHighLevelModel
from src.models.discriminator import Discriminator
from src.dataset.symbolic_music_dataset import SymbolicMusicDataset
from src.dataset.dataloader import get_dataloader
from src.utils import split_dataset

logger = lg.getLogger(__name__)


class SecondStage(pl.LightningModule):
    """GAN-based second-stage training.

    The encoder is frozen from the first stage.  A discriminator is trained
    alongside the main model's decoder to improve sample quality.

    Architecture levels
    -------------------
    * **VAE** – has an encoder whose weights are frozen; decoder is fine-tuned.
    * **SequenceModel** – encoder-free; the full decoder is fine-tuned.

    Notes
    -----
    Because music tokens are discrete, gradients from the discriminator cannot
    flow back through the ``argmax`` operation to the decoder.  The decoder is
    therefore trained primarily by its own reconstruction loss; the GAN term
    acts as an auxiliary signal that trains the discriminator to distinguish
    real from generated music, which in turn can be used for evaluation or
    future differentiable variants (e.g. Gumbel-Softmax).
    """

    def __init__(
        self,
        main_model: BaseHighLevelModel,
        discriminator: Discriminator,
        symbolic_music_dataset: SymbolicMusicDataset,
        batch_size: int,
        generator_optimizer_params: DictConfig,
        num_workers: int = 0,
        train_split: float = 0.8,
        discriminator_learning_rate: float = 1e-3,
        label_smoothing: float = 0.1,
        gan_loss_weight: float = 1.0,
        gumbel_tau: float = 1.0,
        train_augment: bool = False,
        val_augment: bool = False,
        monitor_metric: str = "val_loss",
        warmup_steps: int = 100,
    ) -> None:
        super(SecondStage, self).__init__()

        self.main_model = main_model
        self.main_model.freeze_encoder()  # no-op for encoder-free models
        self.discriminator = discriminator

        # ── Dataset split (identical pattern to FirstStage) ───────────────────
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
        self._monitor_metric = monitor_metric
        self._warmup_steps = warmup_steps
        self._generator_optimizer_params = generator_optimizer_params
        self._discriminator_learning_rate = discriminator_learning_rate
        self._label_smoothing = label_smoothing
        self._gan_loss_weight = gan_loss_weight
        self._gumbel_tau = gumbel_tau
        self._train_augment = train_augment
        self._val_augment = val_augment
        self._vocab_size = symbolic_music_dataset.music_adapter.vocab_size

        # Manual optimisation so we can update discriminator and decoder separately
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.main_model(*args, **kwargs)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        data, class_idx = batch
        x_target = data[:, 1:]  # all tokens except the first (target)

        opt_discriminator, opt_decoder = self.optimizers()
        scheduler = self.lr_schedulers()

        # =====================================================================
        # 0. FORWARD PASS (Run once for both phases to save compute)
        # =====================================================================
        # Let the main model calculate its standard reconstruction loss & logits
        reconstruction, model_loss, loss_metadata = self.main_model.training_step(
            batch, self.global_step
        )

        raw_recon = (
            reconstruction[0] if isinstance(reconstruction, tuple) else reconstruction
        )

        # Generate Gumbel-Softmax discrete-like continuous vectors (Batch, Seq_Len, Vocab)
        gumbel_fakes = F.gumbel_softmax(raw_recon, tau=self._gumbel_tau, hard=True)

        # Convert real targets to one-hot so they match the shape of gumbel_fakes
        real_one_hot = F.one_hot(x_target, num_classes=self._vocab_size).float()

        # Labels for BCE Loss
        real_labels = torch.full(
            (len(x_target), 1), 1.0 - self._label_smoothing, device=self.device
        )
        fake_labels = torch.zeros((len(x_target), 1), device=self.device)
        gen_labels = torch.ones((len(x_target), 1), device=self.device)

        # =====================================================================
        # 1. DISCRIMINATOR UPDATE
        # =====================================================================
        opt_discriminator.zero_grad()

        # Pass real data
        d_real_score = self.discriminator(real_one_hot)
        d_real_loss = F.binary_cross_entropy(d_real_score, real_labels)

        # Pass fake data (DETACHED so gradients don't flow back to the generator)
        d_fake_score = self.discriminator(gumbel_fakes.detach())
        d_fake_loss = F.binary_cross_entropy(d_fake_score, fake_labels)

        loss_discriminator = (d_real_loss + d_fake_loss) / 2

        self.manual_backward(loss_discriminator)
        opt_discriminator.step()

        # =====================================================================
        # 2. GENERATOR (DECODER) UPDATE
        # =====================================================================
        opt_decoder.zero_grad()

        # Pass fake data to discriminator again (NOT DETACHED, to trick the discriminator)
        # We calculate how well the generator fooled the discriminator
        g_fake_score = self.discriminator(gumbel_fakes)
        gan_loss = F.binary_cross_entropy(g_fake_score, gen_labels)

        # Combine standard VAE/Sequence (Main model) loss with the Adversarial GAN loss
        total_decoder_loss = model_loss + (self._gan_loss_weight * gan_loss)

        self.manual_backward(total_decoder_loss)
        opt_decoder.step()

        if scheduler is not None:
            scheduler.step()

        # =====================================================================
        # 3. LOGGING
        # =====================================================================
        with torch.no_grad():
            d_real_acc = (d_real_score >= 0.5).float().mean()
            d_fake_acc = (d_fake_score < 0.5).float().mean()

        log_dict = {
            "train_disc_loss": loss_discriminator,
            "train_disc_real_acc": d_real_acc,
            "train_disc_fake_acc": d_fake_acc,
            "train_gen_total_loss": total_decoder_loss,
            "train_gen_base_loss": model_loss,
            "train_gen_gan_loss": gan_loss,
        }
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(loss_metadata, on_step=False, on_epoch=True, prog_bar=False)

        return total_decoder_loss + loss_discriminator

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
        # Discriminator: constant LR (no warmup, no decay).
        # It learns from scratch and must stay responsive throughout training
        # to provide a useful adversarial signal to the decoder.
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self._discriminator_learning_rate,
        )

        # Decoder: re-uses the main model's optimizer (separate LRs for
        # encoder/decoder in VAE, single AdamW for SequenceModel).
        # Warmup + cosine decay applies here because we are fine-tuning a
        # pre-trained model and want a gentle, stable learning rate schedule.
        decoder_optimizer = self.main_model.get_optimizer(
            self._generator_optimizer_params
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self._warmup_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        decoder_scheduler = LambdaLR(decoder_optimizer, lr_lambda)

        return (
            [discriminator_optimizer, decoder_optimizer],
            # Only one scheduler — for the decoder optimizer (index 1).
            [{"scheduler": decoder_scheduler, "interval": "step", "frequency": 1}],
        )

    def generate(
        self,
        num_samples: int,
        folder_path: Path,
        file_name: str,
        export_format: ExportFormat,
        generation_params: dict | None = None,
    ) -> list[Path]:
        gen_params = generation_params or {}
        self.main_model.set_generation_params(**gen_params)
        return self.main_model.generate(
            folder_path=folder_path,
            file_name=file_name,
            export_format=export_format,
            number_of_songs=num_samples,
            device=self.device,
        )
