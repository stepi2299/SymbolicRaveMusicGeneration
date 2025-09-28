from __future__ import annotations
from typing import Any

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

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.vae(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        data = batch

        # ---------------------
        # Train Discriminator
        # ---------------------

        opt_discriminator = self.optimizers()[0]
        opt_discriminator.zero_grad()
        real_labels = torch.ones(self.batch_size, 1).to(self.device)
        real = self.discriminator(data)
        real_loss = F.binary_cross_entropy(real, real_labels)

        fake_labels = torch.zeros(self.batch_size, 1).to(self.device)
        create_music_vector = self.vae(data, recv_in_music_form=True)[0]
        fake = self.discriminator(create_music_vector)
        fake_loss = F.binary_cross_entropy(fake, fake_labels)

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

        # ---------------------
        # Train Decoder
        # ---------------------

        opt_decoder = self.optimizers()[1]
        opt_decoder.zero_grad()
        reconstruction = self.vae(data, recv_in_music_form=True)[0]
        gen_labels = torch.ones(self.batch_size, 1).to(self.device)
        music_authenticity = self.discriminator(reconstruction)
        generated_loss = F.binary_cross_entropy(music_authenticity, gen_labels)
        self.manual_backward(generated_loss)
        opt_decoder.step()
        self.log(
            "train_decoder_loss",
            generated_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return generated_loss + loss_discriminator

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
