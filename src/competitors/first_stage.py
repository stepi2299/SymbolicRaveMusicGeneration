from typing import Any
import torch
import pytorch_lightning as pl

from src.loss.fmd import FmdLoss
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
        self.fmd_loss = FmdLoss()
        self.train_dataset, self.val_dataset = split_dataset(
            symbolic_music_dataset, train_split
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.learning_patience = learning_patience

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.vae(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        print("AAAAAAAAAAaa", batch)
        data = batch
        reconstruction, mu, log_variance = self.vae(data)
        # kld_loss = self.kld_loss(mu, log_variance)
        # fmd_loss = self.fmd_loss(reconstruction, y)
        # TODO Compare normal Loss calculation for vae with FMD
        raise NotImplementedError("Validation step not implemented")

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        data = batch
        reconstruction, mu, log_variance = self.vae(data)
        kld_loss = self.kld_loss(mu, log_variance)
        raise NotImplementedError("Validation step not implemented")

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
