from typing import Any
import pytorch_lightning as pl

from src.loss.fmd import FmdLoss

from src.models.vae import VAE

class FirstStageTrainer(pl.LightningModule):
    def __init__(self, vae: VAE, lambda_kld: float = 0.1, lambda_adv: float = 0.1) -> None:
        super(FirstStageTrainer, self).__init__()
        self.vae = vae
        self.lambda_kld = lambda_kld
        self.lambda_adv = lambda_adv

        self.fmd_loss = FmdLoss()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.vae(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        reconstruction, mu, log_variance = self.vae(x)
        kld_loss = self.kld_loss(mu, log_variance)
        fmd_loss = self.fmd_loss(reconstruction, y)
        loss = kld_loss * self.lambda_kld + fmd_loss * self.lambda_adv
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        reconstruction, mu, log_variance = self.vae(x)
        kld_loss = self.kld_loss(mu, log_variance)
        fmd_loss = self.fmd_loss(reconstruction, y)
        loss = kld_loss * self.lambda_kld + fmd_loss * self.lambda_adv
        self.log('val_loss', loss)
        return loss