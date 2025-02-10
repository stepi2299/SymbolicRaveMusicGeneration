import hydra
import logging as lg
import wandb
from omegaconf import DictConfig

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.dataset import get_dataset
from src.models import get_vae_model, get_discriminator
from src.competitors.second_stage import SecondStage
from src.callbacks import get_callbacks


@hydra.main(
    config_path="../../configs", config_name="train_decoder", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        save_dir=cfg.wandb.logs_dir,
        log_model=False,
    )
    wandb_logger.log_hyperparams(cfg)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision(cfg.processing_precision)
    lg.basicConfig(level=cfg.log_level)

    symbolic_music_dataset = get_dataset(cfg.dataset, cfg.adapter)
    vae_model = get_vae_model(cfg.model, cfg.adapter)
    discriminator = get_discriminator(cfg.discriminator)
    vae_model.load_state_dict(torch.load(cfg.vae_weights_path))

    competitor = SecondStage(
        vae=vae_model,
        discriminator=discriminator,
        symbolic_music_dataset=symbolic_music_dataset,
        **cfg.competitor,
    )
    used_callbacks = get_callbacks(
        cfg=cfg.callbacks, latent_dim=vae_model.decoder.latent_dim
    )
    trainer = Trainer(
        callbacks=used_callbacks, max_epochs=cfg.epochs, logger=wandb_logger
    )
    trainer.fit(competitor)
    wandb.finish()


if __name__ == "__main__":
    main()
