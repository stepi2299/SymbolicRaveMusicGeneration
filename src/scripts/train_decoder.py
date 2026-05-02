import hydra
import logging as lg
import wandb
from omegaconf import DictConfig
from pathlib import Path
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.dataset import get_dataset
from src.models import get_discriminator
from src.competitors.second_stage import SecondStage
from src.competitors.first_stage import FirstStage
from src.callbacks import get_callbacks


@hydra.main(
    config_path="../../configs", config_name="train_decoder", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    lg.basicConfig(level=cfg.log_level, format=cfg.log_format, force=True)

    # Load the first-stage checkpoint into the main model
    checkpoint_path = Path(cfg.main_model_weights_path).resolve(True)
    first_stage_checkpoint = FirstStage.load_from_checkpoint(
        checkpoint_path=checkpoint_path.as_posix(), weights_only=False
    )
    dataset = get_dataset(cfg.dataset, adapter=first_stage_checkpoint.adapter)

    discriminator = get_discriminator(
        cfg.discriminator,
        adapter=first_stage_checkpoint.adapter,
    )

    if cfg.pretrained:
        discriminator.load_pretrained()

    competitor = SecondStage(
        main_model=first_stage_checkpoint.main_model,
        discriminator=discriminator,
        symbolic_music_dataset=dataset,
        **cfg.competitor,
    )

    callback_name = (
        "gab"
        + "_"
        + first_stage_checkpoint.main_model.__class__.__name__
        + "_"
        + cfg.discriminator.name
        + "_"
        + cfg.dataset.name
    )
    used_callbacks = get_callbacks(
        cfg=cfg.callbacks,
        model_name=callback_name,
        dataset_name=cfg.dataset.name,
    )

    trainer = Trainer(
        callbacks=used_callbacks,
        max_epochs=cfg.epochs,
        logger=wandb_logger,
    )
    trainer.fit(competitor)
    wandb.finish()


if __name__ == "__main__":
    main()
