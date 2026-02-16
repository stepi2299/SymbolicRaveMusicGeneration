import hydra
import logging as lg
import wandb
from omegaconf import DictConfig
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.dataset import get_dataset
from src.models import get_model
from src.competitors.first_stage import FirstStage
from src.callbacks import get_callbacks
from src.loss import get_loss_function


@hydra.main(
    config_path="../../configs",
    config_name="train_patch_representation",
    version_base="1.3",
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

    single_data_training = cfg.single_data_training
    if single_data_training:
        try:
            chosen_index = cfg.chosen_index
        except AttributeError:
            chosen_index = None
        wandb.log({"INFO": f"Single Data Training with index: {chosen_index}"})
    else:
        chosen_index = None
    symbolic_music_dataset = get_dataset(
        cfg.dataset, cfg.adapter, single_data_training, chosen_index
    )
    loss_function = get_loss_function(
        loss_name=cfg.loss.name,
        vocab_size=symbolic_music_dataset.music_adapter.vocab_size,
        pad_token_id=symbolic_music_dataset.music_adapter.pad_token_id,
        **cfg.loss.params,
    )
    high_level_model = get_model(
        model_cfg=cfg.model,
        adapter=symbolic_music_dataset.music_adapter,
        vae_params=cfg.vae_params,
        loss_function=loss_function,
    )
    if cfg.pretrained:
        high_level_model.load_pretrained()

    competitor = FirstStage(
        main_model=high_level_model,
        symbolic_music_dataset=symbolic_music_dataset,
        single_data_training=single_data_training,
        **cfg.competitor,
    )
    used_callbacks = get_callbacks(
        cfg=cfg.callbacks,
        latent_dim=cfg.model.latent_dim,
        model_name=cfg.model.name,
        dataset_name=cfg.dataset.name,
    )
    grad_clip_val = cfg.grad_clip_val
    if grad_clip_val == "None":
        grad_clip_val = None
    trainer = Trainer(
        callbacks=used_callbacks,
        max_epochs=cfg.epochs,
        logger=wandb_logger,
        gradient_clip_val=grad_clip_val,
    )
    trainer.fit(competitor)
    wandb.finish()


if __name__ == "__main__":
    main()
