import logging as lg
import os

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.callbacks import get_callbacks
from src.competitors.first_stage import FirstStage
from src.dataset import get_dataset
from src.loss import get_loss_function
from src.models import get_model


@hydra.main(
    config_path="../../configs",
    config_name="train_representation",
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
        cfg.dataset,
        adapter_config=cfg.adapter,
        single_data_training=single_data_training,
        chosen_index=chosen_index,
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
        num_classes=symbolic_music_dataset.num_classes,
        generation_params=cfg.generation_params,
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
    callback_name = cfg.model.name + "_" + cfg.model.decoder_wrapper.type + "_" + cfg.model.decoder.name
    used_callbacks = get_callbacks(
        cfg=cfg.callbacks,
        model_name=callback_name,
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
