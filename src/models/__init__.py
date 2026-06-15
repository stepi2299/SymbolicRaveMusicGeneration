from omegaconf import DictConfig, OmegaConf

from src.adapter import MusicAdapter
from src.loss.base_loss import BaseLoss
from src.models.discriminator import Discriminator, discriminator_mapper
from src.models.high_level_structures import BaseHighLevelModel, registered_models


def get_model(
    model_cfg: DictConfig,
    adapter: MusicAdapter,
    generation_params: DictConfig,
    loss_function: BaseLoss,
    num_classes: int,
) -> BaseHighLevelModel:
    return registered_models[model_cfg.name](
        model_cfg=model_cfg,
        adapter=adapter,
        loss_function=loss_function,
        num_classes=num_classes,
        **generation_params,
    )


def get_discriminator(discriminator_cfg: DictConfig, adapter: MusicAdapter) -> Discriminator:
    discriminator_obj = discriminator_mapper(discriminator_cfg.name)(
        sequence_length=adapter.sequence_length,
        patch_size=adapter.patch_size,
        vocab_size=adapter.vocab_size,
        **discriminator_cfg.params,
    )
    return discriminator_obj
