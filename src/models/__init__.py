from omegaconf import DictConfig, OmegaConf

from src.models.high_level_structures import registered_models, BaseHighLevelModel
from src.models.discriminator import discriminator_mapper, Discriminator
from src.adapter import MusicAdapter
from src.loss.base_loss import BaseLoss


def get_model(
    model_cfg: DictConfig,
    adapter: MusicAdapter,
    generation_params: DictConfig,
    loss_function: BaseLoss,
) -> BaseHighLevelModel:
    return registered_models[model_cfg.name](
        model_cfg=model_cfg,
        adapter=adapter,
        loss_function=loss_function,
        **generation_params,
    )


def get_discriminator(discriminator_cfg: DictConfig, vocab_size: int) -> Discriminator:
    discriminator_obj = discriminator_mapper(discriminator_cfg.name)(
        sequence_length=discriminator_cfg.sequence_length,
        patch_size=discriminator_cfg.patch_size,
        vocab_size=vocab_size,
        **discriminator_cfg.params,
    )
    return discriminator_obj
