from omegaconf import DictConfig

from src.models.high_level_structures.vae import VAE
from src.models.encoder import encoder_mapper
from src.models.decoder import decoder_mapper
from src.models.discriminator import discriminator_mapper, Discriminator
from src.adapter import MusicAdapter
from src.loss.base_loss import BaseLoss


def get_model(
    model_cfg: DictConfig,
    adapter: MusicAdapter,
    vae_params: DictConfig,
    loss_function: BaseLoss,
) -> VAE:
    model = VAE(
        loss_function=loss_function, model_cfg=model_cfg, adapter=adapter, **vae_params
    )
    return model


def get_discriminator(discriminator_cfg: DictConfig, vocab_size: int) -> Discriminator:
    discriminator_obj = discriminator_mapper(discriminator_cfg.name)(
        sequence_length=discriminator_cfg.sequence_length,
        patch_size=discriminator_cfg.patch_size,
        vocab_size=vocab_size,
        **discriminator_cfg.params,
    )
    return discriminator_obj
