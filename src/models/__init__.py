from omegaconf import DictConfig

from src.models.vae import VAE
from src.models.encoder import encoder_mapper
from src.models.decoder import decoder_mapper
from src.models.discriminator import discriminator_mapper, Discriminator


def get_vae_model(model_cfg: DictConfig, adapter_cfg: DictConfig) -> VAE:
    encoder_obj = encoder_mapper(model_cfg.encoder.name)(
        latent_dim=model_cfg.latent_dim,
        vocab_size=adapter_cfg.params.vocab_size,
        sequence_length=adapter_cfg.params.sequence_length,
        patch_size=adapter_cfg.params.patch_size,
        **model_cfg.encoder.params,
    )
    decoder_obj = decoder_mapper(model_cfg.decoder.name)(
        latent_dim=model_cfg.latent_dim,
        vocab_size=adapter_cfg.params.vocab_size,
        sequence_length=adapter_cfg.params.sequence_length,
        patch_size=adapter_cfg.params.patch_size,
        **model_cfg.decoder.params,
    )
    model = VAE(encoder=encoder_obj, decoder=decoder_obj)
    return model


def get_discriminator(discriminator_cfg: DictConfig, vocab_size: int) -> Discriminator:
    discriminator_obj = discriminator_mapper(discriminator_cfg.name)(
        sequence_length=discriminator_cfg.sequence_length,
        patch_size=discriminator_cfg.patch_size,
        vocab_size=vocab_size,
        **discriminator_cfg.params,
    )
    return discriminator_obj
