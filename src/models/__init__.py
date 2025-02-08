from omegaconf import DictConfig

from src.models.vae import VAE
from src.models.encoder import encoder_mapper
from src.models.decoder import decoder_mapper


def get_vae_model(model_cfg: DictConfig, adapter_cfg: DictConfig) -> VAE:
    encoder_obj = encoder_mapper(model_cfg.encoder.name)(
        latent_dim=model_cfg.latent_dim,
        token_size=model_cfg.token_size,
        patch_len=adapter_cfg.params.patch_length,
        patch_size=adapter_cfg.params.patch_size,
        **model_cfg.encoder.params,
    )
    decoder_obj = decoder_mapper(model_cfg.decoder.name)(
        latent_dim=model_cfg.latent_dim,
        token_size=model_cfg.token_size,
        patch_len=adapter_cfg.params.patch_length,
        patch_size=adapter_cfg.params.patch_size,
        **model_cfg.decoder.params,
    )
    model = VAE(encoder=encoder_obj, decoder=decoder_obj)
    return model
