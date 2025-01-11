from omegaconf import DictConfig

from src.models.vae import VAE
from src.models.encoder import encoder_mapper
from src.models.decoder import decoder_mapper


def get_vae_model(cfg: DictConfig) -> VAE:
    encoder_obj = encoder_mapper(cfg.encoder.name)(**cfg.encoder.params)
    decoder_obj = decoder_mapper(cfg.decoder.name)(**cfg.decoder.params)
    model = VAE(encoder=encoder_obj, decoder=decoder_obj)
    return model
