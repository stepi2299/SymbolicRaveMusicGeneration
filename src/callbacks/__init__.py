from omegaconf import DictConfig
from typing import List

from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping

from src.callbacks.sampling_callback import SamplingCallback


def get_callbacks(config: DictConfig, latent_dim: int) -> List[Callback]:
    used_callbacks = []
    if config.model_checkpoint:
        used_callbacks.append(
            ModelCheckpoint(save_top_k=1, mode="min", **config.model_checkpoint_params)
        )
    elif config.sampling:
        used_callbacks.append(
            SamplingCallback(latent_dim=latent_dim, **config.sampling_params)
        )
    elif config.early_stopping:
        used_callbacks.append(
            EarlyStopping(
                monitor="val_loss", mode="min", **config.early_stopping_params
            )
        )
    return used_callbacks
