from omegaconf import DictConfig
from typing import List
from datetime import datetime

from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping

from src.callbacks.sampling_callback import SamplingCallback


def get_callbacks(cfg: DictConfig, latent_dim: int) -> List[Callback]:
    used_callbacks = []
    if cfg.model_checkpoint:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_params = dict(cfg.model_checkpoint_params)
        checkpoint_params["dirpath"] = checkpoint_params["dirpath"].replace(
            "{unique_suffix}",
            timestamp,
        )
        used_callbacks.append(
            ModelCheckpoint(save_top_k=1, mode="min", **checkpoint_params)
        )
    if cfg.sampling:
        used_callbacks.append(
            SamplingCallback(latent_dim=latent_dim, **cfg.sampling_params)
        )
    if cfg.early_stopping:
        used_callbacks.append(
            EarlyStopping(monitor="val_loss", mode="min", **cfg.early_stopping_params)
        )
    return used_callbacks
