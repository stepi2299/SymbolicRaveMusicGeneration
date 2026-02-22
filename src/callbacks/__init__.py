from omegaconf import DictConfig
from typing import List
from datetime import datetime

from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from src.callbacks.sampling_callback import SamplingCallback


def get_callbacks(
    cfg: DictConfig, model_name: str, dataset_name: str
) -> List[Callback]:
    used_callbacks = []
    timestamp_with_name = (
        model_name
        + "_"
        + dataset_name
        + "_"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir = cfg.output_dirpath.replace(
        "{unique_suffix}",
        timestamp_with_name,
    )
    if cfg.model_checkpoint:
        checkpoint_params = dict(cfg.model_checkpoint_params)
        used_callbacks.append(
            ModelCheckpoint(
                save_top_k=1, mode="min", dirpath=output_dir, **checkpoint_params
            )
        )
    if cfg.sampling:
        used_callbacks.append(
            SamplingCallback(output_dir=output_dir, **cfg.sampling_params)
        )
    if cfg.early_stopping:
        used_callbacks.append(EarlyStopping(mode="min", **cfg.early_stopping_params))
    if cfg.log_learning_rate:
        used_callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    return used_callbacks
