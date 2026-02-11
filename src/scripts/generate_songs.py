import hydra
import logging as lg
from omegaconf import DictConfig
from pathlib import Path
from datetime import datetime

import torch

from src.competitors.first_stage import FirstStage
from src.utils import output_songs_path
from src.constants import ExportFormat
from src.models.high_level_structures.vae import VAE


@hydra.main(
    config_path="../../configs", config_name="generate_song", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision(cfg.processing_precision)
    lg.basicConfig(level=cfg.log_level)
    logger = lg.getLogger(__name__)

    checkpoint_path = Path(cfg.checkpoint_path).resolve(True)
    current_output_songs_path = output_songs_path / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    current_output_songs_path.mkdir(parents=True, exist_ok=False)
    competitor = FirstStage.load_from_checkpoint(checkpoint_path, weights_only=False)
    logger.info(
        f"Starting Generation Script! Expected number generated songs: {cfg.songs_to_generate}, "
        f"Checkpoint path: {current_output_songs_path.as_posix()}"
    )
    if isinstance(competitor.main_model, VAE):
        latent_values = torch.randn(
            cfg.songs_to_generate,
            competitor.main_model.latent_dim,
            device=competitor.device,
        )
        songs = competitor.generate(
            latent_values,
            current_output_songs_path,
            file_name="generated_song",
            export_format=ExportFormat.WAV,
            generation_params=cfg.generation_params,
        )
    else:
        raise NotImplementedError(
            "Generation is only implemented for VAE models for now"
        )
    logger.info(f"Generated songs: {len(songs)}")


if __name__ == "__main__":
    main()
