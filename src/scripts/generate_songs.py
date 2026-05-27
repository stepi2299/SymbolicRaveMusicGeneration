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
    lg.basicConfig(level=cfg.log_level, force=True)
    logger = lg.getLogger(__name__)

    checkpoint_path = Path(cfg.checkpoint_path).resolve(True)
    model_name = checkpoint_path.parent.name
    current_output_songs_path = output_songs_path / (
        model_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_format = ExportFormat(cfg.output_format)
    current_output_songs_path.mkdir(parents=True, exist_ok=False)
    competitor = FirstStage.load_from_checkpoint(checkpoint_path, weights_only=False)
    logger.info(
        f"Starting Generation Script! Expected number generated songs: {cfg.songs_to_generate}, "
        f"Checkpoint path: {current_output_songs_path.as_posix()}"
    )
    songs = competitor.generate(
        folder_path=current_output_songs_path,
        file_name="generated_song",
        export_format=output_format,
        generation_params=cfg.generation_params,
        num_samples=cfg.songs_to_generate,
    )
    logger.info(f"Generated songs: {len(songs)}")


if __name__ == "__main__":
    main()
