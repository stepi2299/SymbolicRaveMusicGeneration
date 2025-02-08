import hydra
import logging as lg
from omegaconf import DictConfig
from pathlib import Path
from datetime import datetime

import torch

from src.competitors.first_stage import FirstStage
from src.utils import output_songs_path, convert_mtf2midi


@hydra.main(
    config_path="../../configs", config_name="generate_song", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision(cfg.processing_precision)
    lg.basicConfig(level=cfg.log_level)
    logger = lg.getLogger(__name__)

    checkpoint_path = Path(cfg.checkpoint_path).resolve(True)
    competitor = FirstStage.load_from_checkpoint(checkpoint_path)
    latent_values = torch.randn(1, competitor.vae.latent_dim, device=competitor.device)
    songs = competitor.generate(latent_values)
    logger.info(f"Number generated songs: {len(songs)}")
    current_output_songs_path = output_songs_path / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    current_output_songs_path.mkdir(parents=True, exist_ok=False)
    logger.info(f"Generated songs will be saved in {current_output_songs_path}")
    for i, song in enumerate(songs):
        midi_song = convert_mtf2midi(song)
        file_path = current_output_songs_path / f"{i}.mid"
        midi_song.save(file_path)
    logger.info(f"Successfully saved {len(songs)} songs in {current_output_songs_path}")


if __name__ == "__main__":
    main()
