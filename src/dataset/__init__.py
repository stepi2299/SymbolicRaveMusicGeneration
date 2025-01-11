from pathlib import Path
from omegaconf import DictConfig

from src.dataset.symbolic_music_dataset import (
    SymbolicMusicDataset,
    MidiDataset,
    AbcDataset,
)


def get_dataset(config: DictConfig) -> SymbolicMusicDataset:
    dataset_path = Path.cwd() / "data" / config.data_dir
    dataset_path = dataset_path.resolve(True)
    if config.extension == "mid" or config.extension == "midi":
        return MidiDataset(dataset_path)
    elif config.extension == "abc":
        return AbcDataset(dataset_path)
    else:
        raise ValueError(f"Unknown dataset type: {config['dataset_type']}")
