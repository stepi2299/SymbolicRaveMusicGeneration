from pathlib import Path
from omegaconf import DictConfig

from src.dataset.symbolic_music_dataset import (
    SymbolicMusicDataset,
    MidiDataset,
    AbcDataset,
)
from src.adapter.adapter import MusicAdapter


def get_dataset(
    dataset_config: DictConfig, adapter_config: DictConfig, single_data_training: bool = False, chosen_index: int = None
) -> SymbolicMusicDataset:
    music_adapter = MusicAdapter(adapter_config, dataset_config.extension)
    dataset_path = Path.cwd() / "data" / dataset_config.data_dir
    dataset_path = dataset_path.resolve(True)
    if dataset_config.extension == "mid" or dataset_config.extension == "midi":
        return MidiDataset(dataset_path, adapter=music_adapter, single_data_training=single_data_training, index_of_overtrained=chosen_index)
    elif dataset_config.extension == "abc":
        return AbcDataset(dataset_path, adapter=music_adapter, single_data_training=single_data_training, index_of_overtrained=chosen_index)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_config['dataset_type']}")
