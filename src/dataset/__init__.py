from pathlib import Path

from omegaconf import DictConfig

from src.adapter.adapter import MusicAdapter
from src.dataset.symbolic_music_dataset import (
    AbcDataset,
    MidiDataset,
    SymbolicMusicDataset,
)


def get_dataset(
    dataset_config: DictConfig,
    adapter_config: DictConfig | None = None,
    adapter: MusicAdapter = None,
    single_data_training: bool = False,
    chosen_index: int = None,
) -> SymbolicMusicDataset:
    if adapter is None:
        if adapter_config is None:
            raise ValueError("adapter_config must be provided if adapter is None")
        music_adapter = MusicAdapter(adapter_config, dataset_config.extension, dataset_config.name)
    else:
        if adapter_config is not None:
            raise ValueError("adapter_config must be provided ONLY if adapter is None, otherwise None")
        music_adapter = adapter

    dataset_path = Path.cwd() / "data" / dataset_config.data_dir
    dataset_path = dataset_path.resolve(True)
    if dataset_config.extension in ("mid", "midi"):
        return MidiDataset(
            dataset_path,
            adapter=music_adapter,
            augmentation_params=dataset_config.augmentation_params,
            single_data_training=single_data_training,
            index_of_overtrained=chosen_index,
        )
    elif dataset_config.extension == "abc":
        return AbcDataset(
            dataset_path,
            adapter=music_adapter,
            augmentation_params=dataset_config.augmentation_params,
            single_data_training=single_data_training,
            index_of_overtrained=chosen_index,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_config['dataset_type']}")
