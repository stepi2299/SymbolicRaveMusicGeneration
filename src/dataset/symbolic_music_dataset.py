import logging as lg
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import torch
from miditoolkit import MidiFile
from torch.utils.data import Dataset

from src.adapter import MusicAdapter
from src.constants import SymbolicMusicFormat
from src.dataset.music_augmenter import get_augmenter
from src.utils import load_abc_song, load_midi_song

logger = lg.getLogger(__name__)


class SymbolicMusicDataset(Dataset, ABC):
    def __init__(
        self,
        dataset_path: Path,
        adapter: MusicAdapter,
        music_format: SymbolicMusicFormat,
        augmentation_params: dict,
        load_music_function: Callable[[Path], str | MidiFile],
        single_data_training: bool = False,
        index_of_overtrained: int | None = None,
    ) -> None:
        self.dataset_path = dataset_path
        self._classes = sorted([d.name for d in self.dataset_path.iterdir() if d.is_dir()])
        self._class_to_idx = {cls_name: i for i, cls_name in enumerate(self._classes)}
        self.load_music_function = load_music_function
        self._augment: bool | None = None
        self._single_data_training = single_data_training
        self._index_of_overtrained = index_of_overtrained
        self.music_adapter = adapter
        self.music_format = music_format
        self._music_augmenter = get_augmenter(music_format=music_format, **augmentation_params)
        self._samples = self._load_music_files_with_labels()
        if self._single_data_training:
            if self._index_of_overtrained is not None:
                if index_of_overtrained >= len(self._samples):
                    raise ValueError(
                        f"Index of overtrained music files: {self._index_of_overtrained} is out of range."
                        f"Files range: {len(self._samples)}"
                    )
                self._samples = [self._samples[index_of_overtrained]]
            else:
                self._index_of_overtrained = random.randint(0, len(self._samples) - 1)
                self._samples = [self._samples[self._index_of_overtrained]]
        logger.info(
            "Loaded dataset with %d samples across %d classes.",
            len(self._samples),
            len(self._classes),
        )

    @abstractmethod
    def imread_music_files(self) -> list[Path]:
        pass

    def _load_music_files_with_labels(self) -> list[tuple[Path, int]]:
        samples = []
        all_paths = self.imread_music_files()
        for path in all_paths:
            class_name = path.relative_to(self.dataset_path).parts[0]
            if class_name in self._class_to_idx:
                class_idx = self._class_to_idx[class_name]
                samples.append((path, class_idx))
        return samples

    def _preprocess_music_file(self) -> str | MidiFile:
        pass

    def turn_augmentation(self, augment: bool) -> None:
        self._augment = augment
        logger.info(f"Augmentation set to: {self._augment}")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        music_file_path, class_idx = self._samples[idx]
        music_file = self.load_music_function(music_file_path)
        if self._augment:
            music_file = self._music_augmenter(music_file)
        music_file = self.music_adapter.encode(music_file)
        music_file = self._get_padded_data(music_file)
        tensor_data = torch.tensor(music_file)
        label_tensor = torch.tensor(class_idx, dtype=torch.long)
        if self._augment:
            tensor_data = self.music_adapter.augment(tensor_data)
        return tensor_data, label_tensor

    def _get_padded_data(self, music_file: list[list[int]] | list[int]) -> list[list[int]] | list[int]:
        return self.music_adapter.get_padded_sequence(music_file)

    def get_music_file(self, idx: int | None = None) -> str | MidiFile:
        if idx is None:
            idx = self._index_of_overtrained
        music_file_path = self._samples[idx]
        return self.load_music_function(music_file_path)

    @property
    def num_classes(self) -> int:
        return len(self._classes)


class MidiDataset(SymbolicMusicDataset):
    def __init__(
        self,
        dataset_path: Path,
        adapter: MusicAdapter,
        augmentation_params: dict,
        load_music_function: Callable[[Path], str] = load_midi_song,
        single_data_training: bool = False,
        index_of_overtrained: int | None = None,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            load_music_function=load_music_function,
            augmentation_params=augmentation_params,
            music_format=SymbolicMusicFormat.MIDI,
            adapter=adapter,
            single_data_training=single_data_training,
            index_of_overtrained=index_of_overtrained,
        )

    def imread_music_files(self) -> list[Path]:
        return list(self.dataset_path.glob("**/*.mid")) + list(self.dataset_path.glob("**/*.midi"))


class AbcDataset(SymbolicMusicDataset):
    def __init__(
        self,
        dataset_path: Path,
        adapter: MusicAdapter,
        augmentation_params: dict,
        load_music_function: Callable[[Path], str] = load_abc_song,
        single_data_training: bool = False,
        index_of_overtrained: int | None = None,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            load_music_function=load_music_function,
            augmentation_params=augmentation_params,
            music_format=SymbolicMusicFormat.ABC,
            adapter=adapter,
            single_data_training=single_data_training,
            index_of_overtrained=index_of_overtrained,
        )

    def imread_music_files(self) -> list[Path]:
        return list(self.dataset_path.glob("**/*.abc"))
