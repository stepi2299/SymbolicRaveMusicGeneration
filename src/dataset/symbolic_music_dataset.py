from pathlib import Path
from typing import Callable
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import random

import torch

from miditoolkit import MidiFile

from src.utils import load_midi_song, load_abc_song, convert_mido_to_score
from src.adapter import MusicAdapter
from src.constants import SymbolicMusicFormat


class SymbolicMusicDataset(Dataset, ABC):
    def __init__(
        self,
        dataset_path: Path,
        adapter: MusicAdapter,
        music_format: SymbolicMusicFormat,
        load_music_function: Callable[[Path], str | MidiFile],
        transform: Callable[[str], str] | None = None,
        single_data_training: bool = False,
        index_of_overtrained: int | None = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.load_music_function = load_music_function
        self.transform = transform
        self._single_data_training = single_data_training
        self._index_of_overtrained = index_of_overtrained
        self.music_adapter = adapter
        self.music_format = music_format
        self.music_files_paths = self.imread_music_files()
        if self._single_data_training:
            if self._index_of_overtrained is not None:
                if index_of_overtrained >= len(self.music_files_paths):
                    raise ValueError(
                        f"Index of overtrained music files: {self._index_of_overtrained} is out of range."
                        f"Files range: {len(self.music_files_paths)}"
                    )
                self.music_files_paths = [self.music_files_paths[index_of_overtrained]]
            else:
                self._index_of_overtrained = random.randint(
                    0, len(self.music_files_paths) - 1
                )
                self.music_files_paths = [
                    self.music_files_paths[self._index_of_overtrained]
                ]

    @abstractmethod
    def imread_music_files(self) -> list[Path]:
        pass

    def _preprocess_music_file(self) -> str | MidiFile:
        pass

    def __len__(self):
        return len(self.music_files_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        music_file_path = self.music_files_paths[idx]
        music_file = self.load_music_function(music_file_path)
        if self.transform:
            music_file = self.transform(music_file)
        music_file = self.music_adapter.encode(music_file)
        return self._get_padded_tensor(music_file)

    def _get_padded_tensor(self, music_file: list[list[int]]) -> torch.Tensor:
        padded_music = self.music_adapter.get_padded_sequence(music_file)
        return torch.tensor(padded_music)


class MidiDataset(SymbolicMusicDataset):
    def __init__(
        self,
        dataset_path: Path,
        adapter: MusicAdapter,
        load_music_function: Callable[[Path], str] = load_midi_song,
        transform: Callable[[str], str] | None = None,
        single_data_training: bool = False,
        index_of_overtrained: int | None = None,
    ) -> None:
        super(MidiDataset, self).__init__(
            dataset_path=dataset_path,
            load_music_function=load_music_function,
            music_format=SymbolicMusicFormat.MIDI,
            adapter=adapter,
            transform=transform,
            single_data_training=single_data_training,
            index_of_overtrained=index_of_overtrained,
        )

    def imread_music_files(self) -> list[Path]:
        return list(self.dataset_path.glob("**/*.mid")) + list(
            self.dataset_path.glob("**/*.midi")
        )


class AbcDataset(SymbolicMusicDataset):
    def __init__(
        self,
        dataset_path: Path,
        adapter: MusicAdapter,
        load_music_function: Callable[[Path], str] = load_abc_song,
        transform: Callable[[str], str] | None = None,
        single_data_training: bool = False,
        index_of_overtrained: int | None = None,
    ) -> None:
        super(AbcDataset, self).__init__(
            dataset_path=dataset_path,
            load_music_function=load_music_function,
            music_format=SymbolicMusicFormat.ABC,
            adapter=adapter,
            transform=transform,
            single_data_training=single_data_training,
            index_of_overtrained=index_of_overtrained,
        )

    def imread_music_files(self) -> list[Path]:
        return list(self.dataset_path.glob("**/*.abc"))
