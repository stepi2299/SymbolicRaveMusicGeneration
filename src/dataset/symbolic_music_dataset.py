from pathlib import Path
from typing import Callable
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

import torch

from src.utils import load_midi_song, load_abc_song
from src.adapter.adapter import MusicAdapter


class SymbolicMusicDataset(Dataset, ABC):
    def __init__(
        self,
        dataset_path: Path,
        adapter: MusicAdapter,
        load_music_function: Callable[[Path], str],
        transform: Callable[[str], str] | None = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.load_music_function = load_music_function
        self.transform = transform
        self.music_files_paths = self.imread_music_files()
        self.music_adapter = adapter

    @abstractmethod
    def imread_music_files(self) -> list[Path]:
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
        padded_bar = [self.music_adapter.pad_token] * self.music_adapter.patch_size
        padded_music_file = [padded_bar] * (
            self.music_adapter.patch_length - len(music_file)
        )
        new_music_file = music_file + padded_music_file
        return torch.tensor(new_music_file)


class MidiDataset(SymbolicMusicDataset):
    def __init__(
        self,
        dataset_path: Path,
        adapter: MusicAdapter,
        load_music_function: Callable[[Path], str] = load_midi_song,
        transform: Callable[[str], str] | None = None,
    ) -> None:
        super(MidiDataset, self).__init__(
            dataset_path=dataset_path,
            load_music_function=load_music_function,
            adapter=adapter,
            transform=transform,
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
    ) -> None:
        super(AbcDataset, self).__init__(
            dataset_path=dataset_path,
            load_music_function=load_music_function,
            adapter=adapter,
            transform=transform,
        )

    def imread_music_files(self) -> list[Path]:
        return list(self.dataset_path.glob("**/*.abc"))
