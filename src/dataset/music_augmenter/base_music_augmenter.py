from abc import ABC, abstractmethod

from mido import MidiFile


class BaseMusicAugmenter(ABC):
    @abstractmethod
    def __call__(self, music_data: list[str] | MidiFile) -> list[str] | MidiFile:
        pass
