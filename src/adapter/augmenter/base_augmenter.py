from torch import Tensor

from abc import ABC, abstractmethod


class BaseAugmenter(ABC):
    @abstractmethod
    def augment(self, music_data: Tensor) -> Tensor:
        pass
