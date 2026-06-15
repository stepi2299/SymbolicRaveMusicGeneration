from abc import ABC, abstractmethod

from torch import Tensor


class BaseAugmenter(ABC):
    @abstractmethod
    def augment(self, music_data: Tensor) -> Tensor:
        pass
