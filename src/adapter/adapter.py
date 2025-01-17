from omegaconf import DictConfig

from src.adapter.patchilizer import MusicPatchilizer
from src.adapter.tokenizer import Tokenizer


class MusicAdapter:
    def __init__(self, adapter_config: DictConfig) -> None:
        self._adapter = self.get_preprocessor(
            adapter_config.name, adapter_config.params
        )

    def encode(self, item: str) -> list[list[int]] | list[int]:
        return self._adapter.encode(item)

    def decode(self, item: list[list[int]] | list[str]) -> str:
        return self._adapter.decode(item)

    @staticmethod
    def get_preprocessor(
        preprocessor_name: str, preprocessor_config: DictConfig
    ) -> MusicPatchilizer | Tokenizer:
        if preprocessor_name == "patchilizer":
            return MusicPatchilizer(**preprocessor_config)
        elif preprocessor_name == "tokenizer":
            return Tokenizer(**preprocessor_config)
        else:
            raise ValueError(f"Unknown preprocessor: {preprocessor_name}")
