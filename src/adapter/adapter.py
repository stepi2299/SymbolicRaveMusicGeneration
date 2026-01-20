import torch
from omegaconf import DictConfig

from src.adapter.patchilizer import MusicPatchilizer
from src.adapter.tokenizer import Tokenizer, get_tokenizer

from mido import MidiFile as MidoMidiFile
from pathlib import Path

from src.constants import SymbolicMusicFormat, MusicAdapterType, ExportFormat


class MusicAdapter:
    def __init__(
        self, adapter_config: DictConfig, symbolic_format: str, dataset_name: str
    ) -> None:
        self.adapter_type = MusicAdapterType(adapter_config.name)
        self._adapter = self.get_preprocessor(
            self.adapter_type,
            dataset_name=dataset_name,
            preprocessor_config=adapter_config.params,
        )
        if symbolic_format == "mid":
            symbolic_format = "midi"
        self.symbolic_format = SymbolicMusicFormat(symbolic_format)

    def encode(self, item: MidoMidiFile | str) -> list[list[int]] | list[int]:
        return self._adapter.encode(item, self.symbolic_format)

    def decode(
        self,
        item: list[list[int]] | list[int],
        output_path: Path,
        export_format: ExportFormat,
    ) -> Path:
        try:
            ret_music = self._adapter.decode(
                item, self.symbolic_format, output_path, export_format
            )
        except Exception as e:
            raise RuntimeError(f"Decoding failed: {e}") from e
        if isinstance(ret_music, str):
            ret_music = ret_music.strip()
        return ret_music

    def augment(self, music_data: torch.Tensor) -> torch.Tensor:
        return self._adapter.augment(music_data)

    def _get_single_dim_padded_sequence(self, music_file: list[int]) -> list[int]:
        return music_file + [self.pad_token_id] * (
            self.sequence_length - len(music_file)
        )

    def _get_two_dim_padded_sequence(
        self, music_file: list[list[int]]
    ) -> list[list[int]]:
        music_file = music_file + [[self.pad_token_id] * self.patch_size] * (
            self.sequence_length - len(music_file)
        )
        return music_file

    def get_padded_sequence(
        self, music_file: list[list[int]] | list[int]
    ) -> list[list[int]] | list[int]:
        if len(music_file) < self.sequence_length:
            if isinstance(music_file[0], list):
                return self._get_two_dim_padded_sequence(music_file)
            else:
                return self._get_single_dim_padded_sequence(music_file)
        elif len(music_file) == self.sequence_length:
            return music_file
        else:
            raise ValueError(
                f"Input music length {len(music_file)} is greater than the defined "
                f"sequence length {self.sequence_length}"
            )

    @property
    def sequence_length(self) -> int | None:
        return self._adapter.sequence_length

    @property
    def pad_token_id(self) -> int | None:
        return self._adapter.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._adapter.eos_token_id

    @property
    def bos_token_id(self) -> int:
        return self._adapter.bos_token_id

    @property
    def vocab_size(self) -> int:
        return self._adapter.vocab_size

    @property
    def patch_size(self) -> int:
        return self._adapter.patch_size

    @property
    def song_dim(self) -> int:
        return self._adapter.song_dim

    @staticmethod
    def get_preprocessor(
        preprocessor_name: MusicAdapterType,
        dataset_name: str,
        preprocessor_config: DictConfig,
    ) -> MusicPatchilizer | Tokenizer:
        if preprocessor_name == MusicAdapterType.PATCHILIZER:
            return MusicPatchilizer(**preprocessor_config)
        elif preprocessor_name == MusicAdapterType.TOKENIZER:
            return get_tokenizer(dataset_name=dataset_name, **preprocessor_config)
        else:
            raise ValueError(f"Unknown preprocessor: {preprocessor_name}")
