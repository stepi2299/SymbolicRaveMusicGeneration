import logging as lg
from pathlib import Path

import torch
from miditok import (
    MMM,
    REMI,
    TSD,
    CPWord,
    MIDILike,
    MuMIDI,
    Octuple,
    Structured,
    TokenizerConfig,
)
from mido import MidiFile as MidoMidiFile

from src.adapter.augmenter import CpWordAugmenter, OneLevelTokAugmenter
from src.constants import ExportFormat, SymbolicMusicFormat
from src.utils import (
    base_dataset_path,
    base_weights_path,
    convert_mido_to_score,
    export_to_wav,
)

logger = lg.getLogger(__name__)


class Tokenizer:
    TOKENIZER_MAPPER = {
        "remi": REMI,
        "midilike": MIDILike,
        "tsd": TSD,
        "structured": Structured,
        "cpword": CPWord,
        "octuple": Octuple,
        "mumidi": MuMIDI,
        "mmm": MMM,
    }
    PATCH_MAPPER = {
        "remi": 1,
        "midilike": 1,
        "tsd": 1,
        "structured": 1,
        "cpword": 5,
        "octuple": 5,
        "mumidi": 1,
        "mmm": 1,
    }
    AUGMENTATION_MAPPER = {
        "remi": OneLevelTokAugmenter,
        "midilike": OneLevelTokAugmenter,
        "tsd": OneLevelTokAugmenter,
        "structured": OneLevelTokAugmenter,
        "cpword": CpWordAugmenter,
        "octuple": OneLevelTokAugmenter,
        "mumidi": OneLevelTokAugmenter,
        "mmm": OneLevelTokAugmenter,
    }
    TOKENIZER_BASE_PATH = base_weights_path / "tokenizers"

    def __init__(
        self,
        tokenizer_name: str,
        training_model: str,
        vocab_size: int,
        dataset_name: str,
        sequence_length: int,
        add_special_patches: bool = True,
        config: dict | None = None,
        augmenter_params: dict | None = None,
        **kwargs,
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.training_model = training_model
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.add_special_patches = add_special_patches
        self._dataset_name = dataset_name
        self.patch_size = self.PATCH_MAPPER[tokenizer_name]
        self._augmenter_params = augmenter_params if augmenter_params is not None else {}
        if tokenizer_name not in self.TOKENIZER_MAPPER:
            raise ValueError(
                f"Tokenizer {tokenizer_name} not supported, supported tokenizers " f"are {self.TOKENIZER_MAPPER.keys()}"
            )
        if self.cache_tokenizer_path.exists():
            logger.info(f"Loading tokenizer {self.tokenizer_name} from {self.cache_tokenizer_path}")
            self.tokenizer = self.TOKENIZER_MAPPER[tokenizer_name](params=self.cache_tokenizer_path)
            self._is_trained = True
        else:
            logger.info(f"Initializing new tokenizer {self.tokenizer_name} with config: {config}")
            tokenizer_config = TokenizerConfig(**config if config is not None else {})
            self.tokenizer = self.TOKENIZER_MAPPER[tokenizer_name](tokenizer_config=tokenizer_config)
            self._is_trained = False
        self._dim_of_song = 2 if self.patch_size > 1 else 1
        augmenter_class = self.AUGMENTATION_MAPPER[tokenizer_name]
        self.augmenter = augmenter_class(tokenizer_config=self.tokenizer.config, **self._augmenter_params)

    def encode(self, music_file: MidoMidiFile, symbolic_format: SymbolicMusicFormat) -> list[int]:
        if symbolic_format == SymbolicMusicFormat.ABC:
            raise ValueError("ABC format is not supported for tokenizer")
        if not self._is_trained:
            logger.warning("Tokenizer is not trained")
        midi_score = convert_mido_to_score(music_file)  # Change from mido into symusic.Score
        tok_sequence = self.tokenizer(midi_score)[0].ids[: self.sequence_length - 2]
        if self.add_special_patches:
            tok_sequence = self._add_special_tokens(tok_sequence)
        return tok_sequence

    def decode(
        self,
        tokens: list[int] | list[list[int]],
        symbolic_format: SymbolicMusicFormat,
        output_path: Path,
        export_format: ExportFormat = ExportFormat.WAV,
    ) -> Path:
        if symbolic_format == SymbolicMusicFormat.ABC:
            raise ValueError("ABC format is not supported for tokenizer")
        if not self._is_trained:
            logger.warning("Tokenizer is not trained")
        score = self.tokenizer.decode([tokens])

        match export_format:
            case ExportFormat.SYMBOLIC:
                output_path = output_path.with_suffix(".mid")
                score.dump_midi(path=output_path.as_posix())
                return output_path
            case ExportFormat.WAV:
                output_path = output_path.with_suffix(".wav")
                export_to_wav(score, output_path)
                return output_path
            case _:
                raise ValueError(f"Export format {export_format} is not supported")

    def augment(self, music_data: torch.Tensor) -> torch.Tensor:
        return self.augmenter.augment(music_data)

    def train(self, files_path: Path) -> None:
        files_paths = []
        for item in files_path.iterdir():
            if item.is_dir():
                files_paths.extend(item.iterdir())
            else:
                files_paths.append(item)
        if self.tokenizer_name != "cpword":
            self.tokenizer.train(
                vocab_size=self.vocab_size,
                files_paths=files_paths,
                model=self.training_model,
            )
            self._is_trained = True
            self.save()
            self.augmenter = self.AUGMENTATION_MAPPER[self.tokenizer_name](
                tokenizer_config=self.tokenizer.config, **self._augmenter_params
            )
        else:
            logger.info("Current tokenizer is CPWord, skipping training step.")
            self._is_trained = True

    def save(self) -> None:
        logger.info(f"Saving tokenizer {self.tokenizer_name} to {self.cache_tokenizer_path}")
        self.tokenizer.save(out_path=self.cache_tokenizer_path)

    @property
    def is_trained(self):
        return self._is_trained

    @property
    def cache_tokenizer_path(self) -> Path:
        return (
            self.TOKENIZER_BASE_PATH
            / f"{self._dataset_name}_{self.tokenizer_name}_{self.training_model}_{self.vocab_size}.json"
        )

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer["EOS_None"]

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer["BOS_None"]

    @property
    def song_dim(self) -> int:
        return self._dim_of_song

    def _add_special_tokens(self, tokens: list[int] | list[list[int]]) -> list[int] | list[list[int]]:
        if self._dim_of_song == 2:
            bos_token = [self.bos_token_id] * self.patch_size
            eos_token = [self.eos_token_id] * self.patch_size
            return [bos_token] + tokens + [eos_token]
        elif self._dim_of_song == 1:
            return [self.bos_token_id] + tokens + [self.eos_token_id]
        else:
            raise ValueError(f"Unsupported song dimension {self._dim_of_song}")


def get_tokenizer(
    tokenizer: str,
    training_model: str,
    dataset_name: str,
    force: bool = False,
    **kwargs,
) -> Tokenizer:
    tokenizer = Tokenizer(
        tokenizer_name=tokenizer,
        training_model=training_model,
        dataset_name=dataset_name,
        **kwargs,
    )
    if force or not tokenizer.is_trained:
        dataset_path = (base_dataset_path / dataset_name).resolve(True)
        tokenizer.train(files_path=dataset_path)
    return tokenizer
