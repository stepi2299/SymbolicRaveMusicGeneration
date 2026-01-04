from pathlib import Path
import logging as lg

from miditok import REMI, MIDILike, TSD, Structured, CPWord, Octuple, MuMIDI, MMM
from mido import MidiFile as MidoMidiFile

from src.utils import (
    base_weights_path,
    base_dataset_path,
    convert_mido_to_score,
    export_to_wav,
)
from src.constants import SymbolicMusicFormat
from src.constants import ExportFormat


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
    TOKENIZER_BASE_PATH = base_weights_path / "tokenizers"

    def __init__(
        self,
        tokenizer_name: str,
        training_model: str,
        vocab_size: int,
        dataset_name: str,
        sequence_length: int,
        add_special_patches: bool = True,
        **kwargs,
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.training_model = training_model
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.add_special_patches = add_special_patches
        self._dataset_name = dataset_name
        self.patch_size = self.PATCH_MAPPER[tokenizer_name]
        if tokenizer_name not in self.TOKENIZER_MAPPER:
            raise ValueError(
                f"Tokenizer {tokenizer_name} not supported, supported tokenizers "
                f"are {self.TOKENIZER_MAPPER.keys()}"
            )
        if self.cache_tokenizer_path.exists():
            logger.info(
                f"Loading tokenizer {self.tokenizer_name} from {self.cache_tokenizer_path}"
            )
            self.tokenizer = self.TOKENIZER_MAPPER[tokenizer_name](
                params=self.cache_tokenizer_path
            )
            self._is_trained = True
        else:
            self.tokenizer = self.TOKENIZER_MAPPER[tokenizer_name]()
            self._is_trained = False

    def encode(
        self, music_file: MidoMidiFile, symbolic_format: SymbolicMusicFormat
    ) -> list[int]:
        if symbolic_format == SymbolicMusicFormat.ABC:
            raise ValueError("ABC format is not supported for tokenizer")
        if not self._is_trained:
            logger.warning(f"Tokenizer is not trained")
        midi_score = convert_mido_to_score(
            music_file
        )  # Change from mido into symusic.Score
        tok_sequence = self.tokenizer(midi_score)[0].ids[: self.sequence_length - 2]
        tok_sequence = (
            [self.bos_token_id] + tok_sequence + [self.eos_token_id]
        )  # TODO add case for cpword (more dimensional)
        return tok_sequence

    def decode(
        self,
        tokens: list[int],
        symbolic_format: SymbolicMusicFormat,
        output_path: Path,
        export_format: ExportFormat = ExportFormat.WAV,
    ) -> Path:
        if symbolic_format == SymbolicMusicFormat.ABC:
            raise ValueError("ABC format is not supported for tokenizer")
        if not self._is_trained:
            logger.warning(f"Tokenizer is not trained")
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

    def train(self, files_path: Path) -> None:
        files_paths = []
        for item in files_path.iterdir():
            if item.is_dir():
                files_paths.extend(item.iterdir())
            else:
                files_paths.append(item)

        self.tokenizer.train(
            vocab_size=self.vocab_size,
            files_paths=files_paths,
            model=self.training_model,
        )
        self._is_trained = True
        self.save()

    def save(self) -> None:
        logger.info(
            f"Saving tokenizer {self.tokenizer_name} to {self.cache_tokenizer_path}"
        )
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
