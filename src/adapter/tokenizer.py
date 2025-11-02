from pathlib import Path
from omegaconf import DictConfig
import logging as lg
from io import BytesIO

from miditok import REMI, MIDILike, TSD, Structured, CPWord, Octuple, MuMIDI, MMM
from mido import MidiFile as MidoMidiFile

from src.utils import base_weights_path, base_dataset_path, convert_mido_to_score
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

    def __init__(
        self, tokenizer_name: str, training_model: str, vocab_size: int
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.training_model = training_model
        self.vocab_size = vocab_size
        self.tokenizer_base_path = base_weights_path / "tokenizers"
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
    ) -> list[list[int]]:
        if symbolic_format == SymbolicMusicFormat.ABC:
            raise ValueError("ABC format is not supported for tokenizer")
        if not self._is_trained:
            raise ValueError(f"Tokenizer is not trained")
        midi_score = convert_mido_to_score(
            music_file
        )  # Change from mido into symusic.Score
        tok_sequence = self.tokenizer(midi_score)[0].ids
        return [tok_sequence]

    def decode(
        self,
        tensor_tokens: list[list[int]],
        symbolic_format: SymbolicMusicFormat,
        output_path: Path,
        export_format: ExportFormat = ExportFormat.WAV,
    ) -> Path:
        from src.utils import export_to_wav

        if symbolic_format == SymbolicMusicFormat.ABC:
            raise ValueError("ABC format is not supported for tokenizer")
        if not self._is_trained:
            raise ValueError(f"Tokenizer is not trained")
        score = self.tokenizer.decode(tensor_tokens)
        match export_format:
            case ExportFormat.SYMBOLIC:
                score.dump_midi(path=output_path.as_posix())
                return output_path
            case ExportFormat.WAV:
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
        self.tokenizer.save_params(out_path=self.cache_tokenizer_path)

    @property
    def is_trained(self):
        return self._is_trained

    @property
    def cache_tokenizer_path(self) -> Path:
        return (
            self.tokenizer_base_path
            / f"{self.tokenizer_name}_{self.training_model}_{self.vocab_size}.json"
        )


def get_tokenizer(cfg: DictConfig, force: bool = False) -> Tokenizer:
    tokenizer = Tokenizer(
        tokenizer_name=cfg.tokenizer,
        training_model=cfg.training_model,
        vocab_size=cfg.vocab_size,
    )
    if force or not tokenizer.is_trained:
        dataset_path = (base_dataset_path / cfg.dataset_name).resolve(True)
        tokenizer.train(files_path=dataset_path)
    return tokenizer
