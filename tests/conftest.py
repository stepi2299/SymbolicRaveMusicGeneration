import pytest
from pathlib import Path
from omegaconf import DictConfig
import shutil
import tempfile

from mido import MidiFile as MidoMidiFile

from src.adapter import MusicAdapter, Tokenizer, MusicPatchilizer
from src.utils import load_midi_song, load_abc_song, convert_midi_to_string
from src.dataset.symbolic_music_dataset import MidiDataset

def prepare_test_results_dir():
    test_result_path = Path("tests") / "results"
    if test_result_path.exists():
        shutil.rmtree(test_result_path)
    test_result_path.mkdir(parents=True, exist_ok=True)

prepare_test_results_dir()


@pytest.fixture(scope="session", name="test_data_path")
def fixture_test_data_path() -> Path:
    base_path = Path.cwd() / "tests" / "data"
    return base_path.resolve(True)


@pytest.fixture(scope="session", name="test_results_path")
def fixture_test_results_path(test_data_path) -> Path:
    results_path = Path("tests") / "results"
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    return results_path



@pytest.fixture(scope="session", name="midi_data_path")
def fixture_midi_data_path(test_data_path) -> Path:
    return test_data_path / "midi"


@pytest.fixture(scope="session", name="abc_data_path")
def fixture_abc_data_path(test_data_path) -> Path:
    return test_data_path / "abc"


@pytest.fixture(scope="session", name="midi_file")
def fixture_midi_file(midi_data_path) -> Path:
    return midi_data_path / "example_1.mid"


@pytest.fixture(scope="session", name="abc_file")
def fixture_abc_file(abc_data_path) -> Path:
    return abc_data_path / "example_2.abc"

@pytest.fixture(scope="session", name="abc_file_2")
def fixture_abc_file_2(abc_data_path) -> Path:
    return abc_data_path / "example_1.abc"

@pytest.fixture(scope="session", name="abc_file_3")
def fixture_abc_file_3(abc_data_path) -> Path:
    return abc_data_path / "example_3.abc"



@pytest.fixture(scope="session", name="loaded_midi_song")
def fixture_loaded_midi_song(midi_file: Path) -> MidoMidiFile:
    return load_midi_song(midi_file)

@pytest.fixture(scope="session", name="midi_song_as_string")
def fixture_midi_song_as_string(loaded_midi_song) -> str:
    # Convert the loaded MIDI song to a string representation
    return convert_midi_to_string(loaded_midi_song)


@pytest.fixture(scope="session", name="loaded_abc_song")
def fixture_loaded_abc_song(abc_file: Path) -> str:
    return load_abc_song(abc_file)

@pytest.fixture(scope="session", name="loaded_abc_song_2")
def fixture_loaded_abc_song_2(abc_file_2: Path) -> str:
    return load_abc_song(abc_file_2)

@pytest.fixture(scope="session", name="loaded_abc_song_3")
def fixture_loaded_abc_song_3(abc_file_3: Path) -> str:
    return load_abc_song(abc_file_3)


@pytest.fixture(scope="session", name="patchilizer_config")
def fixture_patchilizer_config() -> DictConfig:
    config = {"name": "patchilizer", "params": {"sequence_length": 512, "patch_size": 64}}
    return DictConfig(config)


@pytest.fixture(scope="session", name="tokenizer_config")
def fixture_tokenizer_config() -> DictConfig:
    config = {
        "name": "tokenizer",
        "params": {
            "tokenizer": "tsd",
            "training_model": "BPE",
            "vocab_size": 1000,
            "dataset_name": "maestro",
            "sequence_length": 1024,
        },
    }
    return DictConfig(config)


@pytest.fixture(scope="session", name="patchilizer_adapter_midi")
def fixture_patchilizer_adapter_midi(patchilizer_config) -> MusicAdapter:
    return MusicAdapter(patchilizer_config, "midi")

@pytest.fixture(scope="session", name="patchilizer_adapter_abc")
def fixture_patchilizer_adapter_abc(patchilizer_config) -> MusicAdapter:
    return MusicAdapter(patchilizer_config, "abc")


@pytest.fixture(scope="session", name="tokenizer_adapter_midi")
def fixture_tokenizer_adapter_midi(tokenizer_config) -> MusicAdapter:
    return MusicAdapter(tokenizer_config, "midi")


@pytest.fixture(scope="session", name="midi_symbolic_dataset")
def fixture_midi_symbolic_dataset(midi_data_path, patchilizer_adapter_midi) -> MidiDataset:
    return MidiDataset(dataset_path=midi_data_path, adapter=patchilizer_adapter_midi)

@pytest.fixture(scope="function", name="remi_tokenizer")
def fixture_remi_tokenizer() -> Tokenizer:
    tokenizer_config = {
        "tokenizer_name": "remi",
        "training_model": "BPE",
        "vocab_size": 1000,
        "sequence_length": 1024

    }
    Tokenizer.TOKENIZER_BASE_PATH = Path(tempfile.mkdtemp())
    tokenizer = Tokenizer(**tokenizer_config)
    return tokenizer


@pytest.fixture(scope="function", name="patchilizer")
def fixture_patchilizer() -> MusicPatchilizer:
    tokenizer_config = {"sequence_length": 512, "patch_size": 64}
    return MusicPatchilizer(**tokenizer_config)
