import pytest
from pathlib import Path
from omegaconf import DictConfig

from src.adapter.adapter import MusicAdapter
from src.dataset.utils import load_midi_song, load_abc_song
from src.dataset.symbolic_music_dataset import MidiDataset


@pytest.fixture(scope="session", name="test_data_path")
def fixture_test_data_path() -> Path:
    base_path = Path.cwd() / "tests" / "data"
    return base_path.resolve(True)


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


@pytest.fixture(scope="session", name="loaded_midi_song")
def fixture_loaded_midi_song(midi_file: Path) -> str:
    return load_midi_song(midi_file)


@pytest.fixture(scope="session", name="loaded_abc_song")
def fixture_loaded_abc_song(abc_file: Path) -> str:
    return load_abc_song(abc_file)


@pytest.fixture(scope="session", name="patchilizer_config")
def fixture_patchilizer_config() -> DictConfig:
    config = {"name": "patchilizer", "params": {"patch_length": 512, "patch_size": 64}}
    return DictConfig(config)


@pytest.fixture(scope="session", name="tokenizer_config")
def fixture_tokenizer_config() -> DictConfig:
    config = {
        "name": "tokenizer",
        "params": {
            "tokenizer_name": "tsd",
            "training_model": "BPE",
            "vocab_size": 1000,
        },
    }
    return DictConfig(config)


@pytest.fixture(scope="session", name="patchilizer_adapter")
def fixture_patchilizer_adapter(patchilizer_config) -> MusicAdapter:
    return MusicAdapter(patchilizer_config)


@pytest.fixture(scope="session", name="tokenizer_adapter")
def fixture_tokenizer_adapter(tokenizer_config) -> MusicAdapter:
    return MusicAdapter(tokenizer_config)


@pytest.fixture(scope="session", name="midi_symbolic_dataset")
def fixture_midi_symbolic_dataset(midi_data_path, patchilizer_adapter) -> MidiDataset:
    return MidiDataset(dataset_path=midi_data_path, adapter=patchilizer_adapter)
