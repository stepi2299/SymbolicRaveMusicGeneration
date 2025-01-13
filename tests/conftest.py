import pytest
from pathlib import Path


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
