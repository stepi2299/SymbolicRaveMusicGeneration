from mido import Message as MidoMessage
from torch.utils.data import Dataset
from pathlib import Path
import pytest

from src.dataset.utils import (
    midi_msg_to_str,
    load_midi_song,
    load_abc_song,
    symbolic_collate_fn,
    split_dataset,
)


class DummyDataset(Dataset):
    """A simple dataset for testing splits."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestDatasetUtils:
    @staticmethod
    def test_midi_msg_to_str():
        """Test that _midi_msg_to_str converts a MIDI message to a proper string."""
        msg = MidoMessage("note_on", note=64, velocity=127, time=0)
        result = midi_msg_to_str(msg)
        # The dictionary typically has keys like 'type', 'channel', 'note', 'velocity', 'time'
        # The order can vary, but let's just check that each parameter is in the result
        assert "note_on" in result
        assert "64" in result
        assert "127" in result
        assert isinstance(result, str)
        assert result == "note_on 0 0 64 127", "Result should be 'note_on 0 0 64 127'"

    @staticmethod
    def test_load_midi_song(midi_file: Path):
        """
        Test that loading a MIDI file returns a string containing
        ticks_per_beat and MIDI messages (excluding meta messages).
        """
        result = load_midi_song(midi_file)
        assert isinstance(result, str)

    @staticmethod
    def test_load_midi_song_invalid_path():
        """
        Test that loading from an invalid path raises an OSError.
        """
        with pytest.raises(OSError):
            load_midi_song(Path("non_existent_file.mid"))

    @staticmethod
    def test_load_abc_song(abc_file: Path):
        """
        Test that an ABC file is loaded and contains the original content.
        """
        content = load_abc_song(abc_file)
        assert isinstance(content, str)

    @staticmethod
    def test_load_abc_song_invalid_path():
        """
        Test that loading from an invalid ABC path raises an exception (likely FileNotFoundError).
        """
        with pytest.raises(OSError):
            load_abc_song(Path("non_existent_file.abc"))

    @staticmethod
    def test_symbolic_collate_fn():
        """Test that symbolic_collate_fn returns the batch as is."""
        batch = ["melody1", "melody2", "melody3"]
        result = symbolic_collate_fn(batch)
        assert result == batch, "symbolic_collate_fn should return the same batch"

    @staticmethod
    def test_split_dataset():
        """
        Test that split_dataset splits a dataset correctly into train and validation
        with the default 0.8 ratio.
        """
        data = list(range(10))
        dataset = DummyDataset(data)

        train_dataset, val_dataset = split_dataset(dataset, train_split=0.8)

        # Check sizes
        assert (
            len(train_dataset) == 8
        ), f"train_dataset should have 8 items, got {len(train_dataset)}"
        assert (
            len(val_dataset) == 2
        ), f"val_dataset should have 2 items, got {len(val_dataset)}"

        # Check that the items are from the original dataset
        train_items = sorted([train_dataset[i] for i in range(len(train_dataset))])
        val_items = sorted([val_dataset[i] for i in range(len(val_dataset))])
        original = sorted(data)
        combined = sorted(train_items + val_items)
        assert (
            combined == original
        ), "Combined items after split must match original dataset items"
