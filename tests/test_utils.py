from mido import Message as MidoMessage
from torch.utils.data import Dataset
from pathlib import Path
import pytest
import torch

from src.utils import (
    midi_msg_to_str,
    load_midi_song,
    load_abc_song,
    symbolic_collate_fn,
    split_dataset,
    filter_state_dict_keys,
    rename_state_dict_keys,
    remove_state_dict_keys,
    save_midi_song_from_string,
    convert_midi_to_string,
    convert_mido_to_score,
)
from mido import MidiFile as MidoMidiFile
from symusic import Score


class DummyDataset(Dataset):
    """A simple dataset for testing splits."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestDatasetUtils:

    # @staticmethod
    # def test_symbolic_collate_fn():
    #     """Test that symbolic_collate_fn returns the batch as is."""
    #     batch = ["melody1", "melody2", "melody3"]
    #     result = symbolic_collate_fn(batch)
    #     assert result == batch, "symbolic_collate_fn should return the same batch"

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


class TestFilterKeys:
    @staticmethod
    def test_filter_state_dict_keys():
        # Create a sample state dictionary with keys containing 'music_enc'
        state_dict = {
            "music_enc.layer1.weight": torch.tensor([1, 2, 3]),
            "music_enc.layer2.bias": torch.tensor([4, 5, 6]),
            "decoder.layer1.weight": torch.tensor([7, 8, 9]),
            "other_key": torch.tensor([10, 11, 12]),
        }

        # Define the expected output after renaming keys
        expected_state_dict = {
            "layer1.weight": torch.tensor([1, 2, 3]),
            "layer2.bias": torch.tensor([4, 5, 6]),
        }

        # Call the function with the sample state dictionary
        filtered_state_dict = filter_state_dict_keys(state_dict, "music_enc")

        # Assert that the output matches the expected output
        assert filtered_state_dict.keys() == expected_state_dict.keys()
        assert torch.equal(
            filtered_state_dict["layer1.weight"], expected_state_dict["layer1.weight"]
        )

    @staticmethod
    def test_filter_state_dict_keys_no_match():
        # Create a sample state dictionary with keys that do not contain 'music_enc'
        state_dict = {
            "encoder.layer1.weight": torch.tensor([1, 2, 3]),
            "decoder.layer2.bias": torch.tensor([4, 5, 6]),
            "other_key": torch.tensor([7, 8, 9]),
        }

        # Define the expected output (should be the same as the input)
        expected_state_dict = {}

        # Call the function with the sample state dictionary
        filtered_state_dict = filter_state_dict_keys(state_dict, "music_enc")

        # Assert that the output matches the expected output
        assert filtered_state_dict == expected_state_dict

    @staticmethod
    def test_filter_state_dict_keys_empty():
        # Create an empty state dictionary
        state_dict = {}

        # Define the expected output (should also be empty)
        expected_state_dict = {}

        # Call the function with the empty state dictionary
        filtered_state_dict = filter_state_dict_keys(state_dict, "music_enc")

        # Assert that the output matches the expected output
        assert filtered_state_dict == expected_state_dict

    @staticmethod
    def test_filter_and_replace():
        # Create a sample state dictionary with keys containing 'music_enc'
        state_dict = {
            "music_enc.layer1.weight": torch.tensor([1, 2, 3]),
            "music_enc.layer2.bias": torch.tensor([4, 5, 6]),
            "decoder.layer1.weight": torch.tensor([7, 8, 9]),
            "other_key": torch.tensor([10, 11, 12]),
            "music_enc_extra.layer3.weight": torch.tensor(
                [13, 14, 15]
            ),  # Should not be modified
        }

        # Define the expected output after replacing 'music_enc' with 'patch_encoder'
        expected_state_dict = {
            "patch_encoder.layer1.weight": torch.tensor([1, 2, 3]),
            "patch_encoder.layer2.bias": torch.tensor([4, 5, 6]),
            "decoder.layer1.weight": torch.tensor([7, 8, 9]),
            "other_key": torch.tensor([10, 11, 12]),
            "music_enc_extra.layer3.weight": torch.tensor(
                [13, 14, 15]
            ),  # Should remain unchanged
        }

        # Call the function with the sample state dictionary
        filtered_state_dict = rename_state_dict_keys(
            state_dict, "music_enc", "patch_encoder"
        )

        # Assert that the output matches the expected output
        assert filtered_state_dict.keys() == expected_state_dict.keys()
        assert torch.equal(
            filtered_state_dict["patch_encoder.layer1.weight"],
            expected_state_dict["patch_encoder.layer1.weight"],
        )

    @staticmethod
    def test_filter_and_replace_substring():
        # Create a sample state dictionary with keys containing 'music_enc'
        state_dict = {
            "enc.encoder.weight": torch.tensor([1, 2, 3]),
            "music_enc.layer2.bias": torch.tensor([4, 5, 6]),
            "decoder.enc.weight": torch.tensor([7, 8, 9]),
            "other_key": torch.tensor([10, 11, 12]),
            "music_enc_extra.layer3.encoder": torch.tensor(
                [13, 14, 15]
            ),  # Should not be modified
        }

        # Define the expected output after replacing 'music_enc' with 'patch_encoder'
        expected_state_dict = {
            "dududud.encoder.weight": torch.tensor([1, 2, 3]),
            "music_enc.layer2.bias": torch.tensor([4, 5, 6]),
            "decoder.dududud.weight": torch.tensor([7, 8, 9]),
            "other_key": torch.tensor([10, 11, 12]),
            "music_enc_extra.layer3.encoder": torch.tensor(
                [13, 14, 15]
            ),  # Should remain unchanged
        }

        # Call the function with the sample state dictionary
        filtered_state_dict = rename_state_dict_keys(state_dict, "enc", "dududud")

        # Assert that the output matches the expected output
        assert filtered_state_dict.keys() == expected_state_dict.keys()
        assert torch.equal(
            filtered_state_dict["dududud.encoder.weight"],
            expected_state_dict["dududud.encoder.weight"],
        )

    @staticmethod
    def test_filter_and_replace_substring_no_match():
        # Create a sample state dictionary with keys that do not contain 'music_enc'
        state_dict = {
            "encoder.layer1.weight": torch.tensor([1, 2, 3]),
            "decoder.layer2.bias": torch.tensor([4, 5, 6]),
            "other_key": torch.tensor([7, 8, 9]),
        }

        # Define the expected output (should be the same as the input)
        expected_state_dict = state_dict.copy()

        # Call the function with the sample state dictionary
        filtered_state_dict = rename_state_dict_keys(
            state_dict, "music_enc", "patch_encoder"
        )

        # Assert that the output matches the expected output
        assert filtered_state_dict == expected_state_dict

    @staticmethod
    def test_filter_and_replace_substring_empty():
        # Create an empty state dictionary
        state_dict = {}

        # Define the expected output (should also be empty)
        expected_state_dict = {}

        # Call the function with the empty state dictionary
        filtered_state_dict = rename_state_dict_keys(
            state_dict, "music_enc", "patch_encoder"
        )

        # Assert that the output matches the expected output
        assert filtered_state_dict == expected_state_dict

    @staticmethod
    def test_remove_state_dict_keys():
        # Create a sample state dictionary
        state_dict = {
            "layer1.weight": torch.tensor([1, 2, 3]),
            "layer2.bias": torch.tensor([4, 5, 6]),
            "layer3.weight": torch.tensor([7, 8, 9]),
            "other_key": torch.tensor([10, 11, 12]),
        }

        # Define keys to remove
        keys_to_remove = ["layer2", "other_key"]

        # Define the expected output after removing specified keys
        expected_state_dict = {
            "layer1.weight": torch.tensor([1, 2, 3]),
            "layer3.weight": torch.tensor([7, 8, 9]),
        }

        # Call the function with the sample state dictionary and keys to remove
        filtered_state_dict = remove_state_dict_keys(state_dict, keys_to_remove)

        # Assert that the output matches the expected output
        assert filtered_state_dict.keys() == expected_state_dict.keys()
        assert torch.equal(
            filtered_state_dict["layer1.weight"], expected_state_dict["layer1.weight"]
        )

    @staticmethod
    def test_remove_state_dict_keys_no_match():
        # Create a sample state dictionary
        state_dict = {
            "layer1.weight": torch.tensor([1, 2, 3]),
            "layer2.bias": torch.tensor([4, 5, 6]),
            "layer3.weight": torch.tensor([7, 8, 9]),
        }

        # Define keys to remove that do not exist in the state dictionary
        keys_to_remove = ["non_existent_key1", "non_existent_key2"]

        # Define the expected output (should be the same as the input)
        expected_state_dict = state_dict.copy()

        # Call the function with the sample state dictionary and keys to remove
        filtered_state_dict = remove_state_dict_keys(state_dict, keys_to_remove)

        # Assert that the output matches the expected output
        assert filtered_state_dict == expected_state_dict

    @staticmethod
    def test_remove_state_dict_keys_empty():
        # Create an empty state dictionary
        state_dict = {}

        # Define keys to remove
        keys_to_remove = ["layer1.weight", "layer2.bias"]

        # Define the expected output (should also be empty)
        expected_state_dict = {}

        # Call the function with the empty state dictionary and keys to remove
        filtered_state_dict = remove_state_dict_keys(state_dict, keys_to_remove)

        # Assert that the output matches the expected output
        assert filtered_state_dict == expected_state_dict


class TestSymbolicMusicTools:
    @staticmethod
    def test_save_midi_song_from_string_and_load(midi_file):
        ret_file = load_midi_song(midi_file)
        assert isinstance(ret_file, MidoMidiFile)

    @staticmethod
    def test_save_midi_song_from_string_and_load_but_abc(abc_file):
        with pytest.raises(Exception):
            load_midi_song(abc_file)

    @staticmethod
    def test_load_midi_song_invalid_path():
        """
        Test that loading from an invalid path raises an OSError.
        """
        with pytest.raises(OSError):
            load_midi_song(Path("non_existent_file.mid"))

    @staticmethod
    def test_convert_mido_midi_to_miditoolkit(loaded_midi_song):
        assert isinstance(loaded_midi_song, MidoMidiFile)
        miditoolkit_midi = convert_mido_to_score(loaded_midi_song)
        assert isinstance(miditoolkit_midi, Score)

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
    def test_convert_midi_to_string(loaded_midi_song):
        """
        Test that loading a MIDI file returns a string containing
        ticks_per_beat and MIDI messages (excluding meta messages).
        """
        result = convert_midi_to_string(loaded_midi_song)
        assert isinstance(result, str)

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
    def test_save_midi_song_from_string(midi_song_as_string, tmp_path):
        new_file_path = tmp_path / "new_file.mid"
        save_midi_song_from_string(midi_song_as_string, new_file_path)
        assert new_file_path.exists(), "The new MIDI file should exist"

    @staticmethod
    @pytest.mark.skip(reason="Not working properly yet")
    def test_convert_abc_to_score(loaded_abc_song_3, test_results_path, request):
        loaded_abc_song = loaded_abc_song_3
        folder_path = test_results_path / f"{request.node.parent.name}.{request.node.name}"
        folder_path.mkdir(parents=True, exist_ok=True)
        original_abc_path = folder_path / "original.abc"
        with open(original_abc_path, "w") as f:
            f.write(loaded_abc_song)
        decoded_abc_path = folder_path / "decoded.abc"
        score = Score.from_abc(loaded_abc_song)
        score.dump_abc(path=decoded_abc_path.as_posix())
        assert decoded_abc_path.exists(), "The decoded ABC file should exist"
        with open(decoded_abc_path, "r") as f:
            decoded_content = f.read()
        assert (decoded_content == loaded_abc_song), "The decoded ABC content should match the original"
