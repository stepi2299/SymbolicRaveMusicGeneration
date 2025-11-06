import torch
import pytest

from src.dataset.symbolic_music_dataset import MidiDataset, AbcDataset


class TestMidiDataset:
    @staticmethod
    def test_basic_creation(midi_data_path, patchilizer_adapter_midi):
        dataset = MidiDataset(midi_data_path, patchilizer_adapter_midi)
        assert (
            len(dataset.music_files_paths) == 3
        ), "MidiDataset should have 3 music files"
        assert len(dataset) == 3, "MidiDataset should have 3 item"

    @staticmethod
    def test_getitem(midi_data_path, patchilizer_adapter_midi):
        dataset = MidiDataset(midi_data_path, patchilizer_adapter_midi)
        for i in range(len(dataset)):
            item = dataset[i]
            assert len(item.shape) == 2, "MidiDataset should return a 2D tensor"
            assert isinstance(
                item, torch.Tensor
            ), "MidiDataset should return a 2D tensor"
            assert len(item) > 1, "MidiDataset should return a 2D tensor"
            assert isinstance(
                item[0], torch.Tensor
            ), "MidiDataset should return a 2D tensor"
            assert isinstance(
                item[0][0], torch.Tensor
            ), "MidiDataset should return a 2D tensor"
            assert len(item[0]) == dataset.music_adapter._adapter.patch_size


class TestAbcDataset:
    @staticmethod
    def test_basic_creation(abc_data_path, patchilizer_adapter_abc):
        dataset = AbcDataset(abc_data_path, patchilizer_adapter_abc)
        assert (
            len(dataset.music_files_paths) == 3
        ), "MidiDataset should have 3 music files"
        assert len(dataset) == 3, "MidiDataset should have 3 item"

    @staticmethod
    def test_getitem(abc_data_path, patchilizer_adapter_abc):
        dataset = AbcDataset(abc_data_path, patchilizer_adapter_abc)
        for i in range(len(dataset)):
            item = dataset[i]
            assert len(item.shape) == 2, "MidiDataset should return a 2D tensor"
            assert isinstance(
                item, torch.Tensor
            ), "MidiDataset should return a 2D tensor"
            assert len(item) > 1, "MidiDataset should return a 2D tensor"
            assert isinstance(
                item[0], torch.Tensor
            ), "MidiDataset should return a 2D tensor"
            assert isinstance(
                item[0][0], torch.Tensor
            ), "MidiDataset should return a 2D tensor"
            assert len(item[0]) == dataset.music_adapter._adapter.patch_size

    @staticmethod
    def test_set_single_data_training(midi_data_path, patchilizer_adapter_midi):
        dataset = MidiDataset(midi_data_path, patchilizer_adapter_midi, single_data_training=True)
        assert len(dataset) == 1, "MidiDataset should have 3 item"
        assert len(dataset.music_files_paths) == 1
        assert dataset._single_data_training is True
        assert isinstance(dataset._index_of_overtrained, int)

    @staticmethod
    def test_set_single_data_training_with_spec_index(midi_data_path, patchilizer_adapter_midi):
        dataset = MidiDataset(midi_data_path, patchilizer_adapter_midi, single_data_training=True, index_of_overtrained=1)
        assert len(dataset) == 1, "MidiDataset should have 3 item"
        assert len(dataset.music_files_paths) == 1
        assert dataset._single_data_training is True
        assert dataset._index_of_overtrained == 1

    @staticmethod
    def test_set_single_data_index_out_of_range(midi_data_path, patchilizer_adapter_midi):
        with pytest.raises(ValueError):
            MidiDataset(midi_data_path, patchilizer_adapter_midi, single_data_training=True, index_of_overtrained=5)