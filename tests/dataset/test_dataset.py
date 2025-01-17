import torch

from src.dataset.symbolic_music_dataset import MidiDataset, AbcDataset


class TestMidiDataset:
    @staticmethod
    def test_basic_creation(midi_data_path, patchilizer_adapter):
        dataset = MidiDataset(midi_data_path, patchilizer_adapter)
        assert (
            len(dataset.music_files_paths) == 3
        ), "MidiDataset should have 3 music files"
        assert len(dataset) == 3, "MidiDataset should have 3 item"

    @staticmethod
    def test_getitem(midi_data_path, patchilizer_adapter):
        dataset = MidiDataset(midi_data_path, patchilizer_adapter)
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
    def test_basic_creation(abc_data_path, patchilizer_adapter):
        dataset = AbcDataset(abc_data_path, patchilizer_adapter)
        assert (
            len(dataset.music_files_paths) == 3
        ), "MidiDataset should have 3 music files"
        assert len(dataset) == 3, "MidiDataset should have 3 item"

    @staticmethod
    def test_getitem(abc_data_path, patchilizer_adapter):
        dataset = AbcDataset(abc_data_path, patchilizer_adapter)
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
