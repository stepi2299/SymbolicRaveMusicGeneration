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
            assert isinstance(item, list), "MidiDataset should return a list"
            assert len(item) > 1, "MidiDataset should return a list of lists"
            assert isinstance(
                item[0], list
            ), "MidiDataset should return a list of lists"
            assert isinstance(
                item[0][0], int
            ), "MidiDataset should return a list of lists of ints"
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
            assert isinstance(item, list), "MidiDataset should return a list"
            assert len(item) > 1, "MidiDataset should return a list of lists"
            assert isinstance(
                item[0], list
            ), "MidiDataset should return a list of lists"
            assert isinstance(
                item[0][0], int
            ), "MidiDataset should return a list of lists of ints"
            assert len(item[0]) == dataset.music_adapter._adapter.patch_size
