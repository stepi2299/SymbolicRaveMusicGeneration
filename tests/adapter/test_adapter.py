from omegaconf import DictConfig
import pytest

from src.adapter import MusicAdapter, MusicPatchilizer, Tokenizer
from src.constants import SymbolicMusicFormat


class TestMusicAdapter:
    @staticmethod
    def test_create_patchilizer_adapter(patchilizer_config):
        adapter = MusicAdapter(patchilizer_config, "midi")
        assert isinstance(adapter._adapter, MusicPatchilizer)

    @staticmethod
    def test_create_tokenizer_adapter(tokenizer_config):
        adapter = MusicAdapter(tokenizer_config, "midi")
        assert isinstance(adapter._adapter, Tokenizer)

    @staticmethod
    def test_create_unknown_adapter():
        config = {"name": "unknown", "params": {}}
        config = DictConfig(config)
        with pytest.raises(ValueError):
            MusicAdapter(config, "midi")

    @staticmethod
    def test_encode_patchilizer_midi(patchilizer_adapter_midi, loaded_midi_song):
        encoded = patchilizer_adapter_midi.encode(loaded_midi_song)
        assert isinstance(encoded, list)
        for patch in encoded:
            assert isinstance(patch, list)
            for value in patch:
                assert isinstance(value, int)

    @staticmethod
    def test_encode_tokenizer_midi(tokenizer_adapter_midi, loaded_midi_song):
        encoded = tokenizer_adapter_midi.encode(loaded_midi_song)
        assert isinstance(encoded, list)

    @staticmethod
    def test_encode_patchilizer_abc(patchilizer_adapter_abc, loaded_abc_song):
        encoded = patchilizer_adapter_abc.encode(loaded_abc_song)
        assert isinstance(encoded, list)
        for patch in encoded:
            assert isinstance(patch, list)
            for value in patch:
                assert isinstance(value, int)

    @staticmethod
    @staticmethod
    def test_get_two_dim_padded_sequence(patchilizer_adapter_abc: MusicAdapter):
        new_sequence_len = 8
        new_patch_size = 5
        patchilizer_adapter_abc._adapter.patch_size = new_patch_size
        patchilizer_adapter_abc._adapter.sequence_length = new_sequence_len
        example_patch = [[5, 5, 5, 5, 5], [4, 4, 4, 4, 4]]
        result_padded_patches = patchilizer_adapter_abc.get_padded_sequence(example_patch)
        assert isinstance(result_padded_patches, list)
        assert len(result_padded_patches) == new_sequence_len
        for i, patch in enumerate(result_padded_patches):
            assert len(patch) == new_patch_size
            if i > 1:
                assert result_padded_patches[i] == [patchilizer_adapter_abc.pad_token_id] * new_patch_size

    @staticmethod
    def test_get_single_padded_sequence(tokenizer_adapter_midi: MusicAdapter):
        new_sequence_len = 20
        new_patch_size = 1
        example_sequence = [
            tokenizer_adapter_midi.bos_token_id,
            10, 50, 20, 3, 4, 5, 6, 7, 8, 9, 10,
            tokenizer_adapter_midi.eos_token_id
        ]
        tokenizer_adapter_midi._adapter.patch_size = new_patch_size
        tokenizer_adapter_midi._adapter.sequence_length = new_sequence_len
        result_padded_sequence = tokenizer_adapter_midi.get_padded_sequence(example_sequence)
        assert isinstance(result_padded_sequence, list)
        assert len(result_padded_sequence) == tokenizer_adapter_midi.sequence_length
        assert result_padded_sequence[len(example_sequence)-1] == tokenizer_adapter_midi.eos_token_id
        assert result_padded_sequence[-1] == tokenizer_adapter_midi.pad_token_id
        assert result_padded_sequence[0] == tokenizer_adapter_midi.bos_token_id



