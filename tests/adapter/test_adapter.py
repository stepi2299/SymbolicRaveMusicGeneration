from omegaconf import DictConfig
import pytest
from src.adapter.adapter import MusicAdapter, MusicPatchilizer, Tokenizer


class TestMusicAdapter:
    @staticmethod
    def test_create_patchilizer_adapter(patchilizer_config):
        adapter = MusicAdapter(patchilizer_config)
        assert isinstance(adapter._adapter, MusicPatchilizer)

    @staticmethod
    def test_create_tokenizer_adapter(tokenizer_config):
        adapter = MusicAdapter(tokenizer_config)
        assert isinstance(adapter._adapter, Tokenizer)

    @staticmethod
    def test_create_unknown_adapter():
        config = {"name": "unknown", "params": {}}
        config = DictConfig(config)
        with pytest.raises(ValueError):
            MusicAdapter(config)

    @staticmethod
    def test_encode_patchilizer_midi(patchilizer_adapter, loaded_midi_song):
        encoded = patchilizer_adapter.encode(loaded_midi_song)
        assert isinstance(encoded, list)
        for patch in encoded:
            assert isinstance(patch, list)
            for value in patch:
                assert isinstance(value, int)

    @staticmethod
    @pytest.mark.skip(reason="Tokenizer not implemented properly yet")
    def test_encode_tokenizer_midi(tokenizer_adapter, loaded_midi_song):
        encoded = tokenizer_adapter.encode(loaded_midi_song)
        assert isinstance(encoded, list)

    @staticmethod
    def test_encode_patchilizer_abc(patchilizer_adapter, loaded_abc_song):
        encoded = patchilizer_adapter.encode(loaded_abc_song)
        assert isinstance(encoded, list)
        for patch in encoded:
            assert isinstance(patch, list)
            for value in patch:
                assert isinstance(value, int)
