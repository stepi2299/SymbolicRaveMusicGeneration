from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils import convert_mido_to_score, export_to_wav
from src.adapter import Tokenizer
from src.constants import SymbolicMusicFormat, ExportFormat


class TestTokenizer:
    @staticmethod
    def test_basic_creation(remi_tokenizer: Tokenizer):
        assert isinstance(remi_tokenizer, Tokenizer)
        assert remi_tokenizer.vocab_size > 0
        assert remi_tokenizer.is_trained is False
        assert remi_tokenizer.training_model == "BPE"

    @staticmethod
    @patch("src.adapter.tokenizer.logger.warning")
    def test_encode_without_training(mock_logger: MagicMock, remi_tokenizer: Tokenizer, loaded_midi_song):
        remi_tokenizer.encode(loaded_midi_song, symbolic_format=SymbolicMusicFormat.MIDI)
        mock_logger.assert_called_once_with("Tokenizer is not trained")

    @staticmethod
    def test_encode_abc_song(remi_tokenizer: Tokenizer, loaded_abc_song):
        try:
            remi_tokenizer.encode(loaded_abc_song, symbolic_format=SymbolicMusicFormat.ABC)
        except ValueError as e:
            assert str(e) == "ABC format is not supported for tokenizer"

    @staticmethod
    def test_encode(remi_tokenizer: Tokenizer, loaded_midi_song):
        assert remi_tokenizer.is_trained is False
        encoded = remi_tokenizer.encode(loaded_midi_song, symbolic_format=SymbolicMusicFormat.MIDI)
        assert isinstance(encoded, list)
        assert len(encoded) == 1  # TODO check in future
        for patch in encoded:
            assert isinstance(patch, list)
            for token in patch:
                assert isinstance(token, int)

    @staticmethod
    def test_decode_midi_wav(remi_tokenizer: Tokenizer, loaded_midi_song, test_results_path, request):
        assert remi_tokenizer.is_trained is False
        loaded_score = convert_mido_to_score(loaded_midi_song)

        folder_path = test_results_path / f"{request.node.parent.name}_{request.node.name}"
        folder_path.mkdir(parents=True, exist_ok=True)
        original_wav_path = folder_path / "original.wav"
        decoded_wav_path = folder_path / "decoded.wav"
        export_to_wav(loaded_score, original_wav_path)
        encoded = remi_tokenizer.encode(loaded_midi_song, symbolic_format=SymbolicMusicFormat.MIDI)
        output_file = remi_tokenizer.decode(encoded, symbolic_format=SymbolicMusicFormat.MIDI, output_path=decoded_wav_path)
        assert isinstance(output_file, Path)
        assert output_file.exists()
        assert output_file == decoded_wav_path

    @staticmethod
    def test_decode_midi_midi(remi_tokenizer: Tokenizer, loaded_midi_song, test_results_path, request):
        assert remi_tokenizer.is_trained is False
        folder_path = test_results_path / f"{request.node.parent.name}_{request.node.name}"
        folder_path.mkdir(parents=True, exist_ok=True)
        decoded_wav_path = folder_path / "decoded.mid"
        encoded = remi_tokenizer.encode(loaded_midi_song, symbolic_format=SymbolicMusicFormat.MIDI)
        output_file = remi_tokenizer.decode(encoded, symbolic_format=SymbolicMusicFormat.MIDI,
                                            output_path=decoded_wav_path, export_format=ExportFormat.SYMBOLIC)
        assert isinstance(output_file, Path)
        assert output_file.exists()
        assert output_file == decoded_wav_path
        assert output_file.suffix == ".mid"

    @staticmethod
    def test_train_tokenizer(remi_tokenizer: Tokenizer, midi_data_path: Path, tmp_path: Path):
        remi_tokenizer.tokenizer_base_path = tmp_path / "tokenizers"
        assert not remi_tokenizer.cache_tokenizer_path.exists()
        remi_tokenizer.train(midi_data_path)
        assert remi_tokenizer.is_trained is True
        assert remi_tokenizer.cache_tokenizer_path.exists()

    @staticmethod
    def test_encode_decode_after_training(
            remi_tokenizer: Tokenizer,
            midi_data_path: Path,
            tmp_path: Path,
            loaded_midi_song,
            test_results_path,
            request
    ):
        assert remi_tokenizer.is_trained is False
        remi_tokenizer.tokenizer_base_path = tmp_path / "tokenizers"
        remi_tokenizer.train(midi_data_path)
        assert remi_tokenizer.is_trained is True
        loaded_score = convert_mido_to_score(loaded_midi_song)

        folder_path = test_results_path / f"{request.node.parent.name}_{request.node.name}"
        folder_path.mkdir(parents=True, exist_ok=True)
        original_wav_path = folder_path / "original.wav"
        decoded_wav_path = folder_path / "decoded.wav"
        export_to_wav(loaded_score, original_wav_path)
        encoded = remi_tokenizer.encode(loaded_midi_song, symbolic_format=SymbolicMusicFormat.MIDI)
        output_file = remi_tokenizer.decode(encoded, symbolic_format=SymbolicMusicFormat.MIDI,
                                            output_path=decoded_wav_path)
        assert isinstance(output_file, Path)
        assert output_file.exists()
        assert output_file == decoded_wav_path

