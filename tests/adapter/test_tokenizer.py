from src.utils import convert_mido_to_score, export_to_wav
from pathlib import Path
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
    def test_encode_without_training(remi_tokenizer: Tokenizer, loaded_midi_song):
        try:
            remi_tokenizer.encode(loaded_midi_song, symbolic_format=SymbolicMusicFormat.MIDI)
        except ValueError as e:
            assert str(e) == "Tokenizer is not trained"

    @staticmethod
    def test_encode_abc_song(remi_tokenizer: Tokenizer, loaded_abc_song):
        try:
            remi_tokenizer.encode(loaded_abc_song, symbolic_format=SymbolicMusicFormat.ABC)
        except ValueError as e:
            assert str(e) == "ABC format is not supported for tokenizer"

    @staticmethod
    def test_encode(remi_tokenizer: Tokenizer, loaded_midi_song):
        # Mock training the tokenizer
        remi_tokenizer._is_trained = True
        encoded = remi_tokenizer.encode(loaded_midi_song, symbolic_format=SymbolicMusicFormat.MIDI)
        assert isinstance(encoded, list)
        assert len(encoded) == 1  # TODO check in future
        for patch in encoded:
            assert isinstance(patch, list)
            for token in patch:
                assert isinstance(token, int)

    @staticmethod
    def test_decode_midi_wav(remi_tokenizer: Tokenizer, loaded_midi_song, test_results_path, request):
        remi_tokenizer._is_trained = True
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
        remi_tokenizer._is_trained = True

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





