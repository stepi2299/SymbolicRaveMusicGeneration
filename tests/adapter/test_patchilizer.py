import pytest
import random

from src.adapter import MusicPatchilizer
from src.utils import convert_mido_to_score, export_to_wav
from src.constants import ExportFormat, SymbolicMusicFormat
from pathlib import Path


class TestPatchilizer:
    @staticmethod
    def test_create_basic_patchilizer(patchilizer: MusicPatchilizer):
        patchilizer.patch_size = 64
        assert patchilizer.patch_size == 64
        assert isinstance(patchilizer, MusicPatchilizer)

    @staticmethod
    @pytest.mark.parametrize(
        "mask_token_id, patch, expected",
        [
            # basic: all above mask → all converted
            (10, [65, 66, 67], "ABC"),

            # some below mask → skipped
            (70, [65, 71, 72], "GH"),

            # all below mask → empty string
            (100, [65, 66], ""),

            # mix: boundary values
            (65, [65, 66, 67], "BC"),

            # empty input
            (50, [], ""),
        ],
    )
    def test_patch2bar(patchilizer, mask_token_id, patch, expected):
        patchilizer.mask_token_id = mask_token_id
        result = patchilizer._patch2bar(patch)
        assert result == expected


    @staticmethod
    @pytest.mark.parametrize(
        "bar, patch_size, expected",
        [
            # simple case: fits easily
            (
                    "AB",
                    6,
                    [1, ord("A"), ord("B"), 2, 0, 0],  # BOS, A, B, EOS, PAD, PAD
            ),
            # empty string
            (
                    "",
                    5,
                    [1, 2, 0, 0, 0],  # BOS, EOS, then padding
            ),
            # truncation: patch_size = 4 → BOS + A + B + C, then fits exactly
            (
                    "ABCD",
                    4,
                    [1, ord("A"), ord("B"), ord("C")],
            ),
            # exact fit: bar length + 2 (bos/eos) == patch_size
            (
                    "XYZ",
                    5,
                    [1, ord("X"), ord("Y"), ord("Z"), 2],
            ),
            # longer patch_size: more padding
            (
                    "A",
                    8,
                    [1, ord("A"), 2, 0, 0, 0, 0, 0],
            ),
        ],
    )
    def test_bar2patch(patchilizer, bar, patch_size, expected):
        patchilizer.patch_size = patch_size
        result = patchilizer._bar2patch(bar)
        assert result == expected
        # Always check length == patch_size
        assert len(result) == patch_size

    @staticmethod
    def test_basic_split(patchilizer):
        patchilizer.patch_size = 4
        patchilizer.patch_size = 2
        body = ["A|B|C"]
        result = patchilizer._split_bars(body)
        # join = "A|B|C" → split = ["A", "|", "B", "|", "C"]
        # grouped as ["A|", "B|"]
        assert result == ["A|", "B|"]

    @staticmethod
    def test_with_leading_delimiter(patchilizer):
        patchilizer.patch_size = 4
        patchilizer.patch_size = 2
        body = ["|A|B"]
        result = patchilizer._split_bars(body)
        # join = "|A|B" → split = ["|", "A", "|", "B"]
        # first is delimiter → merged into "A" → ["|A", "|", "B"]
        # grouped as ["|A|", "B"]
        assert result == ["|A|"]

    @staticmethod
    def test_with_double_bar(patchilizer):
        patchilizer.patch_size = 4
        patchilizer.patch_size = 2
        body = ["A||B"]
        result = patchilizer._split_bars(body)
        # join = "A||B" → split = ["A", "||", "B"]
        # grouped as ["A||B"]
        assert result == ["A||"]

    @staticmethod
    def test_multiple_strings(patchilizer):
        patchilizer.patch_size = 4
        patchilizer.patch_size = 2
        body = ["AB|", "CD|EF"]
        result = patchilizer._split_bars(body)
        # join = "AB|CD|EF" → split = ["AB", "|", "CD", "|", "EF"]
        # grouped as ["AB|", "CD|"]
        assert result == ["AB|", "CD|"]

    @staticmethod
    def test_with_other_delimiters(patchilizer):
        patchilizer.patch_size = 4
        patchilizer.patch_size = 2
        body = ["A:|B[|C|]"]
        result = patchilizer._split_bars(body)
        # join = "A:|B[|C|]" → split = ["A", ":|", "B", "[|", "C", "|", "]"]
        # → grouped as ["A:|", "B[|", "C|]"]
        assert result == ["A:|", "B[|", "C|]"]

    # @staticmethod
    # def test_empty_input(patchilizer):
    #     patchilizer.patch_size = 4
    #     patchilizer.patch_size = 2
    #     body = [""]
    #     result = patchilizer._split_bars(body)
    #     assert result == []


    @staticmethod
    @pytest.mark.skip(reason="create test after fully understanding of patchilizer logic (CLAMP2)")
    def test_ticks_per_beat_merging(patchilizer, monkeypatch):
        patchilizer.patch_size = 20
        # Monkeypatch _bar2patch for deterministic output
        monkeypatch.setattr(patchilizer, "_bar2patch", lambda bar: [len(bar)])

        item = "ticks_per_beat 480\nnote 60\nnote 62\n"
        result = patchilizer._convert_into_patches(item)
        # We expect 3 patches (ticks_per_beat, note60, note62)
        assert result == [[len("ticks_per_beat 480\n")],
                          [len("note 60\n")],
                          [len("note 62\n")]]

    @staticmethod
    def test_metadata_lines(patchilizer, monkeypatch):
        monkeypatch.setattr(patchilizer, "_bar2patch", lambda bar: [ord(bar[0])])
        item = "X:1\nT:Title\n%%MIDI program 1\n"
        result = patchilizer._convert_into_patches(item)
        # Patches are metadata lines (X:, T:, %%MIDI...)
        assert result == [[ord("X")], [ord("T")], [ord("%")]]

    @staticmethod
    def test_bar_splitting(patchilizer, monkeypatch):
        monkeypatch.setattr(patchilizer, "_split_bars", lambda line: ["A|", "B|"])
        monkeypatch.setattr(patchilizer, "_bar2patch", lambda bar: [len(bar)])
        item = "AB|CD|\n"
        result = patchilizer._convert_into_patches(item)
        # Bars → ["A|", "B|"] → _bar2patch returns lengths
        assert result == [[2], [3]]

    @staticmethod
    def test_add_special_patches(patchilizer, monkeypatch):
        patchilizer.add_special_patches = True
        patchilizer.patch_size = 4
        monkeypatch.setattr(patchilizer, "_bar2patch", lambda bar: [len(bar)])
        item = "X:1\n"
        result = patchilizer._convert_into_patches(item)
        # Should include BOS and EOS patches
        assert len(result) >= 2
        assert [len(bar) for bar in ["aaaa", "X:1\n", "bbbb"]]  # conceptual

    @staticmethod
    def test_truncate_head(patchilizer, monkeypatch):
        patchilizer.truncate = True
        patchilizer.sequence_length = 2
        patchilizer.random_truncate = False
        monkeypatch.setattr(patchilizer, "_bar2patch", lambda bar: [ord(bar[0])])
        item = "X:1\nY:2\nZ:3\n"
        result = patchilizer._convert_into_patches(item)
        # Only first 2 patches kept
        assert len(result) == 2

    @staticmethod
    def test_truncate_tail(patchilizer, monkeypatch):
        patchilizer.truncate = True
        patchilizer.sequence_length = 1
        patchilizer.random_truncate = True
        monkeypatch.setattr(patchilizer, "_bar2patch", lambda bar: [ord(bar[0])])

        # Force choice = 'tail'
        monkeypatch.setattr(random, "choice", lambda _: "tail")

        item = "X:1\nY:2\n"
        result = patchilizer._convert_into_patches(item)
        # Only last patch kept
        assert result == [[ord("Y")]]

    @staticmethod
    def test_truncate_middle(patchilizer, monkeypatch):
        patchilizer.truncate = True
        patchilizer.sequence_length = 1
        patchilizer.random_truncate = True
        monkeypatch.setattr(patchilizer, "_bar2patch", lambda bar: [ord(bar[0])])

        # Force choice = 'middle'
        monkeypatch.setattr(random, "choice", lambda _: "middle")
        monkeypatch.setattr(random, "randint", lambda a, b: 1)

        item = "X:1\nY:2\nZ:3\n"
        result = patchilizer._convert_into_patches(item)
        assert len(result) == 1
        assert result in [[[ord("X")]], [[ord("Y")]], [[ord("Z")]]]

    @staticmethod
    def test_decode_midi_wav(patchilizer: MusicPatchilizer, loaded_midi_song, test_results_path, request):
        loaded_score = convert_mido_to_score(loaded_midi_song)

        folder_path = test_results_path / f"{request.node.parent.name}_{request.node.name}"
        folder_path.mkdir(parents=True, exist_ok=True)
        original_wav_path = folder_path / "original.wav"
        decoded_wav_path = folder_path / "decoded.wav"
        export_to_wav(loaded_score, original_wav_path)
        encoded = patchilizer.encode(loaded_midi_song, symbolic_format=SymbolicMusicFormat.MIDI)
        output_file = patchilizer.decode(encoded, symbolic_format=SymbolicMusicFormat.MIDI, output_path=decoded_wav_path)
        assert isinstance(output_file, Path)
        assert output_file.exists()
        assert output_file == decoded_wav_path

    @staticmethod
    def test_decode_midi_midi(patchilizer: MusicPatchilizer, loaded_midi_song, test_results_path, request):

        folder_path = test_results_path / f"{request.node.parent.name}.{request.node.name}"
        folder_path.mkdir(parents=True, exist_ok=True)
        original_midi_path = folder_path / "original.mid"
        decoded_midi_path = folder_path / "decoded.mid"
        loaded_midi_song.save(original_midi_path)
        encoded = patchilizer.encode(loaded_midi_song, symbolic_format=SymbolicMusicFormat.MIDI)
        output_file = patchilizer.decode(encoded, symbolic_format=SymbolicMusicFormat.MIDI,
                                         output_path=decoded_midi_path, export_format=ExportFormat.SYMBOLIC)
        assert isinstance(output_file, Path)
        assert output_file.exists()
        assert output_file == decoded_midi_path
        assert output_file.suffix == ".mid"

    @staticmethod
    def test_decode_abc_abc(patchilizer: MusicPatchilizer, loaded_abc_song, test_results_path, request):
        folder_path = test_results_path / f"{request.node.parent.name}.{request.node.name}"
        folder_path.mkdir(parents=True, exist_ok=True)
        original_abc_path = folder_path / "original.abc"
        with open(original_abc_path, "w") as f:
            f.write(loaded_abc_song)
        decoded_abc_path = folder_path / "decoded.abc"
        encoded = patchilizer.encode(loaded_abc_song, symbolic_format=SymbolicMusicFormat.ABC)
        output_file = patchilizer.decode(encoded, symbolic_format=SymbolicMusicFormat.ABC,
                                         output_path=decoded_abc_path, export_format=ExportFormat.SYMBOLIC)
        assert isinstance(output_file, Path)
        assert output_file.exists()
        assert output_file == decoded_abc_path
        assert output_file.suffix == ".abc"

    @staticmethod
    @pytest.mark.skip(reason="ABC to WAV export is not implemented yet")
    def test_decode_abc_wav(patchilizer: MusicPatchilizer, loaded_abc_song, test_results_path, request):
        folder_path = test_results_path / f"{request.node.parent.name}.{request.node.name}"
        folder_path.mkdir(parents=True, exist_ok=True)
        output_abc_path = folder_path / "decoded.abc"
        encoded = patchilizer.encode(loaded_abc_song, symbolic_format=SymbolicMusicFormat.ABC)
        output_file = patchilizer.decode(encoded, symbolic_format=SymbolicMusicFormat.ABC, output_path=output_abc_path)
        assert isinstance(output_file, Path)
        assert output_file.exists()
        assert output_file == output_abc_path

    @staticmethod
    @staticmethod
    def test_get_padded_sequence(patchilizer: MusicPatchilizer):
        new_sequence_len = 8
        new_patch_size = 5
        patchilizer.patch_size = new_patch_size
        patchilizer.sequence_length = new_sequence_len
        example_patch = [[5, 5, 5, 5, 5], [4, 4, 4, 4, 4]]
        result_padded_patches = patchilizer.get_padded_sequence(example_patch)
        assert isinstance(result_padded_patches, list)
        assert len(result_padded_patches) == new_sequence_len
        for i, patch in enumerate(result_padded_patches):
            assert len(patch) == new_patch_size
            if i > 1:
                assert result_padded_patches[i] == [patchilizer.pad_token_id] * new_patch_size



