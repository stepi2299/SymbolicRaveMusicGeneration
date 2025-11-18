import random
import re
from typing import Iterable

from mido import MidiFile as MidoMidiFile
from pathlib import Path
from src.constants import ExportFormat

from unidecode import unidecode
from src.constants import SymbolicMusicFormat
from src.utils import (
    convert_midi_to_string,
    create_midi_file_from_string,
    convert_mido_to_score,
    export_to_wav,
    convert_abc_to_score,
)


class MusicPatchilizer:
    def __init__(
        self,
        patch_size: int,
        sequence_length: int,
        add_special_patches: bool = False,
        truncate: bool = False,
        random_truncate: bool = False,
    ) -> None:
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = "(" + "|".join(map(re.escape, self.delimiters)) + ")"
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.add_special_patches = add_special_patches
        self.truncate = truncate
        self.random_truncate = random_truncate

    def encode(
        self,
        loaded_music_file: str | MidoMidiFile,
        symbolic_format: SymbolicMusicFormat,
    ) -> list[list[int]]:
        if symbolic_format == SymbolicMusicFormat.MIDI:
            loaded_music_file = convert_midi_to_string(loaded_music_file)
        return self._convert_into_patches(loaded_music_file)

    def decode(
        self,
        patches: list[list[int]],
        symbolic_format: SymbolicMusicFormat,
        output_path: Path,
        export_format: ExportFormat = ExportFormat.WAV,
    ) -> Path:
        ret_music = "".join(self._patch2bar(patch) for patch in patches)
        if symbolic_format == SymbolicMusicFormat.MIDI:
            match export_format:
                case ExportFormat.SYMBOLIC:
                    output_path = output_path.with_suffix(".mid")
                    midi_file = create_midi_file_from_string(ret_music)
                    midi_file.save(output_path)
                case ExportFormat.WAV:
                    output_path = output_path.with_suffix(".wav")
                    score = convert_mido_to_score(
                        create_midi_file_from_string(ret_music)
                    )
                    export_to_wav(score, output_path)
                case _:
                    raise ValueError(f"Export format {export_format} is not supported")
        if symbolic_format == SymbolicMusicFormat.ABC:
            match export_format:
                case ExportFormat.SYMBOLIC:
                    output_path = output_path.with_suffix(".abc")
                    with open(output_path, "w") as f:
                        f.write(ret_music)
                    return output_path
                case ExportFormat.WAV:
                    output_path = output_path.with_suffix(".wav")
                    abc_score = convert_abc_to_score(ret_music)
                    export_to_wav(abc_score, output_path)
                case _:
                    raise ValueError(f"Export format {export_format} is not supported")
        return output_path

    def _convert_into_patches(self, item: str) -> list[list[int]]:
        item = unidecode(item)
        lines = re.findall(r".*?\n|.*$", item)
        lines = list(filter(None, lines))  # remove empty lines

        patches = []

        if lines[0].split(" ")[0] == "ticks_per_beat":
            patch = ""
            for line in lines:
                if patch.startswith(line.split(" ")[0]) and (
                    len(patch) + len(" ".join(line.split(" ")[1:]))
                    <= self.patch_size - 2
                ):
                    patch = patch[:-1] + "\t" + " ".join(line.split(" ")[1:])
                else:
                    if patch:
                        patches.append(patch)
                    patch = line
            if patch != "":
                patches.append(patch)
        else:
            for line in lines:
                if len(line) > 1 and (
                    (line[0].isalpha() and line[1] == ":") or line.startswith("%%")
                ):
                    patches.append(line)
                else:
                    bars = self._split_bars(line)
                    if bars:
                        bars[-1] += "\n"
                        patches.extend(bars)

        if self.add_special_patches:
            bos_patch = chr(self.bos_token_id) * self.patch_size
            eos_patch = chr(self.eos_token_id) * self.patch_size
            patches = [bos_patch] + patches + [eos_patch]

        if len(patches) > self.sequence_length and self.truncate:
            choices = ["head", "tail", "middle"]
            choice = random.choice(choices)
            if choice == "head" or self.random_truncate is False:
                patches = patches[: self.sequence_length]
            elif choice == "tail":
                patches = patches[-self.sequence_length :]
            else:
                start = random.randint(1, len(patches) - self.sequence_length)
                patches = patches[start : start + self.sequence_length]

        patches = [self._bar2patch(patch) for patch in patches]

        return patches

    def _split_bars(self, body: Iterable[str]) -> list[str]:
        bars = re.split(self.regexPattern, "".join(body))
        bars = list(filter(None, bars))  # remove empty strings
        if bars[0] in self.delimiters:
            bars[1] = bars[0] + bars[1]
            bars = bars[1:]
        bars = [bars[i * 2] + bars[i * 2 + 1] for i in range(len(bars) // 2)]
        return bars

    def _bar2patch(self, bar: str) -> list[int]:
        patch = [self.bos_token_id] + [ord(c) for c in bar] + [self.eos_token_id]
        patch = patch[: self.patch_size]
        patch += [self.pad_token_id] * (self.patch_size - len(patch))
        return patch

    def _patch2bar(self, patch: list[int]) -> str:
        return "".join(chr(idx) if idx > self.mask_token_id else "" for idx in patch)

    def get_padded_sequence(self, music_file: list[list[int]]) -> list[list[int]]:
        padded_bar = [self.pad_token_id] * self.patch_size
        padded_music_file = [padded_bar] * (self.sequence_length - len(music_file))
        new_music_file = music_file + padded_music_file
        return new_music_file
