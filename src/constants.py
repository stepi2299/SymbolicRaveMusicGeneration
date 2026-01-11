from enum import Enum


class SymbolicMusicFormat(Enum):
    MIDI = "midi"
    ABC = "abc"


class MusicAdapterType(Enum):
    PATCHILIZER = "patchilizer"
    TOKENIZER = "tokenizer"


class ExportFormat(Enum):
    SYMBOLIC = "midi"
    WAV = "wav"


CP_WORD_INDEXES = {
    "family": 0,
    "position": 1,
    "pitch": 2,
    "velocity": 3,
    "duration": 4,
}
