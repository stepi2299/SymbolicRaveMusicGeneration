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
