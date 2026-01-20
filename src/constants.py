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

CP_TOKEN_SIZE = {
    "family": 6,  # Np. 0=Note, 1=Rest, 2=EOS...
    "position": 38,  # Pozycje w takcie
    "pitch": 128,  # MIDI pitch (0-127)
    "velocity": 38,  # Skwantyzowana głośność
    "duration": 70,  # Skwantyzowany czas trwania nuty
}
