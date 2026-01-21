from src.dataset.music_augmenter.abc_music_augmenter import AbcAugmentor
from src.dataset.music_augmenter.midi_music_augmenter import MidiAugmentor
from src.constants import SymbolicMusicFormat


def get_augmenter(
    music_format: SymbolicMusicFormat, **augmentation_params
) -> AbcAugmentor | MidiAugmentor:
    if music_format == SymbolicMusicFormat.ABC:
        return AbcAugmentor(**augmentation_params)
    elif music_format == SymbolicMusicFormat.MIDI:
        return MidiAugmentor(**augmentation_params)
    else:
        raise ValueError(f"Unsupported music format for augmenter: {music_format}")
