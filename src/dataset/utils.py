from pathlib import Path
from typing import Tuple

import symusic
from mido.messages.messages import Message
from symusic import Score
from torch.utils.data import Dataset, random_split


def _midi_msg_to_str(msg: Message) -> str:
    str_msg = ""
    for value in msg.dict().values():
        str_msg += " " + str(value)

    return str_msg.strip().encode("unicode_escape").decode("utf-8")


def load_midi_song(filepath: Path) -> symusic.Score:
    """
    Loads a MIDI file and returns a symusic.Score object containing
    relevant events (excluding certain meta or sysex messages).
    """
    # Consider adding processing to filter out certain messages
    try:
        score = Score(filepath, ttype="tick")
    except Exception as e:
        msg = f"Could not load file: {filepath}"
        raise OSError(msg) from e
    return score


def load_abc_song(filepath: Path) -> Score:
    # Consider adding processing to filter out certain messages
    try:
        score = Score(filepath, ttype="tick")
    except Exception as e:
        msg = f"Could not load file: {filepath}"
        raise OSError(msg) from e
    return score


def symbolic_collate_fn(batch: list[str]) -> list[str]:
    # TODO Think if we need to do something here: for example padding to make strings of the same length
    return batch


def split_dataset(
    dataset: Dataset, train_split: float = 0.8
) -> Tuple[Dataset, Dataset]:
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
