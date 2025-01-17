from pathlib import Path
from typing import Tuple

import mido
from mido.messages.messages import Message
from torch.utils.data import Dataset, random_split


def midi_msg_to_str(msg: Message) -> str:
    str_msg = ""
    for value in msg.dict().values():
        str_msg += " " + str(value)

    return str_msg.strip().encode("unicode_escape").decode("utf-8")


def load_midi_song(filepath: Path) -> str:
    skip_elements = {
        "text",
        "copyright",
        "track_name",
        "instrument_name",
        "lyrics",
        "marker",
        "cue_marker",
        "device_name",
        "sequencer_specific",
    }
    try:
        # Load a MIDI file
        mid = mido.MidiFile(str(filepath))
        msg_list = ["ticks_per_beat " + str(mid.ticks_per_beat)]

        # Traverse the MIDI file
        for msg in mid.merged_track:
            if msg.type != "sysex" and not (msg.is_meta and msg.type in skip_elements):
                str_msg = midi_msg_to_str(msg)
                msg_list.append(str_msg)
    except Exception as e:
        msg = f"Could not load file: {filepath}"
        raise OSError(msg) from e

    return "\n".join(msg_list)


def load_abc_song(filepath: Path) -> str:
    with open(filepath, "r", encoding="utf-8") as file:
        data = file.read()

    return data


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
