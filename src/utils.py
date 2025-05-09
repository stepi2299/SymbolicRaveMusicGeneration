from pathlib import Path
from typing import Tuple, Dict
import requests
import torch
from tqdm import tqdm
import logging as lg

import mido
from mido.messages.messages import Message
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence

KB = 1024
MB = 1024 * KB

base_weights_path = Path.cwd() / ".weights"
base_dataset_path = Path.cwd() / "data"
output_songs_path = Path.cwd() / "outputs" / "generated"

logger = lg.getLogger(__name__)


def download_file(
    url: str, destination: Path, verbose: bool = True, chunk_size: int = 10 * MB
) -> None:
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            if verbose:
                progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if verbose:
                        progress_bar.update(len(chunk))
                    file.write(chunk)

            if verbose:
                progress_bar.close()

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading the file from url: {url}. Error: {e}")


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


def symbolic_collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    return pad_sequence(batch, batch_first=True, padding_value=0)


def split_dataset(
    dataset: Dataset, train_split: float = 0.8
) -> Tuple[Dataset, Dataset]:
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def filter_abc(lines: list[str]) -> str:
    """
    Filter out the metadata from the abc file

    Args:
        lines (list): List of lines in the abc file

    Returns:
        music (str): Music string
    """
    music = ""
    for line in lines:
        if (
            line[:2]
            in [
                "A:",
                "B:",
                "C:",
                "D:",
                "F:",
                "G",
                "H:",
                "N:",
                "O:",
                "R:",
                "r:",
                "S:",
                "T:",
                "W:",
                "w:",
                "X:",
                "Z:",
            ]
            or line == "\n"
            or (line.startswith("%") and not line.startswith("%%score"))
        ):
            continue
        else:
            if "%" in line and not line.startswith("%%score"):
                line = "%".join(line.split("%")[:-1])
                music += line[:-1] + "\n"
            else:
                music += line + "\n"
    return music


def get_possible_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str_to_msg(str_msg: str) -> mido.Message:
    type = str_msg.split(" ")[0]
    try:
        msg = mido.Message(type)
    except:
        msg = mido.MetaMessage(type)

    if type in [
        "text",
        "copyright",
        "track_name",
        "instrument_name",
        "lyrics",
        "marker",
        "cue_marker",
        "device_name",
    ]:
        values = [
            type,
            " ".join(str_msg.split(" ")[1:-1]).encode("utf-8").decode("unicode_escape"),
            str_msg.split(" ")[-1],
        ]
    elif "[" in str_msg or "(" in str_msg:
        is_bracket = "[" in str_msg
        left_idx = str_msg.index("[") if is_bracket else str_msg.index("(")
        right_idx = str_msg.index("]") if is_bracket else str_msg.index(")")
        list_str = [int(num) for num in str_msg[left_idx + 1 : right_idx].split(", ")]
        if not is_bracket:
            list_str = tuple(list_str)
        values = (
            str_msg[:left_idx].split(" ")
            + [list_str]
            + str_msg[right_idx + 1 :].split(" ")
        )
        values = [value for value in values if value != ""]
    else:
        values = str_msg.split(" ")

    if len(values) != 1:
        for idx, (key, content) in enumerate(msg.__dict__.items()):
            if key == "type":
                continue
            value = values[idx]
            if isinstance(content, int) or isinstance(content, float):
                float_value = float(value)
                value = float_value
                if value % 1 == 0:
                    value = int(value)
            setattr(msg, key, value)
    return msg


def save_midi_song_from_string(mtf_text: str, output_path: Path) -> None:
    mtf_lines = mtf_text.strip().split("\n")

    ticks_per_beat_line = mtf_lines[0]
    if not ticks_per_beat_line.startswith("ticks_per_beat "):
        logger.warning(
            "Lack of important information about ticks_per_beat when creating midifile from string"
        )
        mid = mido.MidiFile()
    else:
        ticks_per_beat = int(ticks_per_beat_line.split()[1])
        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for msg in mtf_lines[1:]:
        if "unknown_meta" in msg:
            continue
        new_msg = str_to_msg(msg)
        track.append(new_msg)

    mid.save(str(output_path))


def filter_state_dict_keys(state_dict: Dict[str, torch.Tensor], old_name: str):
    new_state_dict = {}
    for key, value in state_dict.items():
        if old_name in key:
            new_key = key.replace(old_name, "")[1:]
            new_state_dict[new_key] = value
    return new_state_dict


def rename_state_dict_keys(
    state_dict: Dict[str, torch.Tensor], substring_to_rename: str, new_name: str
) -> Dict[str, torch.Tensor]:
    new_state_dict = {}
    for keys in state_dict.keys():
        separated_keys = keys.split(".")
        if substring_to_rename in separated_keys:
            for idx, key in enumerate(separated_keys):
                if key == substring_to_rename:
                    separated_keys[idx] = new_name
            new_key = ".".join(separated_keys)
            new_state_dict[new_key] = state_dict[keys]
        else:
            new_state_dict[keys] = state_dict[keys]
    return new_state_dict


def remove_state_dict_keys(
    state_dict: Dict[str, torch.Tensor], keys_to_remove: list
) -> Dict[str, torch.Tensor]:
    new_state_dict = {}
    for keys_string in state_dict.keys():
        list_of_keys = keys_string.split(".")
        for key in list_of_keys:
            if key in keys_to_remove:
                break
            new_state_dict[keys_string] = state_dict[keys_string]
    return new_state_dict
