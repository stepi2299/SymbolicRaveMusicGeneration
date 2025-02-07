from pathlib import Path
from typing import Tuple
import requests
import torch
from tqdm import tqdm

import mido
from mido.messages.messages import Message
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence

KB = 1024
MB = 1024 * KB

base_weights_path = Path.cwd() / ".weights"
base_dataset_path = Path.cwd() / "data"


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
        print(f"Error downloading the file from url: {url}. Error: {e}")


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
        if line[:2] in ["A:", "B:", "C:", "D:", "F:", "G", "H:", "N:", "O:", "R:", "r:", "S:", "T:", "W:", "w:",
                        "X:", "Z:"] \
                or line == "\n" \
                or (line.startswith("%") and not line.startswith("%%score")):
            continue
        else:
            if "%" in line and not line.startswith("%%score"):
                line = "%".join(line.split("%")[:-1])
                music += line[:-1] + "\n"
            else:
                music += line + '\n'
    return music


def get_possible_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
