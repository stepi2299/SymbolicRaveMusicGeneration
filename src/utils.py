import logging as lg
import math
import re
from pathlib import Path

import mido
import requests
import symusic
import torch
from mido.messages.messages import Message
from symusic import Note, Score, Synthesizer, Track, dump_wav
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, random_split
from tqdm import tqdm

KB = 1024
MB = 1024 * KB

base_weights_path = Path.cwd() / ".weights"
base_dataset_path = Path.cwd() / "data"
output_songs_path = Path.cwd() / "outputs" / "generated"

logger = lg.getLogger(__name__)


def download_file(url: str, destination: Path, verbose: bool = True, chunk_size: int = 10 * MB) -> None:
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


def load_midi_song(filepath: Path) -> mido.MidiFile:
    try:
        midi_file = mido.MidiFile(filepath)
    except Exception as e:
        msg = f"Could not load midi file: {filepath}"
        raise OSError(msg) from e

    return midi_file


def convert_mido_to_score(midi_file: mido.MidiFile) -> Score:
    score = Score()

    for mido_track in midi_file.tracks:
        track = Track()
        current_time = 0
        notes_on = {}  # key: (channel, note), value: start_time

        for msg in mido_track:
            current_time += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                notes_on[(msg.channel, msg.note)] = (current_time, msg.velocity)

            elif (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in notes_on:
                    start_time, velocity = notes_on.pop(key)
                    dur = current_time - start_time

                    track.notes.append(
                        Note(
                            time=start_time,
                            duration=dur,
                            pitch=msg.note,
                            velocity=velocity,
                        )
                    )

        if len(track.notes) > 0:
            score.tracks.append(track)
    return score


def convert_midi_to_string(midi_file: mido.MidiFile) -> str:
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
        msg_list = ["ticks_per_beat " + str(midi_file.ticks_per_beat)]
        # Traverse the MIDI file
        for msg in midi_file.merged_track:
            if msg.type != "sysex" and not (msg.is_meta and msg.type in skip_elements):
                str_msg = midi_msg_to_str(msg)
                msg_list.append(str_msg)
    except Exception as e:
        msg = f"Could not convert midi file into string: {midi_file.filename}"
        raise ValueError(msg) from e

    return "\n".join(msg_list)


def load_abc_song(filepath: Path) -> str:
    with open(filepath, encoding="utf-8") as file:
        data = file.read()

    return data


def symbolic_collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    return pad_sequence(batch, batch_first=True, padding_value=0)


def split_dataset(dataset: Dataset, train_split: float = 0.8) -> tuple[Dataset, Dataset]:
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
        elif "%" in line and not line.startswith("%%score"):
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
        values = str_msg[:left_idx].split(" ") + [list_str] + str_msg[right_idx + 1 :].split(" ")
        values = [value for value in values if value != ""]
    else:
        values = str_msg.split(" ")

    if len(values) != 1:
        for idx, (key, content) in enumerate(msg.__dict__.items()):
            if key == "type":
                continue
            value = values[idx]
            if isinstance(content, int | float):
                float_value = float(value)
                value = float_value
                if value % 1 == 0:
                    value = int(value)
            setattr(msg, key, value)
    return msg


def create_midi_file_from_string(mtf_text: str) -> mido.MidiFile:
    mtf_text = clean_generative_midi_text(mtf_text)
    mtf_lines = mtf_text.strip().split("\n")
    mtf_lines = list(filter(None, mtf_lines))

    if not mtf_lines:
        logger.warning("Generated text is empty after cleaning.")
        return mido.MidiFile()

    ticks_per_beat_line = mtf_lines[0]
    if not ticks_per_beat_line.startswith("ticks_per_beat "):
        logger.warning("Lack of important information about ticks_per_beat when creating midifile from string")
        mid = mido.MidiFile()
    else:
        ticks_per_beat = int(ticks_per_beat_line.split()[1])
        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for msg in mtf_lines[1:]:
        msg = msg.strip()
        if "unknown_meta" in msg:
            continue
        if "\t" in msg:
            parts = msg.split("\t")
            msg_type = parts[0].split(" ")[0]

            try:
                track.append(str_to_msg(parts[0]))
            except Exception as e:
                logger.error("Error during parsing the first part of compressed message: " + parts[0] + " -> " + str(e))
                continue

            for part in parts[1:]:
                part = part.strip()
                reconstructed_msg = f"{msg_type} {part}"
                try:
                    track.append(str_to_msg(reconstructed_msg))
                except Exception as e:
                    logger.debug("Error during parsing compressed message: " + reconstructed_msg + " -> " + str(e))
                    continue
        else:
            try:
                track.append(str_to_msg(msg))
            except Exception as e:
                logger.debug("Error during parsing compressed message: " + msg + " -> " + str(e))
                continue

    return mid


def save_midi_song_from_string(mtf_text: str, output_path: Path) -> None:
    midi_file = create_midi_file_from_string(mtf_text)
    midi_file.save(str(output_path))


def filter_state_dict_keys(state_dict: dict[str, torch.Tensor], old_name: str):
    new_state_dict = {}
    for key, value in state_dict.items():
        if old_name in key:
            new_key = key.replace(old_name, "")[1:]
            new_state_dict[new_key] = value
    return new_state_dict


def rename_state_dict_keys(
    state_dict: dict[str, torch.Tensor], substring_to_rename: str, new_name: str
) -> dict[str, torch.Tensor]:
    new_state_dict = {}
    for keys in state_dict:
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


def remove_state_dict_keys(state_dict: dict[str, torch.Tensor], keys_to_remove: list) -> dict[str, torch.Tensor]:
    new_state_dict = {}
    for keys_string in state_dict:
        list_of_keys = keys_string.split(".")
        for key in list_of_keys:
            if key in keys_to_remove:
                break
            new_state_dict[keys_string] = state_dict[keys_string]
    return new_state_dict


def convert_abc_to_score(abc_string: str) -> symusic.Score:
    score = Score()
    score.from_abc(abc_string)
    return score


synth = Synthesizer(sample_rate=44100)


def export_to_wav(score: symusic.Score, filepath: Path, sample_rate: int = 44100) -> None:
    audio = synth.render(score, stereo=True)
    dump_wav(filepath.as_posix(), audio, sample_rate=sample_rate, use_int16=True)


def get_sinusoidal_position_embedding(seq_len: int, d_model: int) -> torch.Tensor:
    """Generuje sinusoidalne osadzenia pozycyjne zgodnie z oryginalnym artykułem o Transformerze."""

    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe.unsqueeze(0)  # [1, L, D]


def get_song_dim(tokens: list[int] | list[list[int]]) -> int:
    if not isinstance(tokens, list):
        raise TypeError("Its not a list")
    if tokens and all(isinstance(el, list) for el in tokens):
        return 2
    return 1


def clean_generative_midi_text(text: str) -> str:
    keywords = r"(note_on|note_off|control_change|program_change|pitchwheel|set_tempo|time_signature)"
    text = re.sub(r"(?<!^)(?<!\n)(" + keywords + r")", r"\n\1", text)

    text = re.sub(r"\b[a-df-zA-Z]+\b", " ", text)
    text = re.sub(r"(?<=\s)[a-zA-Z]+(?=\s)", " ", text)

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n+", "\n", text)

    return text.strip()
