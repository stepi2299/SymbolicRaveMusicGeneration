import multiprocessing
from functools import partial
from pathlib import Path
import torch
from omegaconf import DictConfig

from src.utils import load_midi_song, load_abc_song
from src.adapter.adapter import MusicAdapter

DATASET = 'maestro'  # maestro, midicaps
base_path = Path.cwd() / "data"


def get_extension(dataset_path: Path) -> str:
    if dataset_path.is_file():
        return dataset_path.suffix
    return ""


def filter_music_files(filepath: Path, adapter: MusicAdapter) -> None:
    try:
        extension = get_extension(filepath)
        if extension.lower() in (".mid", ".midi"):
            song = load_midi_song(filepath)
        elif extension.lower() == ".abc":
            song = load_abc_song(filepath)
        else:
            raise ValueError(f"Unknown extension: {extension}")

        song = adapter.encode(song)

        # Truncate or pad the encoded song to match the adapter's patch_length
        if len(song) > adapter.patch_length:
            song = torch.tensor(song[:adapter.patch_length])
        else:
            padded_song = song + [adapter.pad_token] * (adapter.patch_length - len(song))
            song = torch.tensor(padded_song)
    except Exception as e:
        print(f"Error processing file: {filepath}. Error: {e}, removing this file")
        filepath.unlink()


def get_adapter_config():
    return DictConfig({
        "name": "patchilizer",
        "params": {
            "patch_size": 64,
            "patch_length": 512,
            "add_special_patches": False,
            "truncate": True,
            "random_truncate": False,
        },
    })


def main():
    dataset_path = (base_path / DATASET).resolve(True)
    adapter = MusicAdapter(get_adapter_config())

    # Gather all file paths in a single list
    all_files = []
    for item in dataset_path.iterdir():
        if item.is_dir():
            for file in item.iterdir():
                all_files.append(file)
        else:
            all_files.append(item)

    # Create a partial function that injects the adapter as the second argument
    func_with_adapter = partial(filter_music_files, adapter=adapter)

    # Use multiprocessing to process files in parallel
    with multiprocessing.Pool() as pool:
        pool.map(func_with_adapter, all_files)


if __name__ == "__main__":
    main()
