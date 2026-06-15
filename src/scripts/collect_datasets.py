import json
import logging as lg
import shutil
import tarfile
import zipfile
from pathlib import Path

import muspy
import requests

lg.basicConfig(level=lg.INFO)
logger = lg.getLogger(__name__)


GENRE_1 = ["pop", "rock"]
GENRE_2 = ["electronic", "ambient"]
GENRE_3 = ["classical", "soundtrack"]
GENRE_4 = ["jazz", "pop"]
GENRE_5 = ["rock", "electronic"]


MIDI_PATH_KEY = "location"
GENRE_KEY = "genre"
GENRE_TO_SAVE = [GENRE_1, GENRE_2, GENRE_3, GENRE_4, GENRE_5]


def download_archive_from_url(url: str, save_path: Path, archive_path: Path, verbose=False, **kwargs) -> Path:
    logger.info(f"Downloading to {archive_path} ...")
    with requests.get(url, stream=True) as r, open(archive_path, "wb") as f:
        shutil.copyfileobj(r.raw, f)

    logger.info(f"Extracting {archive_path} to {save_path} ...")
    save_path.mkdir(exist_ok=True)
    return archive_path


def download_file_from_url(url: str, verbose=False, **kwargs) -> list:
    try:
        logger.info(f"Downloading {url}")
        response = requests.get(url)
        response.raise_for_status()

        dane_json = []
        for line in response.iter_lines():
            if line:
                dane_json.append(json.loads(line.decode("utf-8")))

        logger.info(f"Success! Downloaded {len(dane_json)} records from {url}")
        return dane_json

    except requests.exceptions.RequestException as e:
        logger.error(f"Error during downloading file from {url}: {e}")
        raise e
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error occurred {e}")
        raise e


def clean_root_dir(root_dir: Path):
    logger.info(f"Cleaning {root_dir}")
    shutil.rmtree(root_dir, ignore_errors=True)
    root_dir.mkdir(exist_ok=True)
    logger.info(f"Successfully cleaned {root_dir}")


def cleanup(root_dir: Path):
    logger.info(f"Cleaning {root_dir}")
    for item in root_dir.iterdir():
        if not item.is_dir():
            try:
                item.unlink()
            except Exception as e:
                print(f"Could not delete {item}. Error: {e}")
    logger.info(f"Successfully cleaned {root_dir}")


def download_and_process_maestro(save_path: Path) -> None:
    muspy.MAESTRODatasetV3(save_path, download_and_extract=True)
    maestro_path = save_path / "maestro-v3.0.0"
    new_maestro_path = save_path / "maestro"
    maestro_path.rename(new_maestro_path)
    cleanup(new_maestro_path)


def download_midicaps(save_path: Path) -> Path:
    midicaps_url = "https://huggingface.co/datasets/amaai-lab/MidiCaps/resolve/main/midicaps.tar.gz?download=true"
    midicaps_path = save_path / "midicaps"
    midicaps_path.mkdir(exist_ok=True)
    midicaps_tar_path = download_archive_from_url(
        midicaps_url,
        save_path,
        save_path / "midicaps.tar.gz",
    )
    with tarfile.open(midicaps_tar_path, mode="r:gz") as tar:
        tar.extractall(path=midicaps_path)
    logger.info(f"Extracted {midicaps_tar_path} to {midicaps_path}")
    return midicaps_path


def process_raw_midicaps(midicaps_path: Path) -> None:
    extra_level_folder = midicaps_path / "lmd_full"
    for item in extra_level_folder.iterdir():
        item.rename(midicaps_path / item.name)
    extra_level_folder.rmdir()
    logger.info(f"Moved {extra_level_folder} to {midicaps_path}")


def download_and_process_pop909(save_path: Path) -> None:
    pop909_url = "https://github.com/music-x-lab/POP909-Dataset/archive/refs/heads/master.zip"
    pop909_path = save_path / "pop909"
    pop909_path.mkdir(exist_ok=True)
    pop909_zip_path = download_archive_from_url(
        pop909_url,
        save_path,
        save_path / "pop909.zip",
    )
    with zipfile.ZipFile(pop909_zip_path, "r") as zip_ref:
        zip_ref.extractall(pop909_path)
    logger.info(f"Extracted {pop909_zip_path} to {pop909_path}")
    extra_level_folder = pop909_path / "POP909-Dataset-master" / "POP909"
    for item in extra_level_folder.iterdir():
        item.rename(pop909_path / item.name)
    shutil.rmtree(extra_level_folder.parent)
    default_folder = pop909_path / "pop_data"
    default_folder.mkdir(exist_ok=True)
    for item in pop909_path.iterdir():
        if not item.is_dir():
            item.unlink()
            continue
        for subitem in item.iterdir():
            if subitem.is_dir():
                shutil.rmtree(subitem)
            elif subitem.suffix not in [".mid", ".midi"]:
                subitem.unlink()
            else:
                subitem.rename(default_folder / subitem.name)
        if item.is_dir() and not any(item.iterdir()):
            item.rmdir()

    logger.info(f"Moved {extra_level_folder} to {pop909_path}")


def prepare_midicaps_classes_dirs(base_dir: Path) -> Path:
    base_dir.resolve(True)
    midicaps_classes_dir = base_dir / "midicaps_classes"
    midicaps_classes_dir.mkdir(exist_ok=True)
    for genre in GENRE_TO_SAVE:
        genre_folder_name = "_".join(genre)
        genre_folder_path = midicaps_classes_dir / genre_folder_name
        genre_folder_path.mkdir(exist_ok=True)
    return midicaps_classes_dir


def divide_midicaps_into_classes(base_path: Path, main_midicaps_file_path: Path) -> None:
    midicaps_json_url = "https://huggingface.co/datasets/amaai-lab/MidiCaps/resolve/main/train.json?download=true"
    divided_midicaps_path = prepare_midicaps_classes_dirs(base_path)
    json_file = download_file_from_url(midicaps_json_url)
    for data_dict in json_file:
        data_path = main_midicaps_file_path / data_dict[MIDI_PATH_KEY]
        try:
            data_path.resolve(True)
            genre = data_dict[GENRE_KEY]
            if genre in GENRE_TO_SAVE:
                genre_folder_name = "_".join(genre)
                genre_folder_path = divided_midicaps_path / genre_folder_name
                shutil.copy(data_path, genre_folder_path / data_path.name)
        except FileNotFoundError:
            logger.warning(f"File {data_path} not found. Skipping.")
    logger.info(f"Divided MidiCaps into classes and saved in {divided_midicaps_path}")


def download_data():
    base_data_path = Path.cwd() / "data"
    clean_root_dir(base_data_path)
    download_and_process_maestro(base_data_path)
    download_and_process_pop909(base_data_path)

    midicaps_file_path = download_midicaps(base_data_path)
    divide_midicaps_into_classes(base_data_path, midicaps_file_path)
    process_raw_midicaps(midicaps_file_path)
    cleanup(base_data_path)


if __name__ == "__main__":
    download_data()
