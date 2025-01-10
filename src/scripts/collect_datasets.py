from pathlib import Path
import shutil
import requests
import tarfile

import muspy


def download_from_url(url, save_path, verbose=False, **kwargs) -> Path:
    archive_path = save_path / 'midicaps.tar.gz'
    print(f'Downloading to {archive_path} ...')
    with requests.get(url, stream=True) as r:
        with open(archive_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    print(f'Extracting {archive_path} to {save_path} ...')
    save_path.mkdir(exist_ok=True)
    return archive_path


def clean_root_dir(root_dir: Path):
    print(f"Cleaning {root_dir}")
    shutil.rmtree(root_dir, ignore_errors=True)
    root_dir.mkdir(exist_ok=True)
    print(f"Successfully cleaned {root_dir}")

def cleanup(root_dir: Path):
    print(f"Cleaning {root_dir}")
    for item in root_dir.iterdir():
        if not item.is_dir():
            try:
                item.unlink()
            except Exception as e:
                print(f"Could not delete {item}. Error: {e}")
    print(f"Successfully cleaned {root_dir}")

def download_and_process_maestro(save_path: Path) -> None:
    muspy.MAESTRODatasetV3(save_path, download_and_extract=True)
    maestro_path = save_path / "maestro-v3.0.0"
    new_maestro_path = save_path / "maestro"
    maestro_path.rename(new_maestro_path)
    cleanup(new_maestro_path)

def download_and_process_midicaps(save_path: Path) -> None:
    midicaps_url = 'https://huggingface.co/datasets/amaai-lab/MidiCaps/resolve/main/midicaps.tar.gz?download=true'
    midicaps_path = save_path / "midicaps"
    midicaps_path.mkdir(exist_ok=True)
    midicaps_tar_path = download_from_url(
        midicaps_url,
        save_path,
    )
    with tarfile.open(midicaps_tar_path, mode='r:gz') as tar:
        tar.extractall(path=midicaps_path)
    print(f"Extracted {midicaps_tar_path} to {midicaps_path}")
    extra_level_folder = midicaps_path / 'lmd_full'
    for item in extra_level_folder.iterdir():
        item.rename(midicaps_path / item.name)
    extra_level_folder.rmdir()
    print(f"Moved {extra_level_folder} to {midicaps_path}")


def download_data():
    base_data_path = Path.cwd() / "data"
    clean_root_dir(base_data_path)
    download_and_process_maestro(base_data_path)
    download_and_process_midicaps(base_data_path)
    cleanup(base_data_path)


if __name__ == "__main__":
    download_data()