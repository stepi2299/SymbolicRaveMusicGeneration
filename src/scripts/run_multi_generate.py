#!/usr/bin/env python3
"""
Run generate_songs for every checkpoint found under outputs/checkpoints/train_repr/,
overriding shared generation parameters via Hydra's command-line override syntax.

Usage examples:
  python src/scripts/run_multi_generate.py
  python src/scripts/run_multi_generate.py --songs 50 --format midi --temperature 0.9
  python src/scripts/run_multi_generate.py --checkpoints-dir outputs/checkpoints/train_repr
"""

import argparse
import subprocess
import sys
from pathlib import Path

CHECKPOINTS_DIR = Path("outputs/checkpoints/train_repr")


def find_checkpoints(base_dir: Path) -> list[Path]:
    """Recursively find all .ckpt files under base_dir."""
    return sorted(base_dir.rglob("*.ckpt"))


def run_generation(
    checkpoint_path: Path,
    songs_to_generate: int,
    output_format: str,
    temperature: float,
) -> int:
    """
    Invoke generate_songs.py for a single checkpoint via Hydra overrides.
    Returns the process exit code.
    """
    cmd = [
        sys.executable,
        "-m",
        "src.scripts.generate_songs",
        f"checkpoint_path='{checkpoint_path.as_posix()}'",
        f"songs_to_generate={songs_to_generate}",
        f"output_format={output_format}",
        f"generation_params.temperature={temperature}",
    ]
    print(f"\n{'='*60}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Songs      : {songs_to_generate}")
    print(f"Format     : {output_format}")
    print(f"Temperature: {temperature}")
    print(f"Command    : {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generate_songs for every checkpoint in a directory.")
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=CHECKPOINTS_DIR,
        help=f"Root directory that contains checkpoint sub-folders (default: {CHECKPOINTS_DIR})",
    )
    parser.add_argument(
        "--songs",
        type=int,
        default=100,
        help="Number of songs to generate per checkpoint (default: 100)",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["midi", "wav"],
        default="wav",
        help="Export format: midi or wav (default: wav)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.checkpoints_dir)
    if not checkpoints:
        print(f"No .ckpt files found under '{args.checkpoints_dir}'. Exiting.")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoint(s) under '{args.checkpoints_dir}':")
    for ckpt in checkpoints:
        print(f"  {ckpt}")

    failed: list[tuple[Path, int]] = []
    for idx, ckpt in enumerate(checkpoints, start=1):
        print(f"\n[{idx}/{len(checkpoints)}] Processing: {ckpt.parent.name}/{ckpt.name}")
        exit_code = run_generation(
            checkpoint_path=ckpt,
            songs_to_generate=args.songs,
            output_format=args.output_format,
            temperature=args.temperature,
        )
        if exit_code != 0:
            print(f"  WARNING: generation failed with exit code {exit_code}")
            failed.append((ckpt, exit_code))

    print(f"\n{'='*60}")
    print(f"Done. {len(checkpoints) - len(failed)}/{len(checkpoints)} checkpoint(s) succeeded.")
    if failed:
        print("Failed checkpoints:")
        for ckpt, code in failed:
            print(f"  {ckpt}  (exit code {code})")
        sys.exit(1)


if __name__ == "__main__":
    main()
