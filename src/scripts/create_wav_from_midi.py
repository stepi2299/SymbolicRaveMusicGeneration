from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import logging
from tqdm import tqdm
from src.utils import export_to_wav
from symusic import Score

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MIDI_SUFFIXES = {".mid", ".midi"}


def collect_midi_files(root: Path, max_depth: int = 2) -> list[Path]:
    """Return all MIDI files inside *root* up to *max_depth* folder levels."""
    midi_files: list[Path] = []
    for depth in range(max_depth + 1):
        pattern = "/".join(["*"] * depth) + ("/" if depth else "") + "*.mid"
        midi_files.extend(root.glob(pattern))
        midi_files.extend(root.glob(pattern.replace(".mid", ".midi")))
    # deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in midi_files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return sorted(unique)


def convert_single_midi_to_wav(midi_path: Path, wav_path: Path) -> None:
    score = Score(midi_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_wav(score, wav_path)


def main() -> None:
    """Parse CLI arguments, convert MIDI files to WAV, and save results."""
    parser = ArgumentParser(
        description="Convert MIDI files to WAV, preserving the folder structure.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs/generated_midi"),
        help="Directory containing .mid/.midi files (searched up to 2 levels deep).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/generated_waves"),
        help="Root directory where .wav files are written (mirrors input structure).",
    )

    args = parser.parse_args()
    input_dir: Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not input_dir.exists():
        parser.error(f"--input_dir does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = collect_midi_files(input_dir)
    if not midi_files:
        logger.warning("No MIDI files found in %s (depth ≤ 2). Exiting.", input_dir)
        return

    logger.info("Found %d MIDI file(s) under %s", len(midi_files), input_dir)

    ok_count = 0
    fail_count = 0
    failed: list[tuple[Path, str]] = []

    for midi_path in tqdm(midi_files, desc="Converting", unit="file"):
        relative = midi_path.relative_to(input_dir)
        wav_path = output_dir / relative.with_suffix(".wav")
        try:
            convert_single_midi_to_wav(midi_path, wav_path)
            ok_count += 1
        except Exception as exc:
            logger.error("Failed to convert %s: %s", midi_path.name, exc)
            failed.append((midi_path, str(exc)))
            fail_count += 1

    logger.info("Done. Converted: %d  |  Failed: %d", ok_count, fail_count)
    if failed:
        logger.warning("Failed files:")
        for path, reason in failed:
            logger.warning("  %s — %s", path, reason)


if __name__ == "__main__":
    main()
