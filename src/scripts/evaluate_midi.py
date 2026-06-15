"""
Evaluate a directory of generated MIDI files using the muspy library.

Two calling modes are auto-detected from the contents of --input_dir:

  Mode A — Single run folder (input_dir contains .mid files directly):
    python src/scripts/evaluate_midi.py \\
        --input_dir outputs/generated_midi/cp_gpt_maestro2026-05-29_00-08-58 \\
        --output_dir outputs/evaluation_results/cp_gpt_maestro \\
        --scale C_major

  Mode B — Parent folder of run folders (input_dir contains sub-dirs with .mid files):
    python src/scripts/evaluate_midi.py \\
        --input_dir outputs/generated_midi \\
        --output_dir outputs/evaluation_results/all_runs \\
        --scale C_major

In Mode B, each sub-directory is evaluated independently and results are mirrored
under --output_dir/<sub_dir_name>/. A combined report is also written to --output_dir/.
"""

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd
import muspy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scale helpers
# ---------------------------------------------------------------------------

SCALE_MAP: dict[str, tuple[int, str]] = {
    "C_major": (0, "major"),
    "C#_major": (1, "major"),
    "Db_major": (1, "major"),
    "D_major": (2, "major"),
    "D#_major": (3, "major"),
    "Eb_major": (3, "major"),
    "E_major": (4, "major"),
    "F_major": (5, "major"),
    "F#_major": (6, "major"),
    "Gb_major": (6, "major"),
    "G_major": (7, "major"),
    "G#_major": (8, "major"),
    "Ab_major": (8, "major"),
    "A_major": (9, "major"),
    "A#_major": (10, "major"),
    "Bb_major": (10, "major"),
    "B_major": (11, "major"),
    "C_minor": (0, "minor"),
    "C#_minor": (1, "minor"),
    "Db_minor": (1, "minor"),
    "D_minor": (2, "minor"),
    "D#_minor": (3, "minor"),
    "Eb_minor": (3, "minor"),
    "E_minor": (4, "minor"),
    "F_minor": (5, "minor"),
    "F#_minor": (6, "minor"),
    "Gb_minor": (6, "minor"),
    "G_minor": (7, "minor"),
    "G#_minor": (8, "minor"),
    "Ab_minor": (8, "minor"),
    "A_minor": (9, "minor"),
    "A#_minor": (10, "minor"),
    "Bb_minor": (10, "minor"),
    "B_minor": (11, "minor"),
}


def parse_scale(scale_str: str) -> tuple[int, str]:
    """Parse a scale string like 'C_major' into (root, mode) tuple."""
    if scale_str not in SCALE_MAP:
        raise ValueError(
            f"Unknown scale '{scale_str}'. Valid options: {sorted(SCALE_MAP)}"
        )
    return SCALE_MAP[scale_str]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_midi_files(input_dir: Path) -> list[tuple[Path, muspy.Music]]:
    """
    Recursively search *input_dir* (up to one level deep) for .mid files and
    load them with muspy.  Files that fail to load are skipped with a warning.

    Returns a list of (path, Music) tuples for successfully loaded files.
    """
    # Collect .mid files: direct children + immediate sub-folder children
    candidates: list[Path] = sorted(input_dir.rglob("*.mid"))
    if not candidates:
        logger.warning("No .mid files found under %s", input_dir)

    results: list[tuple[Path, muspy.Music]] = []
    for path in candidates:
        try:
            music = muspy.read_midi(path)
            results.append((path, music))
            logger.info("Loaded %s", path.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s — failed to load: %s", path.name, exc)

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _safe(value: object) -> float:
    """Convert a metric return value to float, replacing None/NaN with nan."""
    if value is None:
        return float("nan")
    try:
        f = float(value)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def compute_metrics(music: muspy.Music, root: int, mode: str) -> dict[str, float]:
    """
    Compute all required muspy metrics for a single Music object.

    Parameters
    ----------
    music:
        A loaded muspy.Music instance.
    root:
        Scale root as semitone offset (0 = C, 9 = A, …).
    mode:
        Scale mode string accepted by muspy ('major' or 'minor').

    Returns
    -------
    dict mapping metric name → float (NaN when the metric cannot be computed).
    """
    metrics: dict[str, float] = {}

    def _try(fn, *args):  # noqa: ANN001
        try:
            return _safe(fn(*args))
        except Exception:  # noqa: BLE001
            return float("nan")

    metrics["pitch_range"] = _try(muspy.pitch_range, music)
    metrics["pitch_in_scale_rate"] = _try(muspy.pitch_in_scale_rate, music, root, mode)
    metrics["polyphony_rate"] = _try(muspy.polyphony_rate, music)
    metrics["polyphony"] = _try(muspy.polyphony, music)
    metrics["empty_beat_rate"] = _try(muspy.empty_beat_rate, music)
    # measure_resolution = ticks per measure (4 beats × ticks-per-beat)
    measure_resolution = 4 * (music.resolution if music.resolution else 24)
    metrics["groove_consistency"] = _try(
        muspy.groove_consistency, music, measure_resolution
    )
    metrics["n_pitches_used"] = _try(muspy.n_pitches_used, music)
    metrics["n_pitch_classes_used"] = _try(muspy.n_pitch_classes_used, music)
    metrics["drum_in_pattern_rate"] = _try(muspy.drum_in_pattern_rate, music)
    metrics["pitch_entropy"] = _try(muspy.pitch_entropy, music)
    metrics["pitch_class_entropy"] = _try(muspy.pitch_class_entropy, music)

    return metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

METRIC_COLUMNS: list[str] = [
    "pitch_range",
    "pitch_in_scale_rate",
    "polyphony_rate",
    "polyphony",
    "empty_beat_rate",
    "groove_consistency",
    "n_pitches_used",
    "n_pitch_classes_used",
    "drum_in_pattern_rate",
    "pitch_entropy",
    "pitch_class_entropy",
]


def aggregate_results(records: list[dict]) -> pd.DataFrame:
    """
    Build a summary statistics DataFrame from per-file metric records.

    Parameters
    ----------
    records:
        Each dict must contain a 'file' key plus one key per metric.

    Returns
    -------
    A DataFrame with one row per metric and columns:
    metric, mean, std, min, max, median, p25, p75, count.
    """
    per_file_df = pd.DataFrame(records)

    rows = []
    for col in METRIC_COLUMNS:
        if col not in per_file_df.columns:
            continue
        series = pd.to_numeric(per_file_df[col], errors="coerce")
        rows.append(
            {
                "metric": col,
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "median": series.median(),
                "p25": series.quantile(0.25),
                "p75": series.quantile(0.75),
                "count": int(series.count()),
            }
        )

    summary_df = pd.DataFrame(rows).set_index("metric")
    return summary_df


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _build_report(
    per_file_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    meta: dict,
) -> str:
    """Render a human-readable plain-text report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("MIDI EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Input directory : {meta['input_dir']}")
    lines.append(f"Scale           : {meta['scale']}")
    lines.append(f"Files found     : {meta['files_found']}")
    lines.append(f"Files parsed    : {meta['files_parsed']}")
    lines.append(f"Files skipped   : {meta['files_skipped']}")
    lines.append("")
    lines.append("Summary Statistics")
    lines.append("-" * 70)

    try:
        from tabulate import tabulate  # type: ignore

        table = tabulate(
            summary_df.reset_index(),
            headers="keys",
            tablefmt="github",
            floatfmt=".4f",
            showindex=False,
        )
        lines.append(table)
    except ImportError:
        # Plain-text fallback
        header = f"{'metric':<28}" + "".join(
            f"{c:>10}"
            for c in ["mean", "std", "min", "max", "median", "p25", "p75", "count"]
        )
        lines.append(header)
        lines.append("-" * len(header))
        for metric, row in summary_df.iterrows():
            line = f"{metric:<28}" + "".join(
                f"{row[c]:>10.4f}" if not math.isnan(float(row[c])) else f"{'NaN':>10}"
                for c in ["mean", "std", "min", "max", "median", "p25", "p75", "count"]
            )
            lines.append(line)

    lines.append("=" * 70)
    return "\n".join(lines)


def save_results(
    per_file_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    report_meta: dict,
) -> None:
    """
    Persist per-file CSV, summary CSV, summary JSON, and a plain-text report.

    Parameters
    ----------
    per_file_df:
        One row per MIDI file.
    summary_df:
        Summary statistics indexed by metric name.
    output_dir:
        Directory in which to write all output files.
    report_meta:
        Metadata dict forwarded to the report renderer.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    per_file_path = output_dir / "per_file_results.csv"
    per_file_df.to_csv(per_file_path, index=False)
    logger.info("Saved per-file results → %s", per_file_path)

    summary_csv_path = output_dir / "summary_statistics.csv"
    summary_df.to_csv(summary_csv_path)
    logger.info("Saved summary CSV       → %s", summary_csv_path)

    summary_json_path = output_dir / "summary_statistics.json"
    summary_dict = summary_df.to_dict(orient="index")

    # Replace NaN with None for valid JSON
    def _clean(obj):  # type: ignore[return]
        if isinstance(obj, float) and math.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        return obj

    with summary_json_path.open("w") as fh:
        json.dump(_clean(summary_dict), fh, indent=2)
    logger.info("Saved summary JSON      → %s", summary_json_path)

    report_text = _build_report(per_file_df, summary_df, report_meta)
    report_path = output_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved report            → %s", report_path)

    # Also echo the report to stdout
    print(report_text)


# ---------------------------------------------------------------------------
# Detect mode
# ---------------------------------------------------------------------------


def _detect_mode(input_dir: Path) -> str:
    """Return 'single' if input_dir has direct .mid files, 'multi' if it has sub-dirs with .mid files."""
    direct_midis = list(input_dir.glob("*.mid"))
    if direct_midis:
        return "single"
    sub_dirs_with_midi = [
        d for d in input_dir.iterdir() if d.is_dir() and list(d.glob("*.mid"))
    ]
    if sub_dirs_with_midi:
        return "multi"
    return "single"


# ---------------------------------------------------------------------------
# Evaluate single directory
# ---------------------------------------------------------------------------


def evaluate_single_dir(
    input_dir: Path,
    output_dir: Path,
    root: int,
    mode: str,
    scale_str: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full evaluation pipeline on a single directory of .mid files.

    Parameters
    ----------
    input_dir:
        Directory containing .mid files.
    output_dir:
        Directory to write output files into.
    root:
        Scale root as semitone offset.
    mode:
        Scale mode ('major' or 'minor').
    scale_str:
        Human-readable scale string for the report.

    Returns
    -------
    (per_file_df, summary_df) DataFrames.
    """
    loaded = load_midi_files(input_dir)
    files_found = len(list(input_dir.glob("*.mid")))
    files_parsed = len(loaded)
    files_skipped = files_found - files_parsed

    if files_parsed == 0:
        logger.warning("No files could be loaded in %s. Skipping.", input_dir)
        return pd.DataFrame(), pd.DataFrame()

    records: list[dict] = []
    for path, music in loaded:
        metrics = compute_metrics(music, root=root, mode=mode)
        records.append({"file": str(path), **metrics})

    per_file_df = pd.DataFrame(records)
    summary_df = aggregate_results(records)

    report_meta = {
        "input_dir": str(input_dir),
        "scale": scale_str,
        "files_found": files_found,
        "files_parsed": files_parsed,
        "files_skipped": files_skipped,
    }
    save_results(per_file_df, summary_df, output_dir, report_meta)
    return per_file_df, summary_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments, run evaluation, and save results."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated MIDI files using muspy metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs/generated_midi"),
        help="Directory containing .mid files (or sub-folders of .mid files).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/evaluation_results"),
        help="Directory where CSV / JSON / report outputs are written.",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="C_major",
        help=(
            "Scale for pitch-in-scale-rate metric, e.g. C_major, A_minor. "
            f"Valid values: {sorted(SCALE_MAP)}"
        ),
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not input_dir.exists():
        parser.error(f"--input_dir does not exist: {input_dir}")

    root, mode = parse_scale(args.scale)
    logger.info("Scale: %s  (root=%d, mode=%s)", args.scale, root, mode)

    # Detect mode -----------------------------------------------------------
    eval_mode = _detect_mode(input_dir)
    logger.info("Detected mode: %s", eval_mode)

    if eval_mode == "single":
        # Mode A: Single run folder
        logger.info("Evaluating single directory...")
        evaluate_single_dir(input_dir, output_dir, root, mode, args.scale)
    else:
        # Mode B: Parent folder of run folders
        sub_dirs: list[Path] = sorted(
            d for d in input_dir.iterdir() if d.is_dir() and list(d.glob("*.mid"))
        )
        total_runs = len(sub_dirs)
        logger.info("Found %d run sub-directories to evaluate.", total_runs)

        all_per_file_dfs: list[pd.DataFrame] = []
        all_records: list[dict] = []
        total_files_evaluated = 0

        for idx, sub_dir in enumerate(sub_dirs, start=1):
            logger.info("Processing run %d/%d: %s", idx, total_runs, sub_dir.name)
            sub_output_dir = output_dir / sub_dir.name
            per_file_df, _ = evaluate_single_dir(
                sub_dir, sub_output_dir, root, mode, args.scale
            )
            if per_file_df.empty:
                continue
            # Tag with run name
            per_file_df = per_file_df.copy()
            per_file_df.insert(0, "run", sub_dir.name)
            all_per_file_dfs.append(per_file_df)
            total_files_evaluated += len(per_file_df)

        if not all_per_file_dfs:
            logger.error("No files were evaluated across all sub-directories.")
            return

        combined_per_file_df = pd.concat(all_per_file_dfs, ignore_index=True)

        # Build combined records (drop 'run' column for aggregate_results)
        metric_cols = [
            c for c in combined_per_file_df.columns if c not in ("run", "file")
        ]
        combined_records = combined_per_file_df[["file"] + metric_cols].to_dict(
            orient="records"
        )
        combined_summary_df = aggregate_results(combined_records)

        combined_meta = {
            "input_dir": str(input_dir),
            "scale": args.scale,
            "files_found": total_files_evaluated,
            "files_parsed": total_files_evaluated,
            "files_skipped": 0,
        }

        # Save combined outputs to output_dir root
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_per_file_path = output_dir / "per_file_results.csv"
        combined_per_file_df.to_csv(combined_per_file_path, index=False)
        logger.info("Saved combined per-file results → %s", combined_per_file_path)

        summary_csv_path = output_dir / "summary_statistics.csv"
        combined_summary_df.to_csv(summary_csv_path)
        logger.info("Saved combined summary CSV       → %s", summary_csv_path)

        summary_json_path = output_dir / "summary_statistics.json"
        summary_dict = combined_summary_df.to_dict(orient="index")

        def _clean(obj):  # type: ignore[return]
            if isinstance(obj, float) and math.isnan(obj):
                return None
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            return obj

        with summary_json_path.open("w") as fh:
            json.dump(_clean(summary_dict), fh, indent=2)
        logger.info("Saved combined summary JSON      → %s", summary_json_path)

        # Build combined report with extra multi-run info
        report_lines: list[str] = []
        report_lines.append("=" * 70)
        report_lines.append("MIDI EVALUATION REPORT — COMBINED (ALL RUNS)")
        report_lines.append("=" * 70)
        report_lines.append(f"Input directory : {input_dir}")
        report_lines.append(f"Scale           : {args.scale}")
        report_lines.append(f"Total runs      : {total_runs}")
        report_lines.append(f"Total files     : {total_files_evaluated}")
        report_lines.append("")
        report_lines.append("Summary Statistics (aggregated across all runs)")
        report_lines.append("-" * 70)
        try:
            from tabulate import tabulate  # type: ignore

            table = tabulate(
                combined_summary_df.reset_index(),
                headers="keys",
                tablefmt="github",
                floatfmt=".4f",
                showindex=False,
            )
            report_lines.append(table)
        except ImportError:
            header = f"{'metric':<28}" + "".join(
                f"{c:>10}"
                for c in ["mean", "std", "min", "max", "median", "p25", "p75", "count"]
            )
            report_lines.append(header)
            report_lines.append("-" * len(header))
            for metric, row in combined_summary_df.iterrows():
                line = f"{metric:<28}" + "".join(
                    f"{row[c]:>10.4f}"
                    if not math.isnan(float(row[c]))
                    else f"{'NaN':>10}"
                    for c in [
                        "mean",
                        "std",
                        "min",
                        "max",
                        "median",
                        "p25",
                        "p75",
                        "count",
                    ]
                )
                report_lines.append(line)
        report_lines.append("=" * 70)
        report_text = "\n".join(report_lines)
        report_path = output_dir / "report.txt"
        report_path.write_text(report_text, encoding="utf-8")
        logger.info("Saved combined report → %s", report_path)
        print(report_text)

        logger.info(
            "Multi-mode evaluation complete. %d runs, %d files evaluated. Results in %s",
            total_runs,
            total_files_evaluated,
            output_dir,
        )
        return

    logger.info("Evaluation complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
