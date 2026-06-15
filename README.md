
![Python](https://img.shields.io/badge/Python-%3E%3D3.12-blue?logo=python&logoColor=white)
![Poetry](https://img.shields.io/badge/Poetry-%3E%3D2.0-60A5FA?logo=poetry&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.4%20%28CUDA%2012.4%29-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

**Table of Contents**
- [Introduction](#introduction)
- [Quick Setup](#-quick-setup)
- [Dataset Collection](#-dataset-collection)
- [Running Tasks with `poe`](#-running-tasks-with-poe)
- [Utility Scripts](#-utility-scripts)
- [Linting & Static Analysis](#-linting--static-analysis)
- [Architecture Overview](#-architecture-overview)
- [Technology Stack](#-technology-stack)
- [Configuration System](#-configuration-system)
---

# Introduction
> **Generating Adaptive Background Music with Explainability Models for Musical Improvisation**

Symbolic RAVE is a master thesis research codebase for symbolic MIDI music generation using a modular two-stage training pipeline. Stage 1 learns a compact music representation (autoregressive or VAE-based) from MIDI data using interchangeable adapters and sequence decoders. Stage 2 fine-tunes the decoder for downstream tasks. The project incorporates explainability mechanisms to generate adaptive background music for live musical improvisation, with full support for experiment tracking, hierarchical configuration, and multi-dataset training.

📄 **[Research Results & Experiment Findings](docs/results.md)**

---

## 🚀 Quick Setup

### 2.1 — Install Python 3.12 (Or newer)

Recommended option for installing specific python version is `pyenv`.
It gives you isolated, reproducible Python versions per project.

```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to shell — bash example:
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
source ~/.bashrc
```

> **Note (zsh users):** replace `~/.bashrc` with `~/.zshrc` and `bash` with `zsh` in the `pyenv init` line.

```bash
# Install and pin Python 3.12
pyenv install 3.12
pyenv local 3.12
```

### 2.2 — Install Poetry

Poetry manages all dependencies and virtual environments. 
Try to use the official installer, avoid `pip install poetry` 
(unless you don't have root's permission in repo and don't have installed curl).

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to your PATH (add this to your shell rc file to make it permanent):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Verify the installation:

```bash
poetry --version
```

### 2.3 — Clone & Install the Project

```bash
git clone <repo-url>
cd Symbolic_rave
poetry install
```

`poetry.toml` is configured to create the virtualenv **inside the project directory** (`.venv/`), so everything stays self-contained. Activate the environment:

```bash
poetry shell
```
or simply:
```bash
source .venv/bin/activate
```

### 2.4 — Optional: Mamba (State-Space Model) Support

Mamba requires native CUDA extensions and is shipped as an optional dependency group:

```bash
poetry install --with mamba
```

> ⚠️ Requires a CUDA-capable GPU and matching **CUDA 12.4** toolkit. Mamba will fail silently on CPU-only machines.

### 2.5 — GPU / PyTorch Note

The project depends on PyTorch 2.4+ built against **CUDA 12.4** (via the `pytorch_cu124` custom source in `pyproject.toml`). Confirm your driver and toolkit version before installing:

```bash
nvidia-smi
```

Make sure the reported CUDA version is ≥ 12.4.

---

## 📦 Dataset Collection

All datasets are MIDI-based and are downloaded, extracted, and structured automatically:

```bash
poe collect-dataset
```

This populates `./data/` with the following datasets:

| Dataset | Description | Genre / Domain |
|---|---|---|
| **MAESTRO v3** | Virtuosic classical piano performances, aligned with audio | Classical piano |
| **POP909** | 909 pop piano arrangements with melody, bridge, and accompaniment tracks | Pop |
| **MidiCaps** | Large-scale MIDI dataset (~168k files) with natural language captions and genre tags | Mixed / General |
| **MidiCaps Classes** | Curated genre-stratified subset of MidiCaps, split into 5 genre buckets | Genre-conditioned |

The 5 MidiCaps genre subsets are:

- `pop_rock`
- `electronic_ambient`
- `classical_soundtrack`
- `jazz_pop`
- `rock_electronic`

> ⚠️ Downloading MidiCaps requires a stable internet connection and several GB of disk space. The script handles extraction, restructuring, and cleanup automatically.

---

## 🧪 Running Tasks with `poe`

All runnable workflows are defined as **Poe the Poet** tasks in `pyproject.toml`. After `poetry shell`, run any task as `poe <task>`.

### Training

| Task | Command | Description |
|---|---|---|
| Train representation (generic) | `poe train-repr` | First-stage training with default tokenizer adapter |
| Train representation (Patchilizer) | `poe train-repr-patch` | First-stage with patch-based music representation |
| Train representation (Compound Word) | `poe train-repr-cp` | First-stage with Compound Word representation + CP-ELBO loss |
| Train second-stage decoder | `poe train-decoder` | Second-stage decoder fine-tuning |

All training runs are configured via **Hydra** YAML files in `configs/`. Override any config key directly from the CLI:

```bash
poe train-repr dataset=maestro model/decoder=gpt
```

### Generation

| Task | Command | Description |
|---|---|---|
| Generate songs | `poe generate-songs` | Generate MIDI (or WAV) samples from a trained model |

### Evaluation

| Task | Command | Description |
|---|---|---|
| Evaluate FMD | `poe evaluate-fmd` | Compute Fréchet Music Distance between generated and real distributions |

### Testing

```bash
poe test
```

---

## 🔧 Utility Scripts

These scripts are **not wired to `poe` tasks** — run them directly with `python`. They cover post-generation conversion, symbolic evaluation, dataset sanitation, and batch generation.

---

### Convert MIDI → WAV

Renders a directory of generated MIDI files to WAV, mirroring the input folder structure.

```bash
python -m src.scripts.create_wav_from_midi \
    --input_dir  outputs/generated_midi \
    --output_dir outputs/generated_waves
```

| Argument | Default | Description |
|---|---|---|
| `--input_dir` | `outputs/generated_midi` | Root folder containing `.mid`/`.midi` files (searched up to 2 levels deep) |
| `--output_dir` | `outputs/generated_waves` | Output root — folder structure is preserved, extensions changed to `.wav` |

---

### Evaluate Generated MIDI (symbolic metrics)

Runs [muspy](https://salu133445.github.io/muspy/) metrics over a directory of generated `.mid` files and writes CSV / JSON reports.

```bash
# Single run folder (Mode A)
python -m src.scripts.evaluate_midi \
    --input_dir  outputs/generated_midi/my_run \
    --output_dir outputs/evaluation_results/my_run \
    --scale C_major

# Parent folder of multiple run folders (Mode B — evaluates each sub-dir + combined)
python -m src.scripts.evaluate_midi \
    --input_dir  outputs/generated_midi \
    --output_dir outputs/evaluation_results/all_runs \
    --scale A_minor
```

The mode is **auto-detected**: if `.mid` files are found directly in `--input_dir` → Mode A; if sub-directories containing `.mid` files are found → Mode B (per-run reports + a combined aggregated report).

| Argument | Default | Description |
|---|---|---|
| `--input_dir` | `outputs/generated_midi` | Folder with `.mid` files, or parent folder of run sub-dirs |
| `--output_dir` | `outputs/evaluation_results` | Where CSV / JSON / `report.txt` are written |
| `--scale` | `C_major` | Scale for `pitch_in_scale_rate` (e.g. `C_major`, `A_minor`, `F#_major`) |

**Metrics computed per file:**

| Metric | Description |
|---|---|
| `pitch_range` | Semitone range between lowest and highest pitch |
| `pitch_in_scale_rate` | Fraction of notes that fall within the specified scale |
| `polyphony` / `polyphony_rate` | Average simultaneous notes / fraction of polyphonic beats |
| `empty_beat_rate` | Fraction of beats with no notes |
| `groove_consistency` | Rhythmic regularity across measures |
| `n_pitches_used` / `n_pitch_classes_used` | Pitch variety (absolute / modulo octave) |
| `drum_in_pattern_rate` | Drum pattern regularity |
| `pitch_entropy` / `pitch_class_entropy` | Information-theoretic pitch diversity |

Outputs per run: `per_file_results.csv`, `summary_statistics.csv`, `summary_statistics.json`, `report.txt`.

---

### Batch Generation (all checkpoints)

Runs `generate_songs` for **every `.ckpt` file** found under a checkpoint directory. Useful for comparing all trained models in one shot.

```bash
python -m src.scripts.run_multi_generate \
    --checkpoints-dir outputs/checkpoints/train_repr \
    --songs 100 \
    --format midi \
    --temperature 0.8
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoints-dir` | `outputs/checkpoints/train_repr` | Root directory scanned recursively for `.ckpt` files |
| `--songs` | `100` | Number of songs to generate per checkpoint |
| `--format` | `wav` | Export format: `midi` or `wav` |
| `--temperature` | `0.8` | Sampling temperature passed to the decoder |

---

### Filter / Validate Dataset Files

Validates every MIDI (or ABC) file in a dataset directory by trying to encode it through a `MusicAdapter`. **Files that cannot be processed are deleted in-place.** Edit the `DATASET` and adapter config at the top of the script before running.

```bash
python -m src.scripts.filter_music_files
```

> ⚠️ This script **removes corrupt or incompatible files permanently**. Run it on a copy or after committing the dataset to version control.

---

## 🔍 Linting & Static Analysis

The project uses **Ruff** for linting + formatting and **MyPy** for static type checking.

#### Ruff — check only (CI-safe, no writes)

```bash
poe ruff-lint-check     # Check lint rules (no fix)
poe ruff-format-check   # Check formatting (no fix)
```

#### Ruff — auto-fix

```bash
poe ruff-lint           # Fix lint issues automatically
poe ruff-format         # Auto-format code
poe ruff-fix            # Shorthand: runs both lint + format fixes
```

#### MyPy — type checking

```bash
poe mypy
```

Ruff is configured in `pyproject.toml` with the following active rule sets: `pycodestyle (E/W)`, `pyflakes (F)`, `pep8-naming (N)`, `isort (I)`, `flake8-bugbear (B)`, `flake8-bandit (S)`, `flake8-comprehensions (C4)`, `pyupgrade (UP)`, `flynt (FLY)`, `flake8-simplify (SIM)`, and `pylint (PL)`. Line length is **120**.

---

## 🏗️ Architecture Overview

### Two-Stage Training Pipeline

```
Stage 1 — Representation Learning
  ┌──────────────────────────────────────┐
  │  High-Level Model                    │
  │   ├── SequenceModel (autoregressive) │
  │   └── VAE (encoder + decoder)        │
  │         ↑                            │
  │  Music Adapter (tokenizer)           │
  │  Sequence Decoder                    │
  │  Decoder Wrapper                     │
  └──────────────────────────────────────┘
              ↓ trained checkpoint
Stage 2 — Decoder Fine-Tuning / Task Adaptation
  ┌──────────────────────────────────────┐
  │  train-decoder                       │
  │  (fine-tune on a specific dataset    │
  │   or with a discriminator signal)    │
  └──────────────────────────────────────┘
```

### High-Level Models (2 options)

| Model | Description |
|---|---|
| `SequenceModel` | Pure **autoregressive** next-token prediction. No encoder. Genre conditioning via a learned class embedding. |
| `VAE` | **Variational Autoencoder** — encoder projects music into a latent Gaussian; decoder reconstructs from a latent vector. Enables latent-space interpolation and sampling. |

### Music Representations / Adapters (3 options)

| Adapter | Config key | Description |
|---|---|---|
| Tokenizer | `adapter=tokenizer` | Single-token representation via MidiTok. Default scheme is **REMI**; other schemes are selectable in `configs/adapter/tokenizer.yaml` (see below). Also supports BPE / Unigram / WordPiece vocabulary training. |
| Compound Word | `adapter=compound_word` | Multiple attribute tokens grouped per note event (pitch, duration, velocity, etc.) — richer structure, requires CP-ELBO loss |
| Patchilizer | `adapter=patchilizer` | Patch-based representation — sequences of byte-level patches, similar to BPE at the MIDI event level |

The `tokenizer` adapter wraps [MidiTok](https://miditok.readthedocs.io/) and exposes every tokenization scheme it supports. Set `params.tokenizer` in `configs/adapter/tokenizer.yaml`:

| Value | MidiTok scheme | Notes |
|---|---|---|
| `remi` *(default)* | REMI | Position-based events; widely used baseline |
| `midilike` | MIDI-Like | Directly mirrors raw MIDI message stream |
| `tsd` | TSD | Time-Shift Duration encoding |
| `structured` | Structured | Alternating pitch/duration structure |
| `octuple` | Octuple | 8-tuple per note (position, pitch, velocity, duration, …) |
| `mumidi` | MuMIDI | Multi-track representation with bar/beat tokens |
| `mmm` | MMM | Multi-track Music Machine representation |

Vocabulary size and the sub-word training method (`BPE`, `Unigram`, or `WordPiece`) are also configurable in the same file.

### Sequence Decoders (5 options)

| Decoder | Description |
|---|---|
| `GPT` | Standard causal Transformer decoder (GPT-style), learned absolute positional embeddings |
| `Transformer-XL` | Transformer-XL with **relative positional encoding** and memory segments for longer context |
| `Transformer RoPE` | Causal Transformer with **Rotary Positional Encoding (RoPE)** — no learned position embeddings |
| `Mistral` | Mistral-style decoder with **sliding-window attention** for efficient long-sequence modeling |
| `Mamba` | **State-Space Model** (SSM) — linear-time sequence modeling, no attention at all *(optional dependency)* |

### Decoder Wrappers

Wrappers adapt the raw sequence decoder to a specific music representation:

| Wrapper | Used with |
|---|---|
| `SimpleWrapper` | Standard tokenizer |
| `CompoundWordWrapper` | Compound Word representation — multi-head output per group |
| `PatchWrapper` | Patchilizer — patch-level input/output |

---

## 🛠️ Technology Stack

| Technology | Version | Role |
|---|---|---|
| **Python** | ≥ 3.12 | Language |
| **PyTorch** | ≥ 2.4 (CUDA 12.4) | Deep learning framework |
| **PyTorch Lightning** | ≥ 2.5 | Training loop, checkpointing, logging |
| **Hydra** | ≥ 1.3 | Hierarchical config management, CLI overrides |
| **Weights & Biases** | ≥ 0.19 | Experiment tracking, metric logging, artifact storage |
| **MidiTok** | ≥ 3.0 | MIDI tokenization strategies |
| **MuSPy** | ≥ 0.5 | MIDI dataset loading & music processing |
| **HuggingFace Transformers** | 4.57.x | Pretrained model components |
| **Mamba-SSM** | ≥ 2.3 *(optional)* | State-space sequence model |
| **Poetry** | ≥ 1.8 | Dependency management & virtual environments |
| **Poe the Poet** | ≥ 0.32 | Task runner (defined in `pyproject.toml`) |
| **Ruff** | ≥ 0.8 | Linter + formatter (replaces flake8, isort, black) |
| **MyPy** | ≥ 1.14 | Static type checker |
| **Pytest** | ≥ 8.3 | Test framework |

---

## ⚙️ Configuration System

All experiments are configured with **Hydra**. Config files live in `configs/` and follow a hierarchical structure:

```
configs/
├── train_repr.yaml          # First-stage training defaults
├── train_decoder.yaml       # Second-stage training defaults
├── generate_song.yaml       # Generation defaults
├── evaluate_fmd.yaml        # Evaluation defaults
├── adapter/                 # compound_word, patchilizer, tokenizer
├── dataset/                 # maestro, midicaps, midicaps_classes, pop909
├── model/
│   ├── decoder/             # gpt, mamba, mistral, transformer_rope, transformer_xl
│   ├── decoder_wrapper/     # simple, compound_word, patch
│   └── encoder/
├── loss/                    # elbo, cp_elbo
└── discriminator/           # clamp2, dense
```

Override any key on the command line:

```bash
poe train-repr dataset=maestro model/decoder=transformer_rope adapter=patchilizer
```
