# KASEKOR ASR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model](https://img.shields.io/badge/model-whisper--small-blue)](https://huggingface.co/openai/whisper-small)
[![CER](https://img.shields.io/badge/CER-17.48%25-brightgreen)](#results)

Fine-tuning [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) for **Khmer (ខ្មែរ) speech-to-text**.

This repository contains the full, reproducible pipeline used to build KASEKOR v0.0 — from raw audio to a trained, exportable model — along with evaluation and inference tooling.

## Results

| Checkpoint | Epoch | CER ↓      | WER ↓ | Eval loss |
| ---------- | ----- | ---------- | ----- | --------- |
| best       | 5     | **17.48%** | 67.2% | 0.0346    |
| —          | 1     | 18.24%     | —     | 0.0442    |

> **Why CER, not WER?** Khmer text has no spaces between words, so word-level WER is close to meaningless. **Character Error Rate (CER)** is the standard ASR metric for Khmer, Thai, Lao and CJK languages, and is the primary metric here; WER is reported only as a secondary number.

## Pipeline

The scripts are numbered in the order they run. Each is standalone and reads/writes the manifests described below.

| Step | Script                    | Purpose                                                              |
| ---- | ------------------------- | ------------------------------------------------------------------- |
| 4    | `scripts/inspect_dataset.py` | Inspect the raw downloaded dataset.                              |
| 5–6  | `scripts/build_manifest.py`  | Extract audio from parquet → WAV, build a clean manifest, drop bad/duplicate samples. |
| 7    | `scripts/resample_audio.py`  | Standardize all audio to 16 kHz mono WAV.                        |
| 8    | `scripts/split_data.py`      | Split into `train` / `val` / `test` (90 / 5 / 5, speaker-stratified). |
| —    | `scripts/cache_features.py`  | Pre-compute Whisper mel-spectrograms as `.npy` to remove the I/O bottleneck during training. |
| 9    | `scripts/train.py`           | Fine-tune Whisper-small (CER-primary, NFC-normalised, 16 kHz-verified). |
| —    | `scripts/run_eval.py`        | Standalone evaluation of a checkpoint on the val set.            |
| 10   | `scripts/save_model.py`      | Export the best checkpoint as a self-contained model package.   |
| —    | `scripts/push_to_hub.py`     | Publish the exported model to the Hugging Face Hub (one-time, so the `kaskor` CLI can auto-download it). |
| —    | `scripts/transcribe.py`      | Inference from a file, folder, or microphone.                   |

## Install the CLI

For transcription only — no need to clone the repo or set up the training pipeline:

```bash
pip install git+https://github.com/chamroeunhongleng/kaskor-asr.git
kaskor audio.wav
kaskor audio.wav --ref "ខ្ញុំទៅផ្សារ"
kaskor --mic --mic-seconds 15
kaskor audio/ --batch --output results.tsv
```

The CLI downloads the trained model from the Hugging Face Hub the first time it runs. Pass `--model /path/to/checkpoint` to use a local checkpoint instead (e.g. one produced by this repo's own training pipeline). Microphone input needs the optional extra: `pip install "kaskor-asr[mic] @ git+https://github.com/chamroeunhongleng/kaskor-asr.git"`.

## Getting started (full training pipeline)

The rest of this README covers reproducing the training pipeline from raw audio — not needed if you only want to transcribe with the released model above.

### 1. Install dependencies

`torch` / `torchaudio` are CUDA-build-specific and must be installed first, matching your GPU. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# GPU (example: CUDA 12.8)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# CPU only
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

### 2. Configure notifications (optional)

Training and evaluation can send Telegram progress pings. Copy the template and fill in your bot token / chat id:

```bash
cp .env.example .env
```

`.env` is git-ignored and never committed.

### 3. Train

```bash
python scripts/train.py
```

### 4. Evaluate a checkpoint

```bash
python scripts/run_eval.py
python scripts/run_eval.py --checkpoint checkpoints/checkpoint-27050
```

### 5. Transcribe

```bash
python scripts/transcribe.py audio.wav
python scripts/transcribe.py audio.wav --ref "ខ្ញុំទៅផ្សារ"
python scripts/transcribe.py --mic --mic-seconds 15
python scripts/transcribe.py audio/ --batch
```

This is the same code as the installable `kaskor` command (see [Install the CLI](#install-the-cli)) — use whichever fits: `python scripts/transcribe.py` when working inside a clone of this repo, `kaskor` when installed via pip elsewhere.

### 6. Publish the model (optional, one-time)

So the `kaskor` CLI can download the trained model instead of requiring a local checkpoint:

```bash
pip install huggingface_hub
huggingface-cli login
python scripts/push_to_hub.py
```

## Repository layout

```
data/            train / val / test manifests (audio_path, transcript, speaker_id)
scripts/         the numbered pipeline + eval + inference
kaskor/          installable package backing the `kaskor` CLI (pip install git+...)
pyproject.toml   packaging config for the `kaskor` console script
requirements.txt Python dependencies for the full training pipeline (torch installed separately)
.env.example     template for Telegram notification secrets
```

## Data & model weights

The audio dataset and trained checkpoints are **not** stored in this repository (tens of GB of WAV files and multi-hundred-MB weights). They are excluded via `.gitignore`:

- `datasets/` — extracted, resampled audio and the mel-feature cache
- `checkpoints/` — training checkpoints and the exported model

The CSV manifests under `data/` reference audio by relative path, so restore the `datasets/` tree (or re-run steps 5–7) before training.

## License

Released under the [MIT License](LICENSE).
