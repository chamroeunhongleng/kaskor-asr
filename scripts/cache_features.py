"""
Pre-compute Whisper mel-spectrogram features for all audio files and save as
float16 .npy files. Run this ONCE after epoch 1 finishes — then epochs 2-5
load tiny arrays instead of decoding raw audio, removing the disk I/O bottleneck.

Expected output: ~43 GB in datasets/mel_cache/
Expected runtime: ~1-2 hours for 96K samples
"""

import csv
import unicodedata
import re
import numpy as np
import soundfile as sf
from pathlib import Path
from transformers import WhisperProcessor

BASE_MODEL  = "openai/whisper-small"
LANGUAGE    = "km"
TASK        = "transcribe"
SR          = 16000
MAX_SAMPLES = SR * 30
CACHE_DIR   = Path("datasets/mel_cache")
CSVS        = ["data/train.csv", "data/val.csv", "data/test.csv"]


def normalize_khmer(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def _resample(audio, orig_sr, target_sr):
    try:
        import soxr
        return soxr.resample(audio, orig_sr, target_sr)
    except ImportError:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)

    all_rows = []
    for csv_path in CSVS:
        with open(csv_path, encoding="utf-8", newline="") as f:
            all_rows.extend(list(csv.DictReader(f)))
    print(f"Total samples across all splits: {len(all_rows)}")

    already_cached = sum(
        1 for r in all_rows
        if (CACHE_DIR / (Path(r["audio_path"]).stem + ".npy")).exists()
    )
    print(f"Already cached: {already_cached} / {len(all_rows)}")

    skipped = 0
    newly_cached = 0

    for i, row in enumerate(all_rows):
        audio_path  = Path(row["audio_path"])
        cache_path  = CACHE_DIR / (audio_path.stem + ".npy")

        if cache_path.exists():
            continue

        transcript = normalize_khmer(row.get("transcript", ""))
        if not transcript or not audio_path.is_file():
            skipped += 1
            continue

        try:
            audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SR:
                audio = _resample(audio, sr, SR).astype("float32")
            if len(audio) > MAX_SAMPLES:
                audio = audio[:MAX_SAMPLES]

            features = processor(
                audio, sampling_rate=SR, return_tensors="np"
            ).input_features[0]                       # shape (80, 3000) float32

            np.save(str(cache_path), features.astype(np.float16))
            newly_cached += 1

        except Exception as e:
            skipped += 1
            print(f"  [WARN] {audio_path.name}: {e}")

        if (i + 1) % 500 == 0:
            total_done = already_cached + newly_cached
            pct = total_done / len(all_rows) * 100
            print(f"  [{i+1}/{len(all_rows)}]  {pct:.1f}%  newly cached: {newly_cached}  skipped: {skipped}")

    print(f"\nDone. Newly cached: {newly_cached} | Skipped: {skipped}")
    cache_size_gb = sum(f.stat().st_size for f in CACHE_DIR.glob("*.npy")) / 1e9
    print(f"Cache size on disk: {cache_size_gb:.1f} GB")


if __name__ == "__main__":
    main()
