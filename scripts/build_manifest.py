"""
Step 5 & 6: Extract audio bytes from parquet files, write WAV files,
build a clean manifest CSV, and remove bad/duplicate samples.
Output: datasets/audio/          (extracted WAV files)
        datasets/manifest_clean.csv  (audio_path, transcript, speaker_id)
"""

import io
import csv
import hashlib
import re
from pathlib import Path

import pandas as pd
import soundfile as sf

PARQUET_DIR  = Path("datasets/raw/khmer_speech/data")
AUDIO_OUT    = Path("datasets/audio")
OUT_CSV      = Path("datasets/manifest_clean.csv")

MIN_TRANSCRIPT_LEN = 2
MAX_TRANSCRIPT_LEN = 500
MIN_DURATION       = 0.5   # seconds
MAX_DURATION       = 30.0  # Whisper hard limit


def normalize_transcript(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_valid_audio(audio_bytes: bytes) -> bool:
    try:
        with io.BytesIO(audio_bytes) as buf:
            data, sr = sf.read(buf)
        return len(data) > 0
    except Exception:
        return False


def process_parquet(parquet_path: Path, rows: list, seen: set):
    df = pd.read_parquet(parquet_path)

    for _, row in df.iterrows():
        transcript = normalize_transcript(str(row.get("transcript", "")))
        speaker_id = str(row.get("speaker_id", "unknown")).strip()
        duration   = float(row.get("duration", 0))
        sentence_id = str(row.get("sentence_id", "")).strip()

        # Basic filters
        if len(transcript) < MIN_TRANSCRIPT_LEN or len(transcript) > MAX_TRANSCRIPT_LEN:
            continue
        if duration < MIN_DURATION or duration > MAX_DURATION:
            continue

        # Dedup by transcript hash
        key = hashlib.md5(transcript.encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)

        # Extract audio bytes
        audio_field = row.get("audio", {})
        if isinstance(audio_field, dict):
            audio_bytes = audio_field.get("bytes", b"")
        elif isinstance(audio_field, bytes):
            audio_bytes = audio_field
        else:
            continue

        if not audio_bytes:
            continue

        # Write WAV file
        wav_name = f"{sentence_id or key}.wav"
        wav_path = AUDIO_OUT / speaker_id / wav_name
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with io.BytesIO(audio_bytes) as buf:
                data, sr = sf.read(buf)
            sf.write(str(wav_path), data, sr, subtype="PCM_16")
        except Exception as e:
            continue

        rows.append({
            "audio_path": str(wav_path),
            "transcript": transcript,
            "speaker_id": speaker_id,
        })


if __name__ == "__main__":
    parquet_files = sorted(PARQUET_DIR.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files.")
    print(f"Extracting audio to {AUDIO_OUT} ...")

    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen: set[str] = set()
    failed = 0

    for i, pf in enumerate(parquet_files, 1):
        if i % 50 == 0 or i == 1:
            print(f"  [{i}/{len(parquet_files)}] {pf.name}  ({len(rows)} samples so far)")
        try:
            process_parquet(pf, rows, seen)
        except Exception as e:
            print(f"  ERROR in {pf.name}: {e}")
            failed += 1

    print(f"\nExtraction done. Valid samples: {len(rows)}  Failed files: {failed}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "speaker_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved manifest to {OUT_CSV}")
