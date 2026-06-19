"""
Step 7: Standardize all audio files to 16 kHz mono WAV.
Reads datasets/manifest_clean.csv, resamples each file,
writes output to datasets/audio_resampled/, updates the manifest.
Output: datasets/manifest_resampled.csv
"""

import csv
import shutil
from pathlib import Path

import librosa
import soundfile as sf

IN_CSV = Path("datasets/manifest_clean.csv")
OUT_CSV = Path("datasets/manifest_resampled.csv")
OUT_AUDIO_DIR = Path("datasets/audio_resampled")
TARGET_SR = 16000


def resample_file(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        audio, sr = librosa.load(str(src), sr=TARGET_SR, mono=True)
        sf.write(str(dst), audio, TARGET_SR, subtype="PCM_16")
        return True
    except Exception as e:
        print(f"  ERROR resampling {src}: {e}")
        return False


if __name__ == "__main__":
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found. Run build_manifest.py first.")
        exit(1)

    with open(IN_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Resampling {len(rows)} files to {TARGET_SR} Hz mono WAV...")

    out_rows = []
    ok = 0
    fail = 0

    for i, row in enumerate(rows, 1):
        src = Path(row["audio_path"])
        rel = src.name  # use flat naming; change to src.relative_to(...) if needed
        dst = OUT_AUDIO_DIR / (src.stem + ".wav")

        if (i % 100) == 0:
            print(f"  {i}/{len(rows)}...")

        if dst.exists():
            # Already done — skip reprocessing
            out_rows.append({
                "audio_path": str(dst),
                "transcript": row["transcript"],
                "speaker_id": row["speaker_id"],
            })
            ok += 1
            continue

        if resample_file(src, dst):
            out_rows.append({
                "audio_path": str(dst),
                "transcript": row["transcript"],
                "speaker_id": row["speaker_id"],
            })
            ok += 1
        else:
            fail += 1

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "speaker_id"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nDone. OK: {ok}  Failed: {fail}")
    print(f"Saved resampled manifest to {OUT_CSV}")
