"""
Step 8: Split the resampled manifest into train / val / test CSVs.
Default split: 90% train, 5% val, 5% test (stratified by speaker when possible).
Output: data/train.csv, data/val.csv, data/test.csv
"""

import csv
import random
from pathlib import Path
from collections import defaultdict

IN_CSV = Path("datasets/manifest_resampled.csv")
OUT_DIR = Path("data")

TRAIN_RATIO = 0.90
VAL_RATIO   = 0.05
# TEST_RATIO is whatever remains

SEED = 42


def split_rows(rows, train_r, val_r, seed):
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return rows[:n_train], rows[n_train:n_train + n_val], rows[n_train + n_val:]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found. Run resample_audio.py first.")
        exit(1)

    with open(IN_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    fieldnames = ["audio_path", "transcript", "speaker_id"]

    # Stratify by speaker so each speaker appears in all splits
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker_id"]].append(row)

    train_rows, val_rows, test_rows = [], [], []
    for speaker_rows in by_speaker.values():
        t, v, te = split_rows(speaker_rows, TRAIN_RATIO, VAL_RATIO, SEED)
        train_rows.extend(t)
        val_rows.extend(v)
        test_rows.extend(te)

    # Final shuffle so ordering isn't speaker-grouped
    rng = random.Random(SEED)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    write_csv(OUT_DIR / "train.csv", train_rows, fieldnames)
    write_csv(OUT_DIR / "val.csv",   val_rows,   fieldnames)
    write_csv(OUT_DIR / "test.csv",  test_rows,  fieldnames)

    print(f"Split complete:")
    print(f"  train : {len(train_rows):>6} samples  → data/train.csv")
    print(f"  val   : {len(val_rows):>6} samples  → data/val.csv")
    print(f"  test  : {len(test_rows):>6} samples  → data/test.csv")
