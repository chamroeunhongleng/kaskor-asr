"""
Step 4: Inspect the downloaded raw dataset.
Prints file counts, folder structure, and sample entries.
"""

import os
from pathlib import Path
import json

RAW_DIR = Path("datasets/raw/khmer_speech")


def inspect(root: Path, max_files: int = 10):
    print(f"\n=== Inspecting: {root} ===\n")
    if not root.exists():
        print(f"ERROR: {root} does not exist. Run the download step first.")
        return

    counts: dict[str, int] = {}
    samples: list[str] = []

    for p in root.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower()
            counts[ext] = counts.get(ext, 0) + 1
            if len(samples) < max_files:
                samples.append(str(p))

    print("File counts by extension:")
    for ext, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {ext or '(no ext)':>10}: {n}")

    print(f"\nSample paths (up to {max_files}):")
    for s in samples:
        print(f"  {s}")

    # Try to read any metadata/JSON/CSV
    for meta_file in list(root.rglob("*.json"))[:3] + list(root.rglob("*.csv"))[:3]:
        print(f"\nPeeking at: {meta_file}")
        try:
            if meta_file.suffix == ".json":
                with open(meta_file, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    print(f"  JSON list with {len(data)} entries. First entry:")
                    print(f"  {data[0]}")
                elif isinstance(data, dict):
                    print(f"  JSON dict keys: {list(data.keys())[:10]}")
            else:
                with open(meta_file, encoding="utf-8") as f:
                    lines = [f.readline() for _ in range(3)]
                print(f"  First 3 lines: {lines}")
        except Exception as e:
            print(f"  Could not read: {e}")


if __name__ == "__main__":
    inspect(RAW_DIR)
