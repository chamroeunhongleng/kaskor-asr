"""
Diagnostic: explain the eval_loss / CER gap.

Epoch 5 reported eval_loss 0.0346 (perplexity ~1.035) but CER 17.48%. A model
that good under teacher forcing should decode almost perfectly. That gap points
at generation, not at the weights, so this script decodes a sample of the val
set and reports the *distribution* of per-utterance CER instead of one aggregate
number.

Two failure shapes look identical in the aggregate but need opposite fixes:
  * uniform  — every utterance ~17% wrong  -> the model really is weak
  * long-tail — most utterances ~0-2%, a few catastrophic (repetition loops,
                hallucinated continuations on padded silence) -> decoding fix

Usage:
    python scripts/diagnose_cer.py                     # 64 samples, greedy
    python scripts/diagnose_cer.py --n 200 --beams 2   # match run_eval.py
    python scripts/diagnose_cer.py --split test
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
MAX_SAMPLES = SR * 30


def normalize_khmer(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: str, b: str) -> int:
    """Levenshtein on characters. Same quantity jiwer/evaluate's CER counts."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        try:
            import soxr
            audio = soxr.resample(audio, sr, SR).astype("float32")
        except ImportError:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR).astype("float32")
    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best"))
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=225)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / ".tmp" / "diagnose_cer.json"))
    args = ap.parse_args()

    csv_path = ROOT / "data" / f"{args.split}.csv"
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)]

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(rows), size=min(args.n, len(rows)), replace=False)
    rows = [rows[i] for i in idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"checkpoint : {args.checkpoint}")
    print(f"split      : {args.split}  ({len(rows)} sampled of {csv_path.name})")
    print(f"decoding   : beams={args.beams}  max_new_tokens={args.max_new_tokens}")
    print(f"device     : {device} / {dtype}\n")

    processor = WhisperProcessor.from_pretrained(args.checkpoint, language="km", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.checkpoint, torch_dtype=dtype)
    model.to(device).eval()
    model.generation_config.language = "km"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    records = []
    for n, row in enumerate(rows, 1):
        path = row["audio_path"]
        if not Path(path).is_file():
            path = str(ROOT / path)
            if not Path(path).is_file():
                continue
        audio = load_audio(path)
        ref = normalize_khmer(row.get("transcript", ""))
        if not ref:
            continue

        feats = processor(audio, sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(device, dtype=dtype)
        with torch.no_grad():
            ids = model.generate(
                feats,
                language="km",
                task="transcribe",
                num_beams=args.beams,
                max_new_tokens=args.max_new_tokens,
            )
        hyp = normalize_khmer(processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0])

        dist = edit_distance(hyp, ref)
        records.append({
            "audio": Path(path).name,
            "speaker": row.get("speaker_id", ""),
            "dur_s": round(len(audio) / SR, 2),
            "ref_chars": len(ref),
            "hyp_chars": len(hyp),
            "dist": dist,
            "cer": dist / max(1, len(ref)),
            "ref": ref,
            "hyp": hyp,
        })
        if n % 16 == 0:
            print(f"  ...{n}/{len(rows)}")

    if not records:
        print("No usable rows — check audio_path resolution.")
        return 1

    cers = np.array([r["cer"] for r in records])
    # Corpus CER is what evaluate/jiwer reports: total distance / total ref chars.
    corpus_cer = sum(r["dist"] for r in records) / sum(r["ref_chars"] for r in records)

    print("\n" + "=" * 62)
    print(f"CORPUS CER (aggregate, = what run_eval.py reports) : {corpus_cer*100:6.2f}%")
    print(f"MEAN per-utterance CER                             : {cers.mean()*100:6.2f}%")
    print(f"MEDIAN per-utterance CER                           : {np.median(cers)*100:6.2f}%")
    print("=" * 62)
    print("\nper-utterance CER distribution:")
    for p in (10, 25, 50, 75, 90, 95, 99):
        print(f"  p{p:<3} {np.percentile(cers, p)*100:8.2f}%")
    print(f"  max  {cers.max()*100:8.2f}%")

    exact = int((cers == 0).sum())
    under2 = int((cers < 0.02).sum())
    over50 = int((cers > 0.50).sum())
    print(f"\n  exact match      : {exact:4d} / {len(records)}  ({exact/len(records)*100:.1f}%)")
    print(f"  under 2% CER     : {under2:4d} / {len(records)}  ({under2/len(records)*100:.1f}%)")
    print(f"  over 50% CER     : {over50:4d} / {len(records)}  ({over50/len(records)*100:.1f}%)")

    # How much of the total error is concentrated in the worst utterances?
    by_dist = sorted(records, key=lambda r: r["dist"], reverse=True)
    total_dist = sum(r["dist"] for r in records)
    for k in (1, 5, 10):
        if k <= len(by_dist):
            share = sum(r["dist"] for r in by_dist[:k]) / max(1, total_dist)
            print(f"  worst {k:2d} utts   : {share*100:5.1f}% of all character errors")

    # Length blow-up is the signature of a repetition loop.
    ratios = np.array([r["hyp_chars"] / max(1, r["ref_chars"]) for r in records])
    print(f"\n  hyp/ref length ratio: median {np.median(ratios):.2f}  max {ratios.max():.2f}")
    runaway = [r for r in records if r["hyp_chars"] > 2 * r["ref_chars"] + 20]
    print(f"  runaway outputs (hyp > 2x ref) : {len(runaway)} / {len(records)}")

    print("\n" + "=" * 62)
    print("WORST 5 UTTERANCES")
    print("=" * 62)
    for r in by_dist[:5]:
        print(f"\n[{r['audio']}] spk={r['speaker']} dur={r['dur_s']}s "
              f"CER={r['cer']*100:.1f}% dist={r['dist']} ref={r['ref_chars']}ch hyp={r['hyp_chars']}ch")
        print(f"  REF: {r['ref'][:300]}")
        print(f"  HYP: {r['hyp'][:300]}")

    print("\n" + "=" * 62)
    print("TYPICAL (median) UTTERANCE")
    print("=" * 62)
    mid = sorted(records, key=lambda r: r["cer"])[len(records) // 2]
    print(f"[{mid['audio']}] CER={mid['cer']*100:.1f}%")
    print(f"  REF: {mid['ref'][:300]}")
    print(f"  HYP: {mid['hyp'][:300]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "split": args.split,
            "n": len(records),
            "beams": args.beams,
            "corpus_cer": corpus_cer,
            "mean_cer": float(cers.mean()),
            "median_cer": float(np.median(cers)),
            "records": records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nFull records -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
