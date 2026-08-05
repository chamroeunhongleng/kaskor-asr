"""
Standalone evaluation for KASEKOR ASR.
Loads the latest checkpoint, runs inference on val set, reports CER/WER.
Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/checkpoint-10820
"""

import csv
import json
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np
import soundfile as sf
import evaluate
from torch.utils.data import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

EVAL_HISTORY_FILE = Path("eval_history.json")

BASE_MODEL = "openai/whisper-small"
LANGUAGE   = "km"
TASK       = "transcribe"
VAL_CSV    = Path("data/val.csv")
OUTPUT_DIR = Path("checkpoints")
CACHE_DIR  = Path(__file__).resolve().parent.parent / "datasets" / "mel_cache"
BATCH_SIZE = 8
SR         = 16000
MAX_SAMPLES = SR * 30
BF16        = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
NUM_WORKERS = 2

from _env import tg_config
TG_TOKEN, TG_CHAT = tg_config()


def _tg(text):
    try:
        import requests as _r
        _r.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT, "text": text}, timeout=10)
    except Exception as e:
        print(f"[Telegram error] {e}")


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


class KhmerASRDataset(Dataset):
    def __init__(self, csv_path: Path, processor):
        with open(csv_path, encoding="utf-8", newline="") as f:
            raw = list(csv.DictReader(f))
        self.rows = []
        for row in raw:
            transcript = normalize_khmer(row.get("transcript", ""))
            if transcript and Path(row["audio_path"]).is_file():
                row["transcript"] = transcript
                self.rows.append(row)

        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        cache_path = CACHE_DIR / (Path(row["audio_path"]).stem + ".npy")
        if cache_path.exists():
            arr = np.load(str(cache_path)).astype(np.float32)
            input_features = torch.from_numpy(arr)
        else:
            audio, sr = sf.read(row["audio_path"], dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SR:
                audio = _resample(audio, sr, SR).astype("float32")
            if len(audio) > MAX_SAMPLES:
                audio = audio[:MAX_SAMPLES]
            input_features = self.processor(
                audio, sampling_rate=SR, return_tensors="pt"
            ).input_features[0]

        labels = self.processor.tokenizer(
            row["transcript"], return_tensors="pt", max_length=448, truncation=True,
        ).input_ids[0]
        return {"input_features": input_features, "labels": labels}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


_cer_metric = evaluate.load("cer")
_wer_metric = evaluate.load("wer")


def compute_metrics(pred, processor):
    pred_ids  = pred.predictions
    label_ids = np.asarray(pred.label_ids).copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = [normalize_khmer(t) for t in
                 processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
    label_str = [normalize_khmer(t) for t in
                 processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    cer = _cer_metric.compute(predictions=pred_str, references=label_str)
    wer = _wer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": round(cer, 4), "wer": round(wer, 4)}


def find_latest_checkpoint(output_dir: Path):
    ckpts = sorted(output_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else None


def checkpoint_epoch(ckpt: str) -> int:
    state_file = Path(ckpt) / "trainer_state.json"
    if state_file.exists():
        with open(state_file, encoding="utf-8-sig") as f:
            return round(json.load(f).get("epoch", 0))
    return 0


def eval_signature(entry: dict) -> tuple:
    """What makes two CER numbers comparable.

    A CER over 1000 subsampled utterances at beams=1 is not the same quantity as
    one over the full 4807 at beams=2, and neither is comparable to a number
    produced before the 225-token decode cap was lifted. Entries missing these
    fields are pre-fix records, tagged so they never win a comparison.
    """
    return (entry.get("split", "val"), entry.get("samples"), entry.get("beams"),
            entry.get("max_gen_len", 225))


def record_eval_history(epoch: int, ckpt_name: str, cer: float, wer: float,
                        eval_loss: float, split: str, samples: int, beams: int):
    history = []
    if EVAL_HISTORY_FILE.exists():
        with open(EVAL_HISTORY_FILE, encoding="utf-8") as f:
            history = json.load(f)
    entry = {"epoch": epoch, "checkpoint": ckpt_name, "cer": cer, "wer": wer,
             "eval_loss": eval_loss, "split": split, "samples": samples,
             "beams": beams, "max_gen_len": 448}
    sig = eval_signature(entry)
    history = [e for e in history if not (e["epoch"] == epoch and eval_signature(e) == sig)]
    history.append(entry)
    history.sort(key=lambda e: (e["epoch"], str(eval_signature(e))))
    with open(EVAL_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return history


def _flag(name, cast, default):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def main():
    # Allow --checkpoint override
    if "--checkpoint" in sys.argv:
        ckpt = sys.argv[sys.argv.index("--checkpoint") + 1]
    else:
        ckpt = find_latest_checkpoint(OUTPUT_DIR)

    if not ckpt:
        print("No checkpoint found.")
        return

    # Loop-driver flags: which split to score, decode width, and a machine-readable
    # result file so train_loop.py doesn't have to scrape stdout.
    split     = _flag("--split", str, "val")
    beams     = _flag("--beams", int, 2)
    json_out  = _flag("--json-out", str, "")
    record    = "--no-record" not in sys.argv
    eval_csv  = Path("data") / f"{split}.csv"

    print(f"Evaluating checkpoint: {ckpt}  (split={split}, beams={beams})")
    _tg(f"Eval STARTED\nCheckpoint: {ckpt}\nSplit: {split}\nBeams: {beams}")

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(ckpt)

    model.generation_config.language           = LANGUAGE
    model.generation_config.task               = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids            = None
    model.config.suppress_tokens               = []
    model.config.use_cache                     = True   # enable for faster inference

    decoder_start_token_id = model.config.decoder_start_token_id

    val_ds   = KhmerASRDataset(eval_csv, processor)

    # Scoring all 4.8k utterances takes over an hour, which makes a multi-round
    # loop eval-bound. A fixed-seed subsample keeps rounds comparable to each
    # other while staying cheap; the loop re-scores the full split at the end.
    n_eval = _flag("--eval-samples", int, 0)
    if n_eval and n_eval < len(val_ds.rows):
        rng = random.Random(1234)          # fixed: every round scores the SAME utterances
        val_ds.rows = rng.sample(val_ds.rows, n_eval)
        print(f"Subsampled eval set to {len(val_ds.rows)} utterances (seed 1234).")
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
    )

    args = Seq2SeqTrainingArguments(
        output_dir              = str(OUTPUT_DIR),
        per_device_eval_batch_size = BATCH_SIZE,
        bf16                    = BF16,
        fp16                    = False,
        predict_with_generate   = True,
        # 448 = model.config.max_target_positions. The old value of 225 capped
        # decoding at ~102 Khmer characters (Khmer is ~2.2 tokens/char under
        # Whisper's byte-fallback BPE), which truncated 51% of val references
        # mid-word. That truncation, not the acoustic model, produced most of
        # the 17.48% CER reported for epoch 5.
        generation_max_length   = 448,
        generation_num_beams    = beams,
        report_to               = "none",
        remove_unused_columns   = False,
        dataloader_num_workers  = NUM_WORKERS,
        dataloader_pin_memory   = True,
    )

    trainer = Seq2SeqTrainer(
        model         = model,
        args          = args,
        eval_dataset  = val_ds,
        data_collator = collator,
        compute_metrics = lambda pred: compute_metrics(pred, processor),
    )

    t0 = time.time()
    results = trainer.evaluate()
    elapsed = time.time() - t0

    cer = round(results.get("eval_cer", float("nan")) * 100, 2)
    wer = round(results.get("eval_wer", float("nan")) * 100, 1)
    loss = round(results.get("eval_loss", float("nan")), 4)
    mins = round(elapsed / 60, 1)

    epoch = checkpoint_epoch(ckpt)
    prior = []
    if EVAL_HISTORY_FILE.exists():
        with open(EVAL_HISTORY_FILE, encoding="utf-8") as f:
            prior = json.load(f)
    # "Best" must mean lowest CER, not most recently scored. This used to
    # overwrite checkpoints/best/ on every eval, so a worse round would clobber
    # a better model — fatal once a loop is running unattended.
    this_sig = (split, len(val_ds), beams, 448)
    comparable = [e["cer"] for e in prior
                  if e.get("cer") is not None and eval_signature(e) == this_sig]
    prior_best = min(comparable, default=float("inf"))
    is_best = cer < prior_best and "--no-promote" not in sys.argv

    history = (record_eval_history(epoch, Path(ckpt).name, cer, wer, loss,
                                   split, len(val_ds), beams) if record else prior)
    same_setup = [e for e in history if eval_signature(e) == this_sig and e.get("cer") is not None]
    if len(same_setup) > 1:
        first = same_setup[0]
        trend_line = (f"First comparable run (epoch {first['epoch']}) was {first['cer']}% "
                      f"— change: {round(first['cer'] - cer, 2)} pts")
    else:
        trend_line = "First run at this split/beams/length setting — no comparable history yet."

    report = (
        f"EPOCH {epoch} EVAL COMPLETE - KASEKOR ASR v0.0\n"
        f"{'='*30}\n\n"
        f"CER       : {cer}% (lower = better)\n"
        f"WER       : {wer}%\n"
        f"Eval loss : {loss}\n\n"
        f"Checkpoint: {Path(ckpt).name}\n"
        f"Split     : {split} ({len(val_ds)} samples, beams={beams})\n"
        f"Eval time : {mins} min\n"
        f"{'='*30}\n"
        f"{trend_line}\n"
        + ("NEW BEST — promoted to checkpoints/best/\n\n" if is_best
           else f"Not promoted (best at this setting: "
                f"{'none yet' if prior_best == float('inf') else str(prior_best) + '%'})\n\n")
        +
        f"APP INTEGRATION\n"
        f"{'='*30}\n"
        f"Model path : checkpoints/best/\n"
        f"Sample rate: 16000 Hz (REQUIRED)\n"
        f"Channels   : Mono, float32, max 30s\n"
        f"Language   : km (Khmer)\n\n"
        f"from transformers import WhisperProcessor, WhisperForConditionalGeneration\n"
        f"processor = WhisperProcessor.from_pretrained('checkpoints/best')\n"
        f"model = WhisperForConditionalGeneration.from_pretrained('checkpoints/best')\n"
        f"ids = model.generate(inputs, language='km', task='transcribe', num_beams=5)\n"
        f"text = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]"
    )

    print(report)
    _tg(report)

    # Only promote to best/ when this really is the lowest CER seen.
    best_dir = OUTPUT_DIR / "best"
    if is_best:
        trainer.save_model(str(best_dir))
        from transformers import WhisperProcessor as _P
        _P.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK).save_pretrained(str(best_dir))
        prev = "none yet" if prior_best == float("inf") else f"{prior_best}%"
        print(f"NEW BEST ({cer}%, previous {prev}) — model + processor saved to {best_dir}/")
    else:
        why = "--no-promote set" if "--no-promote" in sys.argv else f"does not beat {prior_best}%"
        print(f"Not promoted: {cer}% — {why}. {best_dir}/ left untouched.")

    if json_out:
        out = Path(json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": str(ckpt), "epoch": epoch, "split": split,
                "beams": beams, "samples": len(val_ds),
                "cer": cer, "wer": wer, "eval_loss": loss,
                "minutes": mins, "is_best": is_best, "prior_best": prior_best,
            }, f, indent=2)
        print(f"Result JSON -> {out}")


if __name__ == "__main__":
    main()
