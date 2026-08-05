"""
Step 9: Fine-tune openai/whisper-small for Khmer speech-to-text.

Key changes from the original:
  * Primary metric is now CER, not WER. Khmer has no inter-word spaces, so
    word-level WER is close to meaningless; CER is the standard ASR metric for
    Khmer / Thai / Lao / CJK. WER is still reported as a secondary number.
  * Fixed start-of-transcript stripping: compare against decoder_start_token_id
    (<|startoftranscript|>), NOT bos_token_id. bos_token_id is <|endoftext|>
    (50257) while the prefix the tokenizer prepends is 50258, so the old check
    was always False -> the leading SOT was never removed -> the model was
    trained to predict a duplicated prefix.
  * Language/task set on generation_config (config.forced_decoder_ids is
    deprecated and warns on recent transformers).
  * Audio sample rate is verified and resampled if it isn't 16 kHz, instead of
    being silently ignored.
  * Transcripts are NFC-normalised; empty / missing-file rows are dropped.
  * use_cache disabled (required with gradient checkpointing).
  * Saves the full processor next to the model + seed / save_total_limit /
    early stopping for cleaner, reproducible runs.
"""

import csv
import os
import re
import secrets
import sys
import json
import ctypes
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

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "openai/whisper-small"
LANGUAGE     = "km"
TASK         = "transcribe"
TRAIN_CSV    = Path("data/train.csv")
VAL_CSV      = Path("data/val.csv")
OUTPUT_DIR   = Path("checkpoints")
CACHE_DIR    = Path(__file__).resolve().parent.parent / "datasets" / "mel_cache"
BATCH_SIZE   = 8            # halved to free VRAM — effective batch stays 16 via accumulation
GRAD_ACCUM   = 2            # 2 × 8 = effective batch 16
LR           = 1e-5
WEIGHT_DECAY = 0.01          # L2 regularization — was 0.0 (HF default), contributed to overfitting
NUM_EPOCHS   = 5
EVAL_HISTORY_FILE   = Path("eval_history.json")
PLATEAU_MIN_DELTA   = 1.0    # require >=1.0 absolute CER point improvement over previous epoch
SR           = 16000
MAX_SAMPLES  = SR * 30      # Whisper's fixed 30s receptive field
BF16         = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
FP16         = False        # prefer bf16 on Blackwell
NUM_WORKERS  = 2            # 2 workers avoids system-Python spawn issues on Windows
SEED         = 42
EPOCHS_PER_RUN = 1          # train one epoch per run; resume automatically next session

# Waveform augmentation — fixes overfitting seen in epoch 5 (train_loss 0.012,
# CER barely moved) by giving the model varied versions of the same audio
# instead of the identical sample every epoch. Only ever applied to training
# data, never to eval/val. Bypasses the mel cache (which holds clean features)
# so augmented samples are computed on the fly from the raw waveform.
AUGMENT_PROB        = 0.4    # fraction of train samples augmented per epoch (rest use cached clean mel)
AUG_SPEED_PROB      = 0.5
AUG_SPEED_RANGE     = (0.9, 1.1)
AUG_GAIN_PROB       = 0.8
AUG_GAIN_DB_RANGE   = (-6.0, 6.0)
AUG_NOISE_PROB      = 0.4
AUG_NOISE_SNR_DB    = (15.0, 30.0)
# ──────────────────────────────────────────────────────────────────────────────


def find_latest_checkpoint(output_dir: Path):
    """Return (checkpoint_path, completed_epochs) or (None, 0) if none found."""
    ckpts = sorted(output_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    if not ckpts:
        return None, 0
    latest = ckpts[-1]
    state_file = latest / "trainer_state.json"
    completed = 0
    if state_file.exists():
        with open(state_file, encoding="utf-8-sig") as f:
            completed = int(json.load(f).get("epoch", 0))
    return str(latest), completed


def load_eval_history() -> list[dict]:
    if not EVAL_HISTORY_FILE.exists():
        return []
    with open(EVAL_HISTORY_FILE, encoding="utf-8") as f:
        return json.load(f)


def check_plateau(history: list[dict], min_delta: float) -> str | None:
    """Return a warning message if CER has plateaued, else None."""
    if len(history) < 2:
        return None
    prev_cer, last_cer = history[-2]["cer"], history[-1]["cer"]
    improvement = prev_cer - last_cer
    if improvement < min_delta:
        return (
            f"CER plateau detected: epoch {history[-2]['epoch']} -> {history[-1]['epoch']} "
            f"only improved {improvement:.2f} points ({prev_cer}% -> {last_cer}%), "
            f"below the {min_delta} point threshold. Training another epoch on the same "
            f"data is unlikely to help (overfitting risk) — fix data diversity first.\n"
            f"Pass --force to train anyway."
        )
    return None


def normalize_khmer(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    try:
        import soxr
        return soxr.resample(audio, orig_sr, target_sr)
    except ImportError:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def _load_raw_audio(audio_path: str) -> np.ndarray:
    """Decode + mono-mix + resample to SR + truncate to MAX_SAMPLES."""
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = _resample(audio, sr, SR).astype("float32")
    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    return audio


def augment_waveform(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light, cheap waveform augmentation: speed perturb, gain jitter, additive noise.

    Each transform is independently coin-flipped so samples get a varied mix
    rather than always all three. Speed perturbation can shift sample count,
    so MAX_SAMPLES is re-applied at the end.
    """
    if rng.random() < AUG_SPEED_PROB:
        factor = rng.uniform(*AUG_SPEED_RANGE)
        target_sr = max(1, int(round(SR / factor)))
        audio = _resample(audio, SR, target_sr).astype("float32")

    if rng.random() < AUG_GAIN_PROB:
        gain_db = rng.uniform(*AUG_GAIN_DB_RANGE)
        audio = audio * (10.0 ** (gain_db / 20.0))

    if rng.random() < AUG_NOISE_PROB:
        snr_db = rng.uniform(*AUG_NOISE_SNR_DB)
        sig_power = np.mean(audio ** 2) + 1e-10
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape)
        audio = audio + noise

    audio = np.clip(audio, -1.0, 1.0).astype("float32")
    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    return audio


class KhmerASRDataset(Dataset):
    def __init__(self, csv_path: Path, processor: WhisperProcessor, augment: bool = False):
        with open(csv_path, encoding="utf-8", newline="") as f:
            raw = list(csv.DictReader(f))

        self.rows = []
        skipped = 0
        for row in raw:
            transcript = normalize_khmer(row.get("transcript", ""))
            if not transcript or not Path(row["audio_path"]).is_file():
                skipped += 1
                continue
            row["transcript"] = transcript
            self.rows.append(row)
        if skipped:
            print(f"{csv_path.name}: skipped {skipped} rows (empty transcript or missing file).")

        self.processor = processor
        self.augment = augment
        # Per-process+instance salt so augmentation choices differ across
        # dataloader worker processes instead of all replaying the same
        # sequence (numpy RNG state is duplicated on fork otherwise).
        self._salt = secrets.randbits(32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        rng = np.random.default_rng(hash((idx, os.getpid(), self._salt)) % (2 ** 32))
        do_augment = self.augment and rng.random() < AUGMENT_PROB

        if do_augment:
            # bypass the clean mel cache — augmentation needs the raw waveform
            audio = _load_raw_audio(row["audio_path"])
            audio = augment_waveform(audio, rng)
            input_features = self.processor(
                audio, sampling_rate=SR, return_tensors="pt"
            ).input_features[0]
        else:
            cache_path = CACHE_DIR / (Path(row["audio_path"]).stem + ".npy")
            if cache_path.exists():
                # fast path: load pre-computed float16 mel, cast to float32 for training
                arr = np.load(str(cache_path)).astype(np.float32)
                input_features = torch.from_numpy(arr)
            else:
                # slow path: decode audio + compute mel on the fly
                audio = _load_raw_audio(row["audio_path"])
                input_features = self.processor(
                    audio, sampling_rate=SR, return_tensors="pt"
                ).input_features[0]

        labels = self.processor.tokenizer(
            row["transcript"],
            return_tensors="pt",
            max_length=448,
            truncation=True,
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
        # Strip the leading <|startoftranscript|> token — the model adds it back
        # as decoder_start_token_id during the forward pass.
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


# ── Night mode helpers ────────────────────────────────────────────────────────

from _env import tg_config
TG_TOKEN, TG_CHAT = tg_config()

def _tg(text):
    try:
        import requests as _r
        _r.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT, "text": text}, timeout=10)
    except Exception as e:
        print(f"[Telegram error] {e}")

def _prevent_sleep():
    """Keep PC awake even when screen locks / display turns off."""
    ES = 0x80000000 | 0x00000001 | 0x00000040   # CONTINUOUS | SYSTEM_REQUIRED | AWAYMODE
    ctypes.windll.kernel32.SetThreadExecutionState(ES)
    print("[Night mode] Sleep prevention ON — screen may go dark but process keeps running.")

def _allow_sleep():
    """Restore normal Windows sleep behaviour."""
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS only
    print("[Night mode] Sleep prevention OFF — normal power settings restored.")

# ── Main ──────────────────────────────────────────────────────────────────────

def _flag_value(name, cast, default):
    """Read `--name VALUE` out of argv without disturbing the existing flag style."""
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def main():
    # ── Parse --night N ───────────────────────────────────────────────────────
    night_steps = None
    if "--night" in sys.argv:
        night_steps = int(sys.argv[sys.argv.index("--night") + 1])
    no_eval = "--no-eval" in sys.argv
    force   = "--force" in sys.argv

    # ── Overrides so train_loop.py can drive a round without editing this file ──
    # The loop needs to raise the epoch ceiling (5 epochs are already done) and
    # step the learning rate / augmentation down a ladder between rounds.
    global NUM_EPOCHS, EPOCHS_PER_RUN, LR, AUGMENT_PROB, WEIGHT_DECAY
    NUM_EPOCHS     = _flag_value("--max-epochs",     int,   NUM_EPOCHS)
    EPOCHS_PER_RUN = _flag_value("--epochs-per-run", int,   EPOCHS_PER_RUN)
    LR             = _flag_value("--lr",             float, LR)
    AUGMENT_PROB   = _flag_value("--augment-prob",   float, AUGMENT_PROB)
    WEIGHT_DECAY   = _flag_value("--weight-decay",   float, WEIGHT_DECAY)

    plateau_msg = check_plateau(load_eval_history(), PLATEAU_MIN_DELTA)
    if plateau_msg and not force:
        print(plateau_msg)
        _tg(f"Training BLOCKED — {plateau_msg}")
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Round mode (used by train_loop.py) ────────────────────────────────────
    # A resumed run restores optimizer AND scheduler state, which silently
    # overrides --lr: the param-group LR comes back from optimizer.pt and the
    # schedule picks up where it left off. That makes an LR ladder a no-op.
    # Round mode therefore starts each round as its own short fine-tune —
    # weights from --init-from, fresh optimizer, fresh warmup, own output dir so
    # restarted step numbering can't collide with the existing checkpoint-NNNNN.
    init_from      = _flag_value("--init-from", str, None)
    output_subdir  = _flag_value("--output-subdir", str, None)
    round_mode     = init_from is not None or output_subdir is not None

    global OUTPUT_DIR
    if output_subdir:
        OUTPUT_DIR = OUTPUT_DIR / output_subdir
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if round_mode:
        resume_ckpt = None                       # fresh optimizer/scheduler
        resume_step = 0
        completed_epochs = 0
        target_epochs = EPOCHS_PER_RUN           # relative, not absolute
        model_init = init_from or find_latest_checkpoint(Path("checkpoints"))[0] or BASE_MODEL
        print(f"[round mode] init weights from {model_init}")
        print(f"[round mode] fresh optimizer, lr={LR:g}, aug={AUGMENT_PROB}, wd={WEIGHT_DECAY}, "
              f"{target_epochs} epoch(s) -> {OUTPUT_DIR}")
    else:
        resume_ckpt, completed_epochs = find_latest_checkpoint(OUTPUT_DIR)
        resume_step = int(resume_ckpt.split("-")[-1]) if resume_ckpt else 0
        target_epochs = completed_epochs + EPOCHS_PER_RUN

        if completed_epochs >= NUM_EPOCHS:
            print(f"Already finished {completed_epochs}/{NUM_EPOCHS} epochs. Training complete!")
            return

        target_epochs = min(target_epochs, NUM_EPOCHS)
        model_init = resume_ckpt if resume_ckpt else BASE_MODEL
        if resume_ckpt:
            print(f"Resuming from {resume_ckpt} (epoch {completed_epochs} done → training epoch {completed_epochs + 1})")
        else:
            print(f"Starting fresh — training epoch 1 of {NUM_EPOCHS}")

    if night_steps:
        _prevent_sleep()
        stop_at = resume_step + night_steps
        print(f"[Night mode] {night_steps} steps tonight → stop at global step {stop_at}. PC will shutdown after.")
        _tg(f"Epoch {completed_epochs + 1} training STARTED\n\nSteps tonight : {night_steps}\nFrom step     : {resume_step}\nStop at       : {stop_at}\nEst. time     : ~{round(night_steps/1.675/3600,1)}h\n\nScreen may go dark — process keeps running.")

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)
    # Load from checkpoint if resuming — this gets epoch 1 weights into the model
    # BEFORE torch.compile wraps it. compile adds _orig_mod. prefix to state keys,
    # so HF Trainer's resume can't remap checkpoint keys; we pre-load weights here.
    model = WhisperForConditionalGeneration.from_pretrained(model_init)

    model.generation_config.language           = LANGUAGE
    model.generation_config.task               = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids            = None
    model.config.suppress_tokens               = []
    model.config.use_cache                     = False
    model.config.apply_spec_augment            = True
    model.gradient_checkpointing_enable()
    decoder_start_token_id = model.config.decoder_start_token_id

    train_ds = KhmerASRDataset(TRAIN_CSV, processor, augment=True)
    val_ds   = KhmerASRDataset(VAL_CSV,   processor, augment=False)
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
    )

    # night mode: stop at fixed step, skip eval (saves ~2h)
    # no-eval mode: run full epoch but defer eval to run_eval.py after a rest
    max_steps_val  = (resume_step + night_steps) if night_steps else -1
    # --max-steps is the same hard stop as --night but without sleep-prevention
    # or the shutdown at the end, so a few steps can be run as a smoke test.
    test_steps     = _flag_value("--max-steps", int, None)
    if test_steps:
        max_steps_val = resume_step + test_steps
    eval_strat     = "no" if (night_steps or no_eval) else "epoch"

    args = Seq2SeqTrainingArguments(
        output_dir                  = str(OUTPUT_DIR),
        num_train_epochs            = target_epochs,
        max_steps                   = max_steps_val,   # -1 = use epochs; N = hard stop
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        weight_decay                = WEIGHT_DECAY,
        warmup_ratio                = 0.1,
        fp16                        = FP16,
        bf16                        = BF16,
        optim                       = "adamw_torch_fused",
        eval_strategy               = eval_strat,
        save_strategy               = "steps",
        save_steps                  = 500,
        # Each checkpoint is 2.8 GB (1.9 GB of that is optimizer state). A round
        # is a self-contained fine-tune whose only lasting artifact is its final
        # weights, so keeping 3 per round would cost ~84 GB across a 10-round run.
        save_total_limit            = 1 if round_mode else 3,
        logging_steps               = 25,
        predict_with_generate       = True,
        # See run_eval.py — 225 truncated half the eval set at ~102 Khmer chars,
        # so in-training CER was measuring truncation rather than the model.
        generation_max_length       = 448,
        generation_num_beams        = 2,
        load_best_model_at_end      = False,
        metric_for_best_model       = "cer",
        greater_is_better           = False,
        report_to                   = "none",
        remove_unused_columns       = False,   # torch.compile hides forward signature → RemoveColumnsCollator strips input_features
        dataloader_num_workers      = NUM_WORKERS,
        dataloader_pin_memory       = True,
        seed                        = SEED,
    )

    trainer = Seq2SeqTrainer(
        model            = model,
        args             = args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        data_collator    = collator,
        compute_metrics  = lambda pred: compute_metrics(pred, processor),
        callbacks        = [],
    )

    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)} samples.")
    print(f"Batch {BATCH_SIZE}x{GRAD_ACCUM} | bf16={BF16} | workers={NUM_WORKERS} | metric=CER")
    print(f"Waveform augmentation: {AUGMENT_PROB:.0%} of train samples (speed/gain/noise jitter)")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── Explicit final save (ensures step is checkpointed even if not at save_steps boundary) ──
    final_ckpts  = sorted(OUTPUT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    final_step   = int(final_ckpts[-1].name.split("-")[-1]) if final_ckpts else resume_step
    final_ckpt_dir = OUTPUT_DIR / f"checkpoint-{final_step}"
    trainer.save_model(str(final_ckpt_dir))
    trainer.save_state()
    print(f"Checkpoint saved → {final_ckpt_dir}")

    if round_mode:
        # Round checkpoints get loaded standalone by eval/diagnostics, so keep a
        # processor beside the weights. The marker line is what train_loop.py
        # parses to find this round's output.
        processor.save_pretrained(str(final_ckpt_dir))
        print(f"ROUND_CHECKPOINT={final_ckpt_dir}")

    if night_steps:
        # Last logged loss
        last_logs  = [e for e in trainer.state.log_history if "loss" in e]
        last_loss  = round(last_logs[-1]["loss"], 4) if last_logs else "N/A"

        _allow_sleep()   # restore normal power settings before shutdown
        _tg(f"Epoch {completed_epochs + 1} Training COMPLETE — PC shutting down\n\nSteps trained  : {night_steps}\nFinal step     : {final_step}\nLast loss      : {last_loss}\nCheckpoint     : checkpoint-{final_step}\n\nShutting down in 60s...")
        print("[Night mode] Shutdown in 60 seconds.")
        import subprocess
        subprocess.run("shutdown /s /t 60", shell=True)
    elif no_eval:
        # Defer eval to run_eval.py — monitor will trigger it after a rest
        last_logs = [e for e in trainer.state.log_history if "loss" in e]
        last_loss = round(last_logs[-1]["loss"], 4) if last_logs else "N/A"
        _tg(f"Epoch {completed_epochs + 1} Training COMPLETE (eval deferred)\n\nFinal step : {final_step}\nLast loss  : {last_loss}\nCheckpoint : checkpoint-{final_step}\n\nResting 30 min, then eval starts.")
        print(f"Training complete. Eval deferred. Checkpoint: checkpoint-{final_step}")
    else:
        best_dir = OUTPUT_DIR / "best"
        trainer.save_model(str(best_dir))
        processor.save_pretrained(str(best_dir))
        print(f"Training complete. Best model + processor saved to {best_dir}/")


if __name__ == "__main__":
    main()
