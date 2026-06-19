"""
KASEKOR v0.0 — Khmer ASR inference script.

Usage:
  python scripts/transcribe.py audio.wav
  python scripts/transcribe.py audio.wav --ref "ខ្ញុំទៅផ្សារ"
  python scripts/transcribe.py --mic
  python scripts/transcribe.py --mic --mic-seconds 15
  python scripts/transcribe.py audio/ --batch        # transcribe a folder
"""

import sys
import re
import argparse
import unicodedata
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "checkpoints" / "best"
SR = 16_000
MAX_SAMPLES = SR * 30   # Whisper's 30 s receptive field


def normalize_khmer(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


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


def record_mic(seconds: int = 10) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)
    print(f"Recording {seconds}s from microphone… (speak now)")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    print("Recording done.")
    return audio.flatten()


def transcribe_audio(audio: np.ndarray, processor, model, device: str) -> str:
    inputs = processor(audio, sampling_rate=SR, return_tensors="pt").input_features
    inputs = inputs.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs,
            language="km",
            task="transcribe",
            num_beams=5,
            max_new_tokens=225,
        )
    text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return normalize_khmer(text)


def compute_cer(prediction: str, reference: str) -> float:
    import evaluate
    metric = evaluate.load("cer")
    return metric.compute(predictions=[prediction], references=[reference])


def main():
    parser = argparse.ArgumentParser(
        description="KASEKOR Khmer ASR — transcribe audio files or microphone input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio", nargs="?", help="Path to audio file or folder (use --batch for folder)")
    parser.add_argument("--ref",         help="Reference transcript — enables CER output")
    parser.add_argument("--mic",         action="store_true", help="Record from microphone instead of a file")
    parser.add_argument("--mic-seconds", type=int, default=10, metavar="N", help="Mic recording duration in seconds (default 10)")
    parser.add_argument("--batch",       action="store_true", help="Transcribe all .wav/.flac/.mp3 files in a folder")
    parser.add_argument("--model",       default=str(DEFAULT_MODEL), metavar="DIR", help=f"Model directory (default: checkpoints/best)")
    parser.add_argument("--output",      metavar="FILE", help="Write results to a TSV file (batch mode)")
    args = parser.parse_args()

    if not args.audio and not args.mic:
        parser.print_help()
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Training may not be complete yet. Run train.py first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path} [{device}]…")
    processor = WhisperProcessor.from_pretrained(str(model_path))
    model = WhisperForConditionalGeneration.from_pretrained(str(model_path)).to(device)
    model.eval()
    print("Model loaded.\n")

    # ── Microphone mode ───────────────────────────────────────────────────────
    if args.mic:
        audio = record_mic(args.mic_seconds)
        result = transcribe_audio(audio, processor, model, device)
        print(f"Transcript : {result}")
        if args.ref:
            cer = compute_cer(result, normalize_khmer(args.ref))
            print(f"Reference  : {normalize_khmer(args.ref)}")
            print(f"CER        : {cer:.4f}  ({cer*100:.2f}%)")
        return

    # ── Batch folder mode ────────────────────────────────────────────────────
    if args.batch:
        folder = Path(args.audio)
        if not folder.is_dir():
            print(f"Not a directory: {folder}")
            sys.exit(1)
        audio_files = sorted(
            list(folder.glob("*.wav")) +
            list(folder.glob("*.flac")) +
            list(folder.glob("*.mp3"))
        )
        if not audio_files:
            print(f"No audio files found in {folder}")
            sys.exit(1)
        print(f"Found {len(audio_files)} audio file(s).\n")

        rows = []
        for i, path in enumerate(audio_files, 1):
            audio = load_audio(str(path))
            result = transcribe_audio(audio, processor, model, device)
            duration = len(audio) / SR
            print(f"[{i}/{len(audio_files)}] {path.name} ({duration:.1f}s)")
            print(f"  → {result}\n")
            rows.append((path.name, duration, result))

        if args.output:
            out_path = Path(args.output)
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                f.write("filename\tduration_s\ttranscript\n")
                for name, dur, text in rows:
                    f.write(f"{name}\t{dur:.2f}\t{text}\n")
            print(f"Results saved to {out_path}")
        return

    # ── Single file mode ─────────────────────────────────────────────────────
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    print(f"Audio      : {audio_path}")
    audio = load_audio(str(audio_path))
    print(f"Duration   : {len(audio)/SR:.1f}s")
    print("Transcribing…")

    result = transcribe_audio(audio, processor, model, device)
    print(f"\nTranscript : {result}")

    if args.ref:
        ref = normalize_khmer(args.ref)
        cer = compute_cer(result, ref)
        print(f"Reference  : {ref}")
        print(f"CER        : {cer:.4f}  ({cer*100:.2f}%)")


if __name__ == "__main__":
    main()
