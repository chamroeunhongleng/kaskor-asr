"""
Step 10: Export the best checkpoint as a self-contained model package.
Copies weights + processor/tokenizer into a clean export folder,
and writes a quick inference test to confirm everything works.
Output: kaskor-asr-v0.0-export/
"""

import torch
import librosa
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration

CHECKPOINT_DIR = Path("checkpoints/best")
EXPORT_DIR     = Path("kaskor-asr-v0.0-export")
LANGUAGE       = "km"
TASK           = "transcribe"
SR             = 16000


def export():
    if not CHECKPOINT_DIR.exists():
        print(f"ERROR: {CHECKPOINT_DIR} not found. Run train.py first.")
        return

    print(f"Loading model from {CHECKPOINT_DIR} ...")
    processor = WhisperProcessor.from_pretrained(str(CHECKPOINT_DIR), language=LANGUAGE, task=TASK)
    model     = WhisperForConditionalGeneration.from_pretrained(str(CHECKPOINT_DIR))
    model.eval()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(str(EXPORT_DIR))
    model.save_pretrained(str(EXPORT_DIR))
    print(f"Exported to {EXPORT_DIR}/")

    # Write inference example script
    inference_script = EXPORT_DIR / "transcribe.py"
    inference_script.write_text(
        '''\
"""Inference example: transcribe a WAV file with the exported model."""
import sys
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_DIR = "."   # path to this export folder
SR        = 16000

def transcribe(audio_path: str) -> str:
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    model     = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.eval()

    audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    inputs   = processor(audio, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(**inputs)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
    print(transcribe(path))
''',
        encoding="utf-8",
    )

    # Write preprocessing guide
    guide = EXPORT_DIR / "PREPROCESSING.md"
    guide.write_text(
        """\
# Preprocessing guide for kaskor-asr-v0.0

## Input requirements
- Sample rate : 16 000 Hz
- Channels    : mono (1 channel)
- Format      : WAV (PCM 16-bit recommended)
- Max duration: 30 seconds per clip

## Quick preprocessing with librosa
```python
import librosa, soundfile as sf
audio, _ = librosa.load("input.mp3", sr=16000, mono=True)
sf.write("output.wav", audio, 16000, subtype="PCM_16")
```

## Language
Khmer (km) — trained on DDD-Cambodia/khmer-speech-dataset.
""",
        encoding="utf-8",
    )

    print("Written: transcribe.py, PREPROCESSING.md")
    print("Export complete.")


if __name__ == "__main__":
    export()
