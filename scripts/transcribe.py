"""
Thin wrapper around kaskor.cli for running from a repo clone without
installing the package. If you `pip install` this repo, use the `kaskor`
command instead — see kaskor/cli.py for the actual implementation.

Usage:
  python scripts/transcribe.py audio.wav
  python scripts/transcribe.py audio.wav --ref "ខ្ញុំទៅផ្សារ"
  python scripts/transcribe.py --mic
  python scripts/transcribe.py audio/ --batch --output results.tsv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kaskor.cli import main

if __name__ == "__main__":
    main()
