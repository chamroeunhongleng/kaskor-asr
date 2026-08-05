"""
One-time step: publish the exported model to the Hugging Face Hub so the
`kaskor` CLI can download it automatically for anyone who pip-installs this
repo, instead of requiring a local checkpoints/ directory.

Prerequisites:
  pip install huggingface_hub
  huggingface-cli login        # paste a token with "write" access, from
                                # https://huggingface.co/settings/tokens

Usage:
  python scripts/push_to_hub.py                          # pushes checkpoints/best
  python scripts/push_to_hub.py --dir kaskor-asr-v0.0-export
  python scripts/push_to_hub.py --repo your-username/kaskor-asr
  python scripts/push_to_hub.py --private
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, upload_folder

DEFAULT_REPO = "Hongleng/kasekor-asr-v0.0"  # "chamroeunhongleng/..." does not exist on the Hub (401)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="checkpoints/best", help="Local model directory to upload (default: checkpoints/best)")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Target Hub repo id (default: {DEFAULT_REPO})")
    parser.add_argument("--private", action="store_true", help="Create/push as a private Hub repo")
    args = parser.parse_args()

    model_dir = Path(args.dir)
    if not model_dir.exists():
        print(f"Not found: {model_dir}")
        print("Run save_model.py first, or point --dir at your exported model folder.")
        return

    api = HfApi()
    api.create_repo(args.repo, exist_ok=True, private=args.private)
    print(f"Uploading {model_dir} -> {args.repo} ...")
    upload_folder(repo_id=args.repo, folder_path=str(model_dir))
    print(f"Done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
