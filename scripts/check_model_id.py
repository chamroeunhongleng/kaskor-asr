"""Fail if the released model id shipped in the CLI stops resolving on the Hub.

`pip install kaskor-asr` then `kaskor audio.wav` downloads whatever
`kaskor.cli.DEFAULT_MODEL` points at. If that id is renamed, made private or
deleted, every install keeps working right up until the first transcription and
then fails on a download the user cannot fix. This check is the guard: it reads
the id out of the shipped source and asks the Hub whether it still resolves.

The id is parsed with `ast` rather than imported, so the check needs neither
torch nor transformers and runs in a couple of seconds.
"""

import ast
import sys
import urllib.error
import urllib.request
from pathlib import Path

CLI_SOURCE = Path(__file__).resolve().parent.parent / "kaskor" / "cli.py"
HUB_API = "https://huggingface.co/api/models/{model}"
TIMEOUT_S = 30


def default_model_id(source: Path) -> str:
    """Read DEFAULT_MODEL out of cli.py without importing it."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "DEFAULT_MODEL":
                return ast.literal_eval(node.value)
    raise SystemExit(f"DEFAULT_MODEL is not defined in {source}")


def hub_status(model: str) -> int:
    try:
        with urllib.request.urlopen(HUB_API.format(model=model), timeout=TIMEOUT_S) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        # A missing, renamed or private repo answers 401 here, not 404.
        return exc.code


def main() -> int:
    model = default_model_id(CLI_SOURCE)
    print(f"Shipped model id : {model}")

    status = hub_status(model)
    if status != 200:
        print(
            f"FAIL: the Hub returned HTTP {status} for '{model}'.\n"
            f"      Every `pip install kaskor-asr` user would hit a download failure\n"
            f"      on their first transcription. Fix DEFAULT_MODEL in kaskor/cli.py\n"
            f"      or restore the model repository.",
            file=sys.stderr,
        )
        return 1

    print("Hub status       : 200 — the install path still resolves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
