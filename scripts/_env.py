"""Tiny .env loader so secrets (Telegram bot token/chat) aren't hardcoded in scripts."""
import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
_loaded = False


def load_env():
    global _loaded
    if _loaded or not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
    _loaded = True


def tg_config():
    load_env()
    return os.environ.get("KASEKOR_TG_TOKEN", ""), os.environ.get("KASEKOR_TG_CHAT", "")
