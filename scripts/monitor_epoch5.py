"""
Monitor for epoch 5 training.
Sends Telegram updates every 30 min; final CER result when done. No shutdown.

Usage:
  python scripts/monitor_epoch5.py <PID>
"""

import re
import sys
import time
import subprocess
from datetime import datetime, timedelta

from _env import tg_config
TG_TOKEN, TG_CHAT = tg_config()
ERR_FILE  = r"C:\Users\Admin\kaskor-asr-v0.0\train_err.txt"

EPOCH       = 5
TOTAL_STEPS = 5410          # steps in one epoch (86550 samples / 8 batch / 2 accum)
START_STEP  = 21640
END_STEP    = START_STEP + TOTAL_STEPS  # 27050
UPDATE_INTERVAL = 1800      # seconds between Telegram updates (30 min)


def tg(text: str):
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text},
            timeout=10,
        )
    except Exception as e:
        print(f"[TG error] {e}")


def is_running(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True,
    )
    return str(pid) in result.stdout


def read_err() -> list[str]:
    try:
        with open(ERR_FILE, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except Exception:
        return []


def parse_current_step(lines: list[str]) -> int | None:
    """Extract the latest global step from tqdm progress lines."""
    pattern = re.compile(r"\|\s*(\d+)/27050")
    for line in reversed(lines):
        m = pattern.search(line)
        if m:
            return int(m.group(1))
    return None


def parse_last_loss(lines: list[str]) -> str:
    for line in reversed(lines):
        if "'loss'" in line:
            m = re.search(r"'loss':\s*([\d.]+)", line)
            if m:
                return m.group(1)
    return "N/A"


def parse_eval_results(lines: list[str]) -> dict:
    """Look for the final eval metrics in the log."""
    results = {}
    for line in reversed(lines):
        if "eval_cer" in line:
            m = re.search(r"'eval_cer':\s*([\d.]+)", line)
            if m:
                results["cer"] = float(m.group(1))
        if "eval_wer" in line:
            m = re.search(r"'eval_wer':\s*([\d.]+)", line)
            if m:
                results["wer"] = float(m.group(1))
        if "eval_loss" in line:
            m = re.search(r"'eval_loss':\s*([\d.]+)", line)
            if m:
                results["loss"] = float(m.group(1))
        if results.get("cer") and results.get("wer") and results.get("loss"):
            break
    return results


def progress_bar(pct: float, width: int = 20) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def main():
    if len(sys.argv) < 2:
        print("Usage: python monitor_epoch5.py <PID>")
        sys.exit(1)

    pid = int(sys.argv[1])
    start_time = datetime.now()
    print(f"Monitoring PID {pid} for epoch {EPOCH} training.")

    tg(
        f"Epoch {EPOCH} training STARTED\n"
        f"Steps : {TOTAL_STEPS} (step {START_STEP} → {END_STEP})\n"
        f"Est.  : ~3h train + ~2h eval\n"
        f"Updates every 30 min."
    )

    last_update = time.time()

    while True:
        time.sleep(60)

        if not is_running(pid):
            # Training finished — grab last loss
            lines = read_err()
            last_loss = parse_last_loss(lines)
            elapsed = datetime.now() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed.total_seconds())))

            REST_MINUTES = 30
            VENV_PYTHON = r"C:\Users\Admin\kaskor-asr-v0.0\venv\Scripts\python.exe"
            WORK_DIR    = r"C:\Users\Admin\kaskor-asr-v0.0"
            EVAL_LOG    = r"C:\Users\Admin\kaskor-asr-v0.0\eval_log.txt"
            EVAL_ERR    = r"C:\Users\Admin\kaskor-asr-v0.0\eval_err.txt"

            tg(
                f"Epoch {EPOCH} Training COMPLETE\n"
                f"{'='*30}\n"
                f"Train loss : {last_loss}\n"
                f"Time       : {elapsed_str}\n"
                f"{'='*30}\n"
                f"Resting {REST_MINUTES} min, then eval starts automatically."
            )
            print(f"[{datetime.now().strftime('%H:%M')}] Training done. Resting {REST_MINUTES} min...")

            # Let the PC cool down before eval
            time.sleep(REST_MINUTES * 60)

            # Launch eval as a separate process
            print(f"[{datetime.now().strftime('%H:%M')}] Rest done. Starting eval...")
            tg(f"Rest done. Eval STARTING now (~2h)...")
            eval_proc = subprocess.Popen(
                [VENV_PYTHON, "scripts/run_eval.py"],
                cwd=WORK_DIR,
                stdout=open(EVAL_LOG, "w"),
                stderr=open(EVAL_ERR, "w"),
            )

            # Wait for eval to finish
            eval_proc.wait()

            # Restore normal sleep settings before shutdown
            subprocess.run("powercfg /change standby-timeout-ac 15", shell=True)

            print(f"[{datetime.now().strftime('%H:%M')}] Eval done. Shutting down.")
            tg(f"Eval complete. PC shutting down now. Good rest!")
            subprocess.run("shutdown /s /t 0", shell=True)
            break

        # Periodic update every 30 min
        if time.time() - last_update >= UPDATE_INTERVAL:
            lines = read_err()
            current_step = parse_current_step(lines) or START_STEP
            steps_done = max(0, current_step - START_STEP)
            pct = min(100.0, steps_done / TOTAL_STEPS * 100)
            last_loss = parse_last_loss(lines)
            elapsed = datetime.now() - start_time

            if steps_done > 0:
                sec_per_step = elapsed.total_seconds() / steps_done
                remaining_steps = TOTAL_STEPS - steps_done
                eta_sec = remaining_steps * sec_per_step
                eta_str = datetime.fromtimestamp(time.time() + eta_sec).strftime("%I:%M %p")
                remaining_str = str(timedelta(seconds=int(eta_sec)))
            else:
                eta_str = "calculating..."
                remaining_str = "—"

            msg = (
                f"Epoch {EPOCH} update\n"
                f"[{progress_bar(pct)}] {pct:.1f}%\n"
                f"Step      : {current_step}/{END_STEP}\n"
                f"Loss      : {last_loss}\n"
                f"Remaining : {remaining_str} (ETA {eta_str})\n"
            )
            tg(msg)
            print(f"[{datetime.now().strftime('%H:%M')}] Update sent: {pct:.1f}%")
            last_update = time.time()


if __name__ == "__main__":
    main()
