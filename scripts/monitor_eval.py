"""
Monitors eval process. When done:
  1. Reports CER/WER to Telegram bot
  2. Resets lid-close setting back to Sleep
  3. Shuts down PC after 60 seconds
"""
import time
import subprocess
import requests
from datetime import datetime

from _env import tg_config
TG_TOKEN, TG_CHAT = tg_config()
EVAL_PID      = 11808
TOTAL_BATCHES = 601
START_TIME    = datetime(2026, 6, 15, 20, 50, 0)   # restarted ~8:50 PM
SEC_PER_BATCH = 18

LOG_FILE = r"C:\Users\Admin\kaskor-asr-v0.0\eval_log.txt"

def tg(text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text},
            timeout=10
        )
    except Exception as e:
        print(f"[TG error] {e}")

def is_running(pid):
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True
    )
    return str(pid) in result.stdout

def get_results():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except:
        return None

def reset_lid_setting():
    subprocess.run("powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1", shell=True)
    subprocess.run("powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1", shell=True)
    subprocess.run("powercfg /S SCHEME_CURRENT", shell=True)
    print("Lid-close reset to Sleep.")

def estimate_progress():
    elapsed_sec = (datetime.now() - START_TIME).total_seconds()
    batches_done = min(int(elapsed_sec / SEC_PER_BATCH), TOTAL_BATCHES)
    pct = round(batches_done / TOTAL_BATCHES * 100, 1)
    remaining_sec = max(0, (TOTAL_BATCHES - batches_done) * SEC_PER_BATCH)
    remaining_min = round(remaining_sec / 60)
    eta = datetime.fromtimestamp(time.time() + remaining_sec).strftime("%I:%M %p")
    return batches_done, pct, remaining_min, eta

def main():
    tg("📡 Monitor v2 started — will update every 10 min, auto-shutdown when done.")
    print("Monitor running.")

    while True:
        time.sleep(600)

        if not is_running(EVAL_PID):
            # ── Eval finished ──────────────────────────────
            results = get_results()

            if results:
                tg(results)
            else:
                tg("✅ Eval finished! Could not read log file — check eval_log.txt manually.")

            # Reset lid-close to Sleep
            reset_lid_setting()
            tg("🔧 Lid-close setting reset to Sleep.")

            # Shutdown in 60 seconds
            tg("🔴 Shutting down in 60 seconds...")
            subprocess.run("shutdown /s /t 60", shell=True)
            print("Shutdown triggered.")
            break

        # ── Still running — send progress update ──────────
        batches_done, pct, remaining_min, eta = estimate_progress()
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        msg = (
            f"📊 Eval Progress Update\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"[{bar}]\n"
            f"🔢 Batch : {batches_done}/{TOTAL_BATCHES}\n"
            f"📈 Done  : {pct}%\n"
            f"⏳ ETA   : {remaining_min} min ({eta})\n"
            f"⚡ Status: Running ✅\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        tg(msg)
        print(f"[{datetime.now().strftime('%H:%M')}] Update sent: {pct}%")

if __name__ == "__main__":
    main()
