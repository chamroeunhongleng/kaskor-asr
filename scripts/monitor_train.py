"""
Monitors epoch 3 training. When done:
  1. Sends CER/loss results to Telegram
  2. Resets lid-close to Sleep
  3. Shuts down PC in 60 seconds
"""
import time
import subprocess
import requests
import sys
from datetime import datetime

from _env import tg_config
TG_TOKEN, TG_CHAT = tg_config()
TRAIN_PID = int(sys.argv[1]) if len(sys.argv) > 1 else None

LOG_FILE  = r"C:\Users\Admin\kaskor-asr-v0.0\train_log.txt"
ERR_FILE  = r"C:\Users\Admin\kaskor-asr-v0.0\train_err.txt"

START_TIME     = datetime.now()
TOTAL_STEPS    = 5410
SEC_PER_STEP   = 1.52   # ~0.657 steps/sec from epoch 2

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

def get_last_loss():
    try:
        with open(ERR_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "'loss'" in line:
                return line.strip()
    except:
        pass
    return "N/A"

def get_final_results():
    try:
        with open(ERR_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        results = [l.strip() for l in lines if "eval_loss" in l or "eval_cer" in l or "train_loss" in l and "epoch" in l]
        return "\n".join(results[-5:]) if results else None
    except:
        return None

def reset_lid():
    subprocess.run("powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1", shell=True)
    subprocess.run("powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1", shell=True)
    subprocess.run("powercfg /S SCHEME_CURRENT", shell=True)

def estimate_progress():
    elapsed = (datetime.now() - START_TIME).total_seconds()
    steps_done = min(int(elapsed / SEC_PER_STEP), TOTAL_STEPS)
    pct = round(steps_done / TOTAL_STEPS * 100, 1)
    remaining_sec = max(0, (TOTAL_STEPS - steps_done) * SEC_PER_STEP)
    remaining_min = round(remaining_sec / 60)
    eta = datetime.fromtimestamp(time.time() + remaining_sec).strftime("%I:%M %p")
    return steps_done, pct, remaining_min, eta

def main():
    if not TRAIN_PID:
        print("Usage: python monitor_train.py <PID>")
        return

    tg(f"🚀 Epoch 3 Training STARTED\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n📍 PID     : {TRAIN_PID}\n🔢 Steps   : {TOTAL_STEPS}\n⏱ Est. time: ~2.3 hours\n🔔 Will auto-report + shutdown when done")
    print(f"Monitoring PID {TRAIN_PID}")

    while True:
        time.sleep(1800)

        if not is_running(TRAIN_PID):
            results = get_final_results()
            last_loss = get_last_loss()

            if results:
                tg(f"🎉 EPOCH 3 TRAINING COMPLETE!\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n{results}\n\nLast logged loss:\n{last_loss}")
            else:
                tg(f"✅ Epoch 3 training done!\n\nLast logged loss:\n{last_loss}\n\nCheck train_log.txt for full results.")

            reset_lid()
            tg("🔧 Lid-close reset to Sleep.\n🔴 Shutting down in 60 seconds...")
            subprocess.run("shutdown /s /t 60", shell=True)
            print("Done. Shutdown triggered.")
            break

        steps_done, pct, remaining_min, eta = estimate_progress()
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        last_loss = get_last_loss()
        msg = (
            f"🏋️ Epoch 3 Training Update\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"[{bar}]\n"
            f"🔢 Step  : {steps_done}/{TOTAL_STEPS}\n"
            f"📈 Done  : {pct}%\n"
            f"📉 Loss  : {last_loss}\n"
            f"⏳ ETA   : {remaining_min} min ({eta})\n"
            f"⚡ Status: Running ✅\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        tg(msg)
        print(f"[{datetime.now().strftime('%H:%M')}] Update sent: {pct}%")

if __name__ == "__main__":
    main()
