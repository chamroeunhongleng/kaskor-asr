"""
Continuous train -> eval -> decide loop for KASEKOR ASR.

Runs unattended: trains a chunk, scores it, records the result, picks what to do
next, repeats. Stops on its own when the target CER is reached, when the
improvement ladder is exhausted, or when it runs out of rounds / wall-clock.

WHY A LADDER AND NOT JUST "MORE EPOCHS"
Epochs 1-5 moved CER 18.24% -> 17.48%. Repeating that would have kept
plateauing, because two different things were capping it:
  1. Decoding was truncated at 225 tokens (~102 Khmer chars) while half the
     references are longer. Fixed in run_eval.py / train.py — that is a
     measurement fix, not a model improvement, so round 0 re-establishes the
     real baseline before any training happens.
  2. Teacher-forced loss is already 0.0346 (perplexity ~1.035). The model fits
     the data; what is left is mostly free-running decode error. Hammering the
     same LR on the same data does not move that, so each plateau advances a
     rung (lower LR, more augmentation) instead of repeating the last round.

Every round is journalled to loop_state.json, so a reboot, a crash, or a
Ctrl-C resumes where it stopped rather than starting over.

Usage:
    python scripts/train_loop.py                          # target 2.0% CER
    python scripts/train_loop.py --target-cer 5 --max-rounds 6
    python scripts/train_loop.py --dry-run                # print the plan, run nothing
    python scripts/train_loop.py --status                 # show the journal and exit
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
STATE_FILE = ROOT / "loop_state.json"
TMP = ROOT / ".tmp"

# Each rung is tried until it stops paying off, then the loop advances.
# Ordered from "cheapest thing that might still help" to "fine polish".
LADDER = [
    {"name": "resume-baseline", "lr": 1e-5, "augment_prob": 0.4, "weight_decay": 0.01,
     "why": "current settings, one more epoch — cheapest test that training still helps"},
    {"name": "lr-decay-1",      "lr": 5e-6, "augment_prob": 0.5, "weight_decay": 0.01,
     "why": "halve LR and widen augmentation — the usual fix when train loss is flat but decode lags"},
    {"name": "lr-decay-2",      "lr": 2e-6, "augment_prob": 0.6, "weight_decay": 0.02,
     "why": "smaller steps, heavier augmentation, more L2 — target exposure-bias error"},
    {"name": "polish",          "lr": 1e-6, "augment_prob": 0.2, "weight_decay": 0.01,
     "why": "low LR on mostly-clean audio — settle onto the data distribution"},
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"created": now(), "rounds": []}


def save_state(state: dict) -> None:
    state["updated"] = now()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def tg(text: str) -> None:
    """Telegram ping — same channel the training scripts already use."""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from _env import tg_config
        import requests
        token, chat = tg_config()
        if not token or not chat:
            return
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text}, timeout=10)
    except Exception as e:
        print(f"[Telegram skipped] {e}")


def run(cmd: list[str], log_name: str) -> tuple[int, str]:
    """Run a subprocess, streaming to console and tee-ing to .tmp/<log_name>.

    Each round is a fresh process on purpose: an 8 GB card fragments badly if
    training and beam-search generation share one long-lived CUDA context.
    Returns (returncode, captured_stdout).
    """
    TMP.mkdir(parents=True, exist_ok=True)
    log_path = TMP / log_name
    print(f"\n$ {' '.join(cmd)}\n  (log -> {log_path})", flush=True)
    captured = []
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            captured.append(line)
        proc.wait()
    return proc.returncode, "".join(captured)


def current_weights(state: dict) -> str:
    """Which checkpoint the next round should start from.

    Rounds chain: each fine-tunes the previous round's output. Before any round
    has run, that is the epoch-5 checkpoint the original training left behind.
    """
    for r in reversed(state.get("rounds", [])):
        if r.get("produced_checkpoint"):
            return r["produced_checkpoint"]
    ckpts = sorted((ROOT / "checkpoints").glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    return str(ckpts[-1]) if ckpts else str(ROOT / "checkpoints" / "best")


def do_eval(state: dict, tag: str, args, checkpoint: str | None = None) -> dict | None:
    """Score a checkpoint and return the parsed result dict."""
    out_json = TMP / f"eval_{tag}.json"
    cmd = [PY, "scripts/run_eval.py",
           "--split", args.split,
           "--beams", str(args.beams),
           "--eval-samples", str(args.eval_samples),
           "--json-out", str(out_json)]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    rc, _ = run(cmd, f"eval_{tag}.log")
    if rc != 0 or not out_json.exists():
        print(f"[loop] eval failed (rc={rc}) — see .tmp/eval_{tag}.log")
        return None
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def best_cer(state: dict) -> float:
    cers = [r["cer"] for r in state["rounds"] if r.get("cer") is not None]
    return min(cers) if cers else float("inf")


def print_status(state: dict) -> None:
    rounds = state.get("rounds", [])
    if not rounds:
        print("No rounds recorded yet.")
        return
    print(f"\n{'round':<6} {'action':<18} {'lr':<9} {'aug':<5} {'CER%':<8} {'WER%':<8} {'best':<6} when")
    print("-" * 92)
    for r in rounds:
        cer = f"{r['cer']:.2f}" if r.get("cer") is not None else "—"
        wer = f"{r['wer']:.1f}" if r.get("wer") is not None else "—"
        lr = f"{r['lr']:.1e}" if r.get("lr") else "—"
        aug = f"{r['augment_prob']:.1f}" if r.get("augment_prob") is not None else "—"
        star = "  *" if r.get("is_best") else ""
        print(f"{r['round']:<6} {r.get('action','')[:18]:<18} {lr:<9} {aug:<5} "
              f"{cer:<8} {wer:<8} {star:<6} {r.get('finished','')[:19]}")
    print("-" * 92)
    print(f"best CER so far: {best_cer(state):.2f}%   rounds: {len(rounds)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-cer", type=float, default=2.0,
                    help="stop once eval CER is at or below this (default 2.0)")
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--epochs-per-round", type=int, default=1)
    ap.add_argument("--patience", type=int, default=2,
                    help="rounds below min-delta before advancing a rung")
    ap.add_argument("--min-delta", type=float, default=0.30,
                    help="absolute CER points that count as real improvement")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--beams", type=int, default=1,
                    help="decode width during the loop (final eval uses --final-beams)")
    ap.add_argument("--final-beams", type=int, default=5)
    ap.add_argument("--eval-samples", type=int, default=1000,
                    help="fixed-seed subsample scored each round (0 = full split)")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    state = load_state()

    if args.status:
        print_status(state)
        return 0

    if args.dry_run:
        print("PLAN")
        print(f"  target CER      : <= {args.target_cer}%")
        print(f"  stop after      : {args.max_rounds} rounds or {args.max_hours}h")
        print(f"  each round      : {args.epochs_per_round} epoch, then eval on "
              f"{args.eval_samples or 'all'} {args.split} utterances at beams={args.beams}")
        print(f"  advance a rung  : after {args.patience} rounds improving < {args.min_delta} CER points")
        print(f"  ladder:")
        for i, rung in enumerate(LADDER):
            print(f"    {i}. {rung['name']:<16} lr={rung['lr']:.0e} aug={rung['augment_prob']} "
                  f"wd={rung['weight_decay']}  — {rung['why']}")
        print(f"  journal         : {STATE_FILE}")
        print_status(state)
        return 0

    t_start = time.time()
    deadline = t_start + args.max_hours * 3600

    # ── Round 0: re-baseline with the truncation fix in place ─────────────────
    # The 17.48% on record was measured with decoding capped at ~102 chars, so
    # it cannot be compared against anything this loop produces.
    if not args.skip_baseline and not any(r.get("action") == "baseline" for r in state["rounds"]):
        print("=" * 72)
        print("ROUND 0 — baseline with the 225-token decode cap removed")
        print("The 17.48% on record was measured through that cap and is not comparable.")
        print("=" * 72)
        res = do_eval(state, "baseline", args)
        if res is None:
            print("[loop] baseline eval failed — stopping before training.")
            return 1
        state["rounds"].append({
            "round": 0, "action": "baseline", "started": now(), "finished": now(),
            "cer": res["cer"], "wer": res["wer"], "eval_loss": res["eval_loss"],
            "checkpoint": res["checkpoint"], "samples": res["samples"],
            "beams": res["beams"], "is_best": True,
            "note": "decode cap lifted 225 -> 448; not comparable to the 17.48% on record",
        })
        save_state(state)
        tg(f"KASEKOR loop — baseline (decode cap fixed)\nCER {res['cer']}%  WER {res['wer']}%\n"
           f"prior on-record number was 17.48% but truncated at ~102 chars")
        print(f"\n[loop] TRUE BASELINE: {res['cer']}% CER  (was reported as 17.48% under truncation)")

    # ── Main loop ─────────────────────────────────────────────────────────────
    rung = 0
    stagnant = 0
    for r in state["rounds"]:
        if r.get("rung") is not None:
            rung = r["rung"]
        if r.get("stagnant") is not None:
            stagnant = r["stagnant"]

    trained_rounds = [r for r in state["rounds"] if r.get("action") != "baseline"]
    round_no = len(state["rounds"])

    while True:
        cur_best = best_cer(state)

        if cur_best <= args.target_cer:
            msg = f"TARGET MET — best CER {cur_best:.2f}% <= {args.target_cer}%"
            print(f"\n[loop] {msg}")
            tg(f"KASEKOR loop finished\n{msg}")
            break
        if len(trained_rounds) >= args.max_rounds:
            print(f"\n[loop] STOP — {args.max_rounds} training rounds used. Best {cur_best:.2f}%.")
            tg(f"KASEKOR loop stopped: round budget spent. Best CER {cur_best:.2f}%")
            break
        if time.time() > deadline:
            print(f"\n[loop] STOP — {args.max_hours}h budget spent. Best {cur_best:.2f}%.")
            tg(f"KASEKOR loop stopped: time budget spent. Best CER {cur_best:.2f}%")
            break
        if rung >= len(LADDER):
            print(f"\n[loop] STOP — ladder exhausted at {cur_best:.2f}% CER without reaching "
                  f"{args.target_cer}%.\n       Remaining error is not a hyperparameter problem; "
                  f"see the summary below.")
            tg(f"KASEKOR loop stopped: ladder exhausted. Best CER {cur_best:.2f}% "
               f"(target {args.target_cer}%). Needs a data change, not a knob change.")
            break

        cfg = LADDER[rung]
        print("\n" + "=" * 72)
        print(f"ROUND {round_no} — rung {rung} '{cfg['name']}'  lr={cfg['lr']:.0e} "
              f"aug={cfg['augment_prob']} wd={cfg['weight_decay']}")
        print(f"  {cfg['why']}")
        print(f"  best so far {cur_best:.2f}%  target {args.target_cer}%  "
              f"stagnant rounds {stagnant}/{args.patience}")
        print("=" * 72)

        started = now()
        init_from = current_weights(state)
        print(f"  starting from {init_from}")
        # --force bypasses train.py's own plateau guard; this loop owns that decision.
        # --init-from + --output-subdir put train.py in round mode: fresh
        # optimizer so --lr actually takes effect, isolated output dir so the
        # restarted step numbering cannot collide with checkpoint-27050.
        rc, out = run([PY, "scripts/train.py", "--no-eval", "--force",
                       "--init-from", init_from,
                       "--output-subdir", f"round_{round_no}",
                       "--epochs-per-run", str(args.epochs_per_round),
                       "--lr", str(cfg["lr"]),
                       "--augment-prob", str(cfg["augment_prob"]),
                       "--weight-decay", str(cfg["weight_decay"])],
                      f"train_r{round_no}.log")
        if rc != 0:
            print(f"[loop] training failed (rc={rc}) — see .tmp/train_r{round_no}.log. Stopping.")
            tg(f"KASEKOR loop ABORTED — training round {round_no} failed (rc={rc})")
            state["rounds"].append({"round": round_no, "action": cfg["name"], "rung": rung,
                                    "started": started, "finished": now(), "failed": True,
                                    "returncode": rc})
            save_state(state)
            return 1

        produced = None
        for line in out.splitlines():
            if line.startswith("ROUND_CHECKPOINT="):
                produced = line.split("=", 1)[1].strip()
        if not produced:
            print(f"[loop] training round {round_no} produced no checkpoint marker — stopping.")
            tg(f"KASEKOR loop ABORTED — round {round_no} produced no checkpoint")
            return 1

        res = do_eval(state, f"r{round_no}", args, checkpoint=produced)
        if res is None:
            tg(f"KASEKOR loop ABORTED — eval round {round_no} failed")
            return 1

        improvement = cur_best - res["cer"]
        is_best = res["cer"] < cur_best
        if improvement < args.min_delta:
            stagnant += 1
        else:
            stagnant = 0

        entry = {
            "round": round_no, "action": cfg["name"], "rung": rung,
            "lr": cfg["lr"], "augment_prob": cfg["augment_prob"],
            "weight_decay": cfg["weight_decay"],
            "started": started, "finished": now(),
            "cer": res["cer"], "wer": res["wer"], "eval_loss": res["eval_loss"],
            "checkpoint": res["checkpoint"], "produced_checkpoint": produced,
            "init_from": init_from, "samples": res["samples"],
            "improvement": round(improvement, 3), "is_best": is_best,
            "stagnant": stagnant,
        }
        state["rounds"].append(entry)
        trained_rounds.append(entry)
        save_state(state)

        print(f"\n[loop] round {round_no}: CER {res['cer']}%  "
              f"({improvement:+.2f} vs best)  {'NEW BEST' if is_best else ''}")
        tg(f"KASEKOR loop round {round_no} ({cfg['name']})\n"
           f"CER {res['cer']}%  WER {res['wer']}%\n"
           f"change vs best: {improvement:+.2f} pts\nbest now {min(cur_best, res['cer']):.2f}%")

        if stagnant >= args.patience:
            rung += 1
            stagnant = 0
            entry["rung_advanced_to"] = rung
            save_state(state)
            if rung < len(LADDER):
                print(f"[loop] {args.patience} rounds without a {args.min_delta}-point gain "
                      f"— advancing to rung {rung} '{LADDER[rung]['name']}'")

        round_no += 1

    # ── Final: re-score the best model properly, full split, wide beams ────────
    final = None
    if any(r.get("cer") is not None for r in state["rounds"]):
        print("\n" + "=" * 72)
        print(f"FINAL — re-scoring checkpoints/best on the FULL {args.split} split "
              f"at beams={args.final_beams}")
        print("=" * 72)
        final_args = argparse.Namespace(**vars(args))
        final_args.beams = args.final_beams
        final_args.eval_samples = 0            # full split
        final = do_eval(state, "final", final_args, checkpoint=str(ROOT / "checkpoints" / "best"))
        if final:
            state["final"] = final
            save_state(state)

    print_status(state)
    print(f"\nJournal: {STATE_FILE}")
    if final:
        print(f"FINAL (full {args.split}, beams={args.final_beams}): "
              f"CER {final['cer']}%  WER {final['wer']}%")
        tg(f"KASEKOR loop COMPLETE\nFinal CER {final['cer']}%  WER {final['wer']}% "
           f"(full {args.split}, beams={args.final_beams})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
