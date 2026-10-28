"""Run the complete browser-student experiment through the shared $100 ledger."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "output/distill-20260924/budget.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="student-l6e384-repro")
    parser.add_argument("--profile", default="second")
    parser.add_argument("--gpu", choices=["H100", "B200"], default="B200")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("run-name must contain only letters, numbers, hyphens, and underscores")

    base = f"/data/checkpoints/distill/{args.run_name}"
    out_dir = ROOT / "output" / args.run_name
    if out_dir.exists():
        parser.error(f"local output already exists: {out_dir}; choose a new run-name")
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"runs": []}
    if any(run["stage"].startswith(f"{args.run_name}-") for run in ledger["runs"]):
        parser.error("run-name already has budgeted stages; choose a new run-name")
    modal = shutil.which("modal") or str(Path.home() / ".local/bin/modal")
    modal_env = {**os.environ, "MODAL_PROFILE": args.profile}
    listing = subprocess.check_output(
        [modal, "volume", "ls", "--json", "nanodanya-data", "checkpoints/distill"],
        cwd=ROOT, env=modal_env, text=True,
    )
    if any(item["Filename"].rstrip("/") == base.removeprefix("/data/") for item in json.loads(listing)):
        parser.error("run-name already has checkpoints on the Modal volume; choose a new run-name")

    data = {"DATASET_DIR": "datasets/games/15m"}
    float_dir, qat_dir = f"{base}/float", f"{base}/qat"
    archive = f"{base}/student-int4.onnx"
    teacher = "plain/games-15m/l12"
    train_shape = {"BATCH_SIZE": "64" if args.gpu == "B200" else "32",
                   "GRAD_ACCUM_STEPS": "1" if args.gpu == "B200" else "2",
                   "VAL_BATCHES": "4" if args.gpu == "B200" else "8"}
    stages = [
        ("pilot", args.gpu, 600, "training/train_distill.py", data | {
            "CKPT_DIR": float_dir, "BATCH_SIZE": "32", "BLOCK_SIZE": "256", "GRAD_ACCUM_STEPS": "1",
            "MAX_STEPS": "100", "MAX_SECONDS": "450", "LR_SCHEDULE": "constant", "LR": "0.0003",
            "EVAL_INTERVAL": "100", "VAL_BATCHES": "4", "CKPT_INTERVAL": "100"}),
        ("main", args.gpu, 3600, "training/train_distill.py", data | train_shape | {
            "CKPT_DIR": float_dir, "BLOCK_SIZE": "256",
            "MAX_STEPS": "25000", "MAX_SECONDS": "3300", "LR_SCHEDULE": "cosine", "LR": "0.0003",
            "MIN_LR": "0.00003", "WARMUP_STEPS": "200", "DECAY_STEPS": "25000",
            "EVAL_INTERVAL": "1000", "CKPT_INTERVAL": "1000"}),
        ("qat", args.gpu, 900, "training/train_distill.py", data | train_shape | {
            "INIT_FROM": f"{float_dir}/student_best.pt", "CKPT_DIR": qat_dir, "QAT": "1",
            "BLOCK_SIZE": "256",
            "MAX_STEPS": "2000", "MAX_SECONDS": "600", "LR_SCHEDULE": "cosine", "LR": "0.00003",
            "MIN_LR": "0.000003", "WARMUP_STEPS": "0", "DECAY_STEPS": "2000",
            "EVAL_INTERVAL": "250", "CKPT_INTERVAL": "250"}),
        ("export", "none", 300, "inference/export_onnx.py", {
            "STUDENT_CHECKPOINT": f"{qat_dir}/student_best.pt", "STUDENT_ARTIFACT": archive, "EXPORT_PRECISION": "int4"}),
        ("heldout", "L4", 1200, "training/distill_eval.py", {
            "STUDENT_CHECKPOINT": f"{qat_dir}/student_best.pt", "STUDENT_ARTIFACT": archive,
            "DISTILL_EVAL_MAX_POSITIONS": "1024", "DISTILL_EVAL_OUTPUT": f"{base}/heldout.jsonl"}),
    ]
    for name, model, opponent in (
        ("teacher-sf1000", teacher, ""),
        ("float-sf1000", f"{float_dir}/student_best.pt", ""),
        ("q4-sf1000", archive, ""),
        ("q4-h2h", archive, teacher),
    ):
        settings = {"DISTILL_MODEL": model, "DISTILL_GAMES": "100",
                    "DISTILL_GAME_OUTPUT": f"{base}/{name}.jsonl"}
        settings |= ({"DISTILL_GAME_MODE": "h2h", "DISTILL_OPPONENT": opponent} if opponent else
                     {"DISTILL_GAME_MODE": "stockfish", "DISTILL_STOCKFISH_ELO": "1000"})
        stages.append((name, args.gpu, 1800, "training/distill_games.py", settings))

    gpu_rates = {"H100": 0.001097, "B200": 0.001736, "L4": 0.000222, "none": 0}
    planned = sum((timeout + 180) * (gpu_rates[gpu] + 4 * 0.0000131 + 16 * 0.00000222)
                  for _, gpu, timeout, _, _ in stages)
    reserved = ledger.get("overhead_reserve_usd", 10) + sum(run["reserved_usd"] for run in ledger["runs"])
    if reserved + planned > ledger.get("limit_usd", 100):
        parser.error(f"full pipeline would reserve ${reserved + planned:.2f}, above the shared budget")

    for name, gpu, timeout, script, settings in stages:
        print(f"Running {name} for {args.run_name}", flush=True)
        subprocess.run([sys.executable, "training/run_distillation.py", "--stage", f"{args.run_name}-{name}",
                        "--timeout", str(timeout), "--gpu", gpu, "--profile", args.profile,
                        "--script", script, "--env", json.dumps(settings)], cwd=ROOT, check=True)

    out_dir.mkdir(parents=True)
    for filename in ("student-int4.onnx", "heldout.jsonl", "teacher-sf1000.jsonl",
                     "float-sf1000.jsonl", "q4-sf1000.jsonl", "q4-h2h.jsonl"):
        subprocess.run([modal, "volume", "get", "nanodanya-data",
                        f"checkpoints/distill/{args.run_name}/{filename}", str(out_dir / filename)],
                       cwd=ROOT, env=modal_env, check=True)
    print(f"Saved ONNX and results to {out_dir}")

if __name__ == "__main__":
    main()
