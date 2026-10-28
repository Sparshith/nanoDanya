"""Reserve capped Modal runs against one persistent experiment budget."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--env", default="{}")
    parser.add_argument("--profile", default="second")
    parser.add_argument("--gpu", choices=["H100", "B200", "L4", "none"], default="H100")
    parser.add_argument("--ledger", type=Path, default=ROOT / "output/distill-20260924/budget.json")
    args = parser.parse_args()
    if not 0 < args.timeout <= 12 * 3600:
        parser.error("timeout must be between 1 and 43200 seconds")
    modal = shutil.which("modal") or str(Path.home() / ".local/bin/modal")
    settings = {"PYTHONPATH": "/root/project:/root/project/nanochat", **json.loads(args.env)}
    # Modal public rates, checked 2026-09-24. Reserve startup and idle time as well.
    gpu_rate = {"H100": 0.001097, "B200": 0.001736, "L4": 0.000222, "none": 0}[args.gpu]
    rate = gpu_rate + 4 * 0.0000131 + 16 * 0.00000222
    reserved = (args.timeout + 180) * rate
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(args.ledger.read_text()) if args.ledger.exists() else {
            "limit_usd": 100, "overhead_reserve_usd": 10, "runs": []}
        total = ledger["overhead_reserve_usd"] + sum(r["reserved_usd"] for r in ledger["runs"])
        if total + reserved > ledger["limit_usd"]:
            raise SystemExit(f"Budget refused: ${total + reserved:.2f} > ${ledger['limit_usd']}")
        if any(r["stage"] == args.stage for r in ledger["runs"]):
            raise SystemExit("Stage already reserved. Use a new stage name for an intentional retry.")
        run = {"stage": args.stage, "reserved_usd": reserved, "rate_usd_per_second": rate,
               "timeout_seconds": args.timeout, "profile": args.profile, "gpu": args.gpu,
               "script": args.script, "env": settings, "started_at": time.time(), "status": "reserved"}
        ledger["runs"].append(run)
        args.ledger.write_text(json.dumps(ledger, indent=2) + "\n")
        print(f"Reserved ${reserved:.2f}; cumulative ceiling ${total + reserved:.2f}/${ledger['limit_usd']:g}", flush=True)
        environment = {**os.environ, "MODAL_PROFILE": args.profile, "MODAL_GPU": args.gpu,
                       "MODAL_CPU": "4", "MODAL_MEMORY": "16384", "MODAL_BUDGETED": "1",
                       "MODAL_TIMEOUT": str(args.timeout)}
        log_path = args.ledger.parent / f"{args.stage}.log"
        snapshot = tempfile.TemporaryDirectory(prefix="nanodanya-distill-")
        try:
            subprocess.run(["rsync", "-a", "--exclude-from", str(ROOT / ".modalignore"),
                            str(ROOT) + "/", snapshot.name + "/"], check=True)
            environment["MODAL_PROJECT_DIR"] = snapshot.name
            command = [modal, "run", str(Path(snapshot.name) / "modal_train.py"),
                       "--datasets", "datasets/games/15m", "--script", args.script,
                       "--env-overrides", json.dumps(settings)]
            with log_path.open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    if "RESOURCE_RECEIPT=" in line:
                        receipt = json.loads(line.split("RESOURCE_RECEIPT=", 1)[1])
                        run["receipt"] = receipt
                        run["estimated_compute_usd"] = receipt["seconds"] * rate
                run["returncode"] = process.wait()
                run["status"] = "complete" if process.returncode == 0 else "failed"
        finally:
            snapshot.cleanup()
            # Keep the full reservation even after an early exit or lost connection.
            run["finished_at"] = time.time()
            args.ledger.write_text(json.dumps(ledger, indent=2) + "\n")
        raise SystemExit(run["returncode"])


if __name__ == "__main__":
    main()
