"""Run canonical game benchmarks from an environment-driven Modal training job."""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
import subprocess
import sys


def main():
    mode = os.getenv("DISTILL_GAME_MODE", "h2h")
    model = os.environ["DISTILL_MODEL"]
    output = os.environ["DISTILL_GAME_OUTPUT"]
    if mode not in ("h2h", "stockfish"):
        raise ValueError("DISTILL_GAME_MODE must be h2h or stockfish")
    command = [
        sys.executable, "benchmark/run.py", "games",
        "--game-mode", mode,
        "--model", model,
        "--games", os.getenv("DISTILL_GAMES", "100"),
        "--batch-size", os.getenv("DISTILL_GAME_BATCH_SIZE", "16"),
        "--max-plies", os.getenv("DISTILL_MAX_PLIES", "200"),
        "--seed", os.getenv("DISTILL_GAME_SEED", "5000"),
        "--temperature", "0.2",
        "--output", output,
    ]
    if mode == "h2h":
        command += ["--opponent-model", os.environ["DISTILL_OPPONENT"]]
    else:
        digest = hashlib.sha256(Path("/usr/local/bin/fairy-stockfish").read_bytes()).hexdigest()
        if digest != "41b8b4d539adfd9924929ee4a948d1a37dd1e9beaa535a811cb5e7fee9e4cb99":
            raise ValueError("Game comparisons require the canonical Fairy-Stockfish 14 binary")
        print(f"Fairy-Stockfish 14 sha256={digest}", flush=True)
        command += [
            "--stockfish", "/usr/local/bin/fairy-stockfish",
            "--stockfish-elo", os.getenv("DISTILL_STOCKFISH_ELO", "500"),
            "--stockfish-depth", "1",
        ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
