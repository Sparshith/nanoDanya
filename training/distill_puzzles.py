"""Evaluate a student ZIP on the canonical 8,800-puzzle leaderboard sample."""
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmark.run import sample_puzzles_by_rating


def main():
    dataset = Path(os.getenv("PUZZLE_DATASET", "/data/puzzles_raw"))
    if os.getenv("PUZZLE_PREPARE") == "1":
        items, _, _ = sample_puzzles_by_rating(
            ndjson=dataset / "puzzle_games_ndjson.txt",
            metadata=dataset / "puzzle_metadata.txt",
            rating_min=600, rating_max=2800, bin_width=100, per_bin=400, scan_cap=0,
        )
        if len(items) != 8800:
            raise ValueError(f"Expected 8,800 puzzles, got {len(items)}")
        output = Path(os.environ["PUZZLE_SAMPLE_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "puzzle_metadata.txt").write_text(json.dumps(
            {item["game_id"]: [item["puzzle"]] for item in items}, separators=(",", ":")))
        (output / "puzzle_games_ndjson.txt").write_text("".join(
            json.dumps({"id": item["game_id"], "moves": " ".join(item["moves"])},
                       separators=(",", ":")) + "\n" for item in items))
        print(f"Saved the canonical sample to {output}", flush=True)
        return

    subprocess.run([
        sys.executable, "benchmark/run.py", "puzzles",
        "--model", os.environ["STUDENT_ARTIFACT"],
        "--ndjson", str(dataset / "puzzle_games_ndjson.txt"),
        "--metadata", str(dataset / "puzzle_metadata.txt"),
        "--rating-min", "600", "--rating-max", "2800", "--bin-width", "100",
        "--per-bin", "400", "--batch-size", "256", "--write-positions",
        "--output", os.environ["PUZZLE_OUTPUT"],
    ], check=True)


if __name__ == "__main__":
    main()
