"""Rate an exported student with the canonical fixed-opening tournament protocol."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def main():
    artifact = os.environ["STUDENT_ARTIFACT"]
    output = Path(os.environ["TOURNAMENT_OUTPUT"])
    label = os.getenv("TOURNAMENT_LABEL", "nanodanya-student-int4")
    openings = int(os.getenv("TOURNAMENT_OPENINGS", "200"))
    workers = int(os.getenv("TOURNAMENT_WORKERS", "4"))
    stockfish = "/usr/local/bin/fairy-stockfish"
    digest = hashlib.sha256(Path(stockfish).read_bytes()).hexdigest()
    if digest != "41b8b4d539adfd9924929ee4a948d1a37dd1e9beaa535a811cb5e7fee9e4cb99":
        raise ValueError("Tournament requires the canonical Fairy-Stockfish 14 binary")
    directory = output.parent / (output.stem + "-shards")
    directory.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run_shard(index):
        start, stop = index * openings // workers, (index + 1) * openings // workers
        path = directory / f"{index}.jsonl"
        command = [sys.executable, "benchmark/run.py", "tournament", "--players",
                   f"{label}={artifact}", "sf1000", "sf1500", "sf2000",
                   "--openings", str(stop - start), "--opening-offset", str(start),
                   "--stockfish", stockfish, "--stockfish-depth", "1",
                   "--adjudicate-time", "0.01", "--adjudicate-cp", "1300",
                   "--max-plies", "400", "--temperature", "0", "--seed", "0",
                   "--batch-size", "64", "--output", str(path)]
        with path.with_suffix(".log").open("w") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        summary = next(record for record in records if record["type"] == "tournament_summary")
        print(json.dumps({"shard": index, **summary}), flush=True)
        return records

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = [record for shard in pool.map(run_shard, range(workers)) for record in shard]
    games = [record for record in records if record["type"] == "game"]
    if len(games) != openings * 2 * 3:
        raise ValueError(f"Incomplete tournament: {len(games)} games")
    output.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    print(f"Saved {len(games)} tournament games to {output}", flush=True)


if __name__ == "__main__":
    main()
