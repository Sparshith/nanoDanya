import csv
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import zstandard as zstd
from tqdm import tqdm

PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
EXPORT_URL = "https://lichess.org/api/games/export/_ids"
BATCH_SIZE = 300
MAX_WORKERS = 8
DATA_DIR = Path(__file__).parent.parent / "data"


def extract_puzzles():
    r = requests.get(PUZZLE_URL, stream=True)
    r.raise_for_status()
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(r.raw)
    text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace", newline="")

    puzzles_by_game = {}
    for row in tqdm(csv.DictReader(text), desc="Streaming puzzle CSV"):
        url = row.get("GameUrl", "")
        m = re.search(r"lichess\.org/(\w{8})", url)
        if not m:
            continue
        game_id = m.group(1)
        move_match = re.search(r"#(\d+)", url)
        move_num = int(move_match.group(1)) if move_match else None

        puzzles_by_game.setdefault(game_id, []).append({
            "puzzle_id": row["PuzzleId"],
            "fen": row["FEN"],
            "moves": row["Moves"],
            "rating": int(row["Rating"]),
            "move_num": move_num,
        })

    return puzzles_by_game


def download_batch(session, game_ids, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = session.post(
                EXPORT_URL,
                data=",".join(game_ids),
                headers={"Accept": "application/x-ndjson"},
                timeout=120,
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return [l for l in resp.text.strip().split("\n") if l.strip()]
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Batch failed: {e}")
                return []
            time.sleep(2 ** attempt)
    return []


def load_downloaded_ids(path):
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ids_file = DATA_DIR / "game_ids.txt"
    meta_file = DATA_DIR / "puzzle_metadata.json"
    games_file = DATA_DIR / "puzzle_games.ndjson"

    if ids_file.exists() and meta_file.exists():
        print(f"Loading cached game IDs from {ids_file}")
        game_ids = ids_file.read_text().strip().split("\n")
    else:
        puzzles_by_game = extract_puzzles()
        game_ids = sorted(puzzles_by_game.keys())
        ids_file.write_text("\n".join(game_ids))
        with open(meta_file, "w") as f:
            json.dump(puzzles_by_game, f)
        print(f"{len(game_ids)} unique games, {sum(len(v) for v in puzzles_by_game.values())} puzzles")

    downloaded = load_downloaded_ids(games_file)
    remaining = [gid for gid in game_ids if gid not in downloaded]
    print(f"{len(game_ids)} total, {len(downloaded)} done, {len(remaining)} remaining")

    if not remaining:
        print("All games downloaded!")
        return

    batches = [remaining[i : i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    lock = threading.Lock()
    session = requests.Session()

    with open(games_file, "a") as f:
        pbar = tqdm(total=len(remaining), desc="Downloading games")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(download_batch, session, batch): batch
                for batch in batches
            }
            for future in as_completed(futures):
                lines = future.result()
                batch = futures[future]
                with lock:
                    for line in lines:
                        f.write(line + "\n")
                    f.flush()
                pbar.update(len(batch))

        pbar.close()

    print(f"Done! {len(game_ids)} games saved to {games_file}")


if __name__ == "__main__":
    main()
