import json
from datetime import date


def log_result(result, model, opponent_elo, games=100, opponent="stockfish", notes=None):
    entry = {
        "date": str(date.today()),
        "model": model,
        "opponent": opponent,
        "opponent_elo": opponent_elo,
        "games": games,
        **result,
        "win_rate": (result["win"] + 0.5 * result["draw"]) / games
    }
    if notes:
        entry["notes"] = notes
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load_results():
    results = []
    with open("results.jsonl") as f:
        for line in f:
            results.append(json.loads(line))
    return results
