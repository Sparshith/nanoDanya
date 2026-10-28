import json
import os
import sys
from pathlib import Path

import modal

if not modal.is_local():
    os.chdir("/root/project")
    sys.path.insert(0, "/root/project")

from modal_train import image, volume

app = modal.App("nanodanya-probes")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={"/data": volume}, timeout=600)
def probe_timeline(run: str, plies: list[int]):
    sys.path[:0] = ["/root/project", "/root/project/nanochat"]
    import torch
    from probes.inspect_game import inspect_timeline

    torch.set_num_threads(4)
    directory = Path(run)
    original = json.loads((directory / "report.json").read_text())
    reports = inspect_timeline(
        original["model"], (directory / "game.pgn").read_text(), plies,
        directory / "heads", directory,
    )
    volume.commit()
    return reports


@app.local_entrypoint()
def timeline(run: str, plies: str, out: str = "data/probes/game/timeline.json"):
    reports = probe_timeline.remote(run, [int(p) for p in plies.split(",")])
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(reports, indent=2) + "\n")
    print(f"Timeline: {destination}")
    for report in reports:
        layer = next(row for row in report["layers"] if row["layer"] == report["selected_layer"])
        print(f"Ply {report['ply']}, before {report['next_move']}: "
              f"{layer['correct_occupied']}/{layer['occupied_squares']} occupied squares")


@app.function(image=image, gpu="T4", cpu=4, memory=16384, volumes={"/data": volume}, timeout=3600)
def probe_game(pgn: str, ply: int, model: str, dataset: str, max_games: int, layers: str, run_name: str):
    from types import SimpleNamespace

    import torch

    os.chdir("/root/project")
    sys.path[:0] = ["/root/project", "/root/project/nanochat"]
    from probes.build_positions import build_positions
    from probes.cache_hiddens import cache_hiddens
    from probes.inspect_game import inspect_game
    from probes.run import run_layer, split_indices

    torch.set_num_threads(4)
    torch.manual_seed(0)
    out = Path("/data/probes") / run_name
    out.mkdir(parents=True, exist_ok=False)
    positions_path = out / "positions.pt"
    labels = build_positions(SimpleNamespace(
        data=f"/data/{dataset}/val.bin", meta=f"/data/{dataset}/meta.pkl",
        max_games=max_games, max_positions=0, stockfish="", stockfish_depth=8,
    ))
    torch.save(labels, positions_path)
    cached = cache_hiddens(SimpleNamespace(
        model=model, positions=str(positions_path), layers=layers, batch_size=16,
    ))
    torch.save(cached, out / "hiddens.pt")
    train_idx, val_idx = split_indices(labels, 0.8, 0)
    args = SimpleNamespace(
        epochs=20, batch_size=256, lr=1e-3, seed=0, positions=str(positions_path),
        heads_out=str(out / "heads"), pieces_only=True,
    )
    rows = []
    for layer, h in cached["hiddens"].items():
        print(f"Training board readout, layer {layer}", flush=True)
        run_layer(args, rows, model, layer, h, labels, train_idx, val_idx)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    report = inspect_game(model, pgn, ply, out / "heads", out)
    report["protocol"] = {
        "dataset": dataset, "max_games": max_games, "positions": len(labels["prefixes"]),
        "train_positions": len(train_idx), "validation_positions": len(val_idx),
        "epochs": 20, "seed": 0, "volume_directory": str(out),
        "layer_selection": "Highest occupied-square accuracy on held-out validation games",
    }
    (out / "game.pgn").write_text(pgn)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return report


@app.local_entrypoint()
def main(pgn: str, ply: int, model: str, dataset: str = "datasets/games/15m",
         max_games: int = 100, layers: str = "all", out: str = "data/probes/game"):
    from datetime import datetime, timezone

    run_name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report = probe_game.remote(Path(pgn).read_text(), ply, model, dataset, max_games, layers, run_name)
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {directory}/report.json")
    for row in report["layers"]:
        print(f"Layer {row['layer']}: held-out occupied accuracy={row['validation']['occupied_acc']:.3f}, "
              f"game occupied={row['correct_occupied']}/{row['occupied_squares']}, "
              f"all={row['correct_squares']}/64")
