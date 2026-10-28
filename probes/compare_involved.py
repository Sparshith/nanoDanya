"""Independent frozen-model board readouts, with no empty-square accuracy score."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from probes.cache_hiddens import cache_hiddens
from probes.inspect_game import inspect_timeline
from probes.run import split_indices, train_head


def piece_metrics(pred, truth):
    occupied = truth != 0
    predicted = pred != 0
    correct = (pred == truth) & occupied
    recovered = int(correct.sum())
    actual_count, predicted_count = int(occupied.sum()), int(predicted.sum())
    return {
        "recovered": recovered, "actual_pieces": actual_count, "predicted_pieces": predicted_count,
        "recall": recovered / actual_count if actual_count else None,
        "precision": recovered / predicted_count if predicted_count else 0.0,
        "missing": int((occupied & ~predicted).sum()),
        "wrong_piece": int((occupied & predicted & ~correct).sum()),
        "invented": int((~occupied & predicted).sum()),
        "exact_positions": int((pred == truth).all(-1).sum()),
        "positions": len(truth),
    }


def compare_model(model_path, positions_path, pgn, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    heads = output / "heads"
    heads.mkdir(exist_ok=True)
    labels = torch.load(positions_path, map_location="cpu", weights_only=False)
    train_idx, val_idx = split_indices(labels, 0.8, 0)
    cached = cache_hiddens(SimpleNamespace(model=model_path, positions=str(positions_path),
                                          layers="all", batch_size=32))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_pieces, val_pieces = labels["pieces"][train_idx].to(device), labels["pieces"][val_idx].to(device)
    validation = []
    for layer, hidden in cached["hiddens"].items():
        torch.manual_seed(layer)
        x_train, x_val = hidden[train_idx].to(device), hidden[val_idx].to(device)
        head = train_head(nn.Linear(hidden.shape[1], 64 * 13).to(device), x_train, train_pieces,
                          lambda out, y: F.cross_entropy(out.view(-1, 13), y.view(-1)),
                          epochs=20, batch_size=256, lr=1e-3)
        with torch.no_grad():
            pred = head(x_val).view(-1, 64, 13).argmax(-1)
            metrics = piece_metrics(pred, val_pieces)
        torch.save({"model": model_path, "layer": layer,
                    "state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                    "metrics": {"occupied_acc": metrics["recall"]}}, heads / f"pieces_layer_{layer}.pt")
        validation.append({"layer": layer, **metrics})
        print(json.dumps({"model": model_path, "layer": layer, "validation": metrics}), flush=True)
        del head, x_train, x_val
    del cached
    plies = list(range(1, 72, 2))
    reports = inspect_timeline(model_path, pgn, plies, heads, output)
    timeline = []
    truth_all, prediction_all = [], []
    symbols = ".PNBRQKpnbrqk"
    for report in reports:
        selected = next(row for row in report["layers"] if row["layer"] == report["selected_layer"])
        truth = torch.tensor([[symbols.index(row["actual"]) for row in selected["squares"]]])
        pred = torch.tensor([[symbols.index(row["predicted"]) for row in selected["squares"]]])
        truth_all.append(truth)
        prediction_all.append(pred)
        timeline.append({"ply": report["ply"], "next_move": report["next_move"],
                         "layer": report["selected_layer"], **piece_metrics(pred, truth),
                         "errors": [row for row in selected["squares"] if row["actual"] != row["predicted"]]})
    summary = {"model": model_path, "selected_layer": reports[0]["selected_layer"],
               "validation": validation, "timeline": timeline,
               "aggregate": piece_metrics(torch.cat(prediction_all), torch.cat(truth_all)),
               "before_g5": timeline[-1],
               "protocol": {"positions": len(labels["prefixes"]), "probe_game_split": "80/20, seed 0",
                            "epochs": 20, "batch_size": 256, "lr": 1e-3,
                            "head_seed": "layer number, identical across models",
                            "selection": "highest held-out occupied-square recall, never traced game",
                            "training": "frozen-model linear 64x13 readouts; no backbone updates",
                            "metrics": "correct colored pieces, missing pieces, wrong identities/types, invented pieces; no empty-square accuracy"}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
