from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_head(head, x_train, y_train, loss_fn, *, epochs: int, batch_size: int, lr: float):
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    n = x_train.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            loss = loss_fn(head(x_train[idx]), y_train(idx) if callable(y_train) else y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


def legal_dense(legal_lists: list[list[int]], vocab_size: int, idx) -> torch.Tensor:
    idx_list = idx.tolist() if isinstance(idx, torch.Tensor) else list(idx)
    y = torch.zeros((len(idx_list), vocab_size), dtype=torch.float32)
    for row, pos in enumerate(idx_list):
        ids = legal_lists[int(pos)]
        if ids:
            y[row, ids] = 1.0
    return y


def split_indices(labels: dict, train_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(labels["prefixes"])
    if "game_id" in labels:
        game_ids = labels["game_id"]
        unique_games = torch.unique(game_ids)
        gen = torch.Generator().manual_seed(seed)
        shuffled = unique_games[torch.randperm(len(unique_games), generator=gen)]
        cut = max(1, int(len(shuffled) * train_frac))
        if cut >= len(shuffled) and len(shuffled) > 1:
            cut = len(shuffled) - 1
        train_games = set(shuffled[:cut].tolist())
        train_mask = torch.tensor([int(g) in train_games for g in game_ids.tolist()])
        train_idx = torch.nonzero(train_mask, as_tuple=False).flatten()
        val_idx = torch.nonzero(~train_mask, as_tuple=False).flatten()
        if len(train_idx) and len(val_idx):
            return train_idx, val_idx

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    cut = int(n * train_frac)
    return perm[:cut], perm[cut:]


def majority_binary(train_y: torch.Tensor, val_y: torch.Tensor) -> float:
    label = float(train_y.mean().item() >= 0.5)
    return float((val_y == label).float().mean().item())


def legal_frequency_baseline(train_lists: list[list[int]], val_lists: list[list[int]], vocab_size: int) -> float:
    counts = np.zeros(vocab_size, dtype=np.int64)
    total = 0
    for ids in train_lists:
        counts[ids] += 1
        total += len(ids)
    k = max(1, int(round(total / max(len(train_lists), 1))))
    counts[:2] = -1
    pred = set(np.argpartition(counts, -k)[-k:].tolist())
    tp = sum(sum(i in pred for i in ids) for ids in val_lists)
    pp = len(pred) * len(val_lists)
    ap = sum(len(ids) for ids in val_lists)
    p = tp / pp if pp else 0.0
    r = tp / ap if ap else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def add_metric(rows, model, layer, task, metric, value, baseline):
    rows.append({
        "model": model,
        "layer": layer,
        "task": task,
        "metric": metric,
        "value": float(value),
        "baseline": float(baseline) if baseline is not None else "",
    })


def run_layer(args, rows, model_name, layer, h, labels, train_idx, val_idx):
    x_train = h[train_idx]
    x_val = h[val_idx]
    n_embd = h.shape[1]
    vocab_size = labels["vocab_size"]

    pieces = labels["pieces"]
    train_pieces = pieces[train_idx]
    val_pieces = pieces[val_idx]
    head = train_head(
        nn.Linear(n_embd, 64 * 13),
        x_train,
        train_pieces,
        lambda out, y: F.cross_entropy(out.view(-1, 13), y.view(-1)),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    with torch.no_grad():
        pred = head(x_val).view(-1, 64, 13).argmax(dim=2)
        all_acc = float((pred == val_pieces).float().mean().item())
        occ = val_pieces != 0
        occ_acc = float((pred[occ] == val_pieces[occ]).float().mean().item()) if occ.any() else math.nan
    piece_counts = torch.bincount(train_pieces.view(-1), minlength=13)
    piece_base = float((val_pieces == int(piece_counts.argmax())).float().mean().item())
    occ_train = train_pieces[train_pieces != 0]
    if occ_train.numel() and occ.any():
        occ_counts = torch.bincount(occ_train, minlength=13)
        occ_counts[0] = -1
        occ_base = float((val_pieces[occ] == int(occ_counts.argmax())).float().mean().item())
    else:
        occ_base = math.nan
    add_metric(rows, model_name, layer, "pieces", "all_acc", all_acc, piece_base)
    add_metric(rows, model_name, layer, "pieces", "occupied_acc", occ_acc, occ_base)

    for task in ("side_to_move", "in_check"):
        y = labels[task]
        head = train_head(
            nn.Linear(n_embd, 1),
            x_train,
            y[train_idx],
            lambda out, target: F.binary_cross_entropy_with_logits(out, target),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        with torch.no_grad():
            acc = float(((head(x_val).sigmoid() > 0.5).float() == y[val_idx]).float().mean().item())
        base = majority_binary(y[train_idx], y[val_idx])
        add_metric(rows, model_name, layer, task, "acc", acc, base)

    legal = labels["legal_move_ids"]
    train_legal = [legal[int(i)] for i in train_idx.tolist()]
    val_legal = [legal[int(i)] for i in val_idx.tolist()]
    head = train_head(
        nn.Linear(n_embd, vocab_size),
        x_train,
        lambda idx: legal_dense(train_legal, vocab_size, idx),
        lambda out, target: F.binary_cross_entropy_with_logits(out, target),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    with torch.no_grad():
        pred = (head(x_val).sigmoid() > 0.5).float()
        target = legal_dense(val_legal, vocab_size, range(len(val_legal)))
        tp = float((pred * target).sum().item())
        pp = float(pred.sum().item())
        ap = float(target.sum().item())
        precision = tp / pp if pp else 0.0
        recall = tp / ap if ap else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    base_f1 = legal_frequency_baseline(train_legal, val_legal, vocab_size)
    add_metric(rows, model_name, layer, "legal_moves", "f1", f1, base_f1)

    eval_cp = labels.get("eval_cp")
    if eval_cp is not None:
        finite = torch.isfinite(eval_cp)
        mask_train = finite[train_idx]
        mask_val = finite[val_idx]
        if mask_train.any() and mask_val.any():
            train_cp = torch.clamp(eval_cp[train_idx][mask_train], -1500, 1500) / 1500.0
            val_cp = torch.clamp(eval_cp[val_idx][mask_val], -1500, 1500) / 1500.0
            x_train_eval = x_train[mask_train]
            x_val_eval = x_val[mask_val]
            train_bucket = eval_bucket(train_cp)
            val_bucket = eval_bucket(val_cp)
            head = train_head(
                nn.Linear(n_embd, 5),
                x_train_eval,
                train_bucket,
                lambda out, target: F.cross_entropy(out, target),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
            with torch.no_grad():
                acc = float((head(x_val_eval).argmax(dim=1) == val_bucket).float().mean().item())
            base = float((val_bucket == int(torch.bincount(train_bucket, minlength=5).argmax())).float().mean().item())
            add_metric(rows, model_name, layer, "stockfish_eval", "bucket_acc", acc, base)


def eval_bucket(cp_norm: torch.Tensor) -> torch.Tensor:
    cp = cp_norm * 1500.0
    y = torch.zeros_like(cp, dtype=torch.long)
    y[(cp >= -200) & (cp < -50)] = 1
    y[(cp >= -50) & (cp <= 50)] = 2
    y[(cp > 50) & (cp <= 200)] = 3
    y[cp > 200] = 4
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear probes from cached hidden states.")
    parser.add_argument("--positions", default="data/probes/positions_val_small.pt")
    parser.add_argument("--hiddens", default="data/probes/hiddens_baseline_l12_val_small.pt")
    parser.add_argument("--out", default="data/probes/probe_metrics.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    labels = torch.load(args.positions, map_location="cpu", weights_only=False)
    cached = torch.load(args.hiddens, map_location="cpu", weights_only=False)
    train_idx, val_idx = split_indices(labels, args.train_frac, args.seed)

    rows = []
    for layer, h in cached["hiddens"].items():
        print(f"layer {layer}")
        run_layer(args, rows, cached["model"], layer, h, labels, train_idx, val_idx)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "layer", "task", "metric", "value", "baseline"])
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['model']} layer={row['layer']} {row['task']}.{row['metric']}="
            f"{row['value']:.4f} baseline={row['baseline']}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
