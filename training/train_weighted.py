import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT

from common import arch_name, config_from_env, env_flag, get_device, sample_batch, train_loop

dataset_dir_name = os.getenv("DATASET_DIR", "puzzle_weighted").strip() or "puzzle_weighted"
data_dir = Path(__file__).parent.parent / "data" / dataset_dir_name
meta = pickle.loads((data_dir / "meta.pkl").read_bytes())
context_length = meta["context_length"]
train_tokens = np.fromfile(data_dir / "train.bin", dtype=np.uint16)
val_tokens = np.fromfile(data_dir / "val.bin", dtype=np.uint16)
train_weights = np.fromfile(data_dir / "train_weights.bin", dtype=np.float32)
val_weights = np.fromfile(data_dir / "val_weights.bin", dtype=np.float32)

use_uniform_weights = env_flag("UNIFORM_WEIGHTS")
if use_uniform_weights:
    train_weights = np.ones_like(train_weights)
    val_weights = np.ones_like(val_weights)

assert len(train_tokens) == len(train_weights), f"token/weight mismatch: {len(train_tokens)} vs {len(train_weights)}"
assert len(val_tokens) == len(val_weights), f"token/weight mismatch: {len(val_tokens)} vs {len(val_weights)}"
print(f"dataset: {dataset_dir_name}")
print(f"train: {len(train_tokens)} tokens, val: {len(val_tokens)} tokens")
print(f"uniform weights: {use_uniform_weights}")
print(f"train weighted fraction: {(train_weights > 1).sum() / len(train_weights):.4f}")

train_data = torch.from_numpy(train_tokens.astype(np.int64))
val_data = torch.from_numpy(val_tokens.astype(np.int64))
train_w = torch.from_numpy(train_weights)
val_w = torch.from_numpy(val_weights)

device = get_device()
config = config_from_env(context_length, meta["vocab_size"])
model = GPT(config).to(device)
lr = float(os.getenv("LR", "1e-4"))
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
print(f"arch {arch_name(config)} lr {lr:g}")

ckpt_name = os.getenv("CKPT_NAME", f"chess_puzzle_weighted_{arch_name(config)}.pt")
ckpt_dir = Path(os.getenv("CKPT_DIR", f"/data/{dataset_dir_name}"))
ckpt_dir.mkdir(parents=True, exist_ok=True)

batch_size = int(os.getenv("BATCH_SIZE", "64"))
metric_key = os.getenv("EARLY_STOP_METRIC", "uniform").strip().lower()
if metric_key not in {"uniform", "weighted"}:
    raise ValueError(f"Unsupported EARLY_STOP_METRIC={metric_key!r}")


def weighted_ce(m, xb, yb, wb):
    logits = m(xb)
    per_token = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction="none")
    return per_token, (per_token * wb.view(-1)).sum() / wb.sum()


def train_step(m, xb, yb, wb):
    return weighted_ce(m, xb, yb, wb)[1]


def val_step(m, xb, yb, wb):
    per_token, weighted = weighted_ce(m, xb, yb, wb)
    return {"weighted": weighted.item(), "uniform": per_token.mean().item()}


train_loop(
    model, optimizer, config, meta, device,
    ckpt_path=ckpt_dir / ckpt_name,
    best_ckpt_path=ckpt_dir / ckpt_name.replace(".pt", "_best.pt"),
    sample_train=lambda: sample_batch(train_data, context_length, batch_size, shifted=(train_w,)),
    sample_val=lambda: sample_batch(val_data, context_length, batch_size, shifted=(val_w,)),
    train_step=train_step,
    val_step=val_step,
    metric_key=metric_key,
    max_iters=int(os.getenv("MAX_ITERS", "100000")),
    grad_accum_steps=int(os.getenv("GRAD_ACCUM_STEPS", "1")),
)
