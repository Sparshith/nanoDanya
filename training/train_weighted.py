import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_checkpoint(model, optimizer, config, meta, step, best_metric, best_step, no_improve_evals):
    raw_sd = model.state_dict()
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
    return {
        "model": clean_sd,
        "meta": {"model_config": config.__dict__, "tokenizer": meta},
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_metric": best_metric,
        "best_step": best_step,
        "no_improve_evals": no_improve_evals,
    }


dataset_dir_name = os.getenv("DATASET_DIR", "puzzle_weighted").strip() or "puzzle_weighted"
data_dir = Path(__file__).parent.parent / "data" / dataset_dir_name
meta = pickle.loads((data_dir / "meta.pkl").read_bytes())
vocab_size = meta["vocab_size"]
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


def nanogpt_iter(data, weights, block_size, batch_size):
    max_start = data.size(0) - block_size - 1
    idx = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    w = torch.stack([weights[i + 1 : i + block_size + 1] for i in idx])
    return x, y, w


device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

if device.type == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

config = GPTConfig(sequence_len=context_length, vocab_size=vocab_size, n_layer=12, n_head=6, n_kv_head=6, n_embd=768)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

default_ckpt_name = f"chess_puzzle_weighted_L{config.n_layer}_H{config.n_head}_E{config.n_embd}.pt"
ckpt_name = os.getenv("CKPT_NAME", default_ckpt_name)
best_ckpt_name = ckpt_name.replace(".pt", "_best.pt")
ckpt_path = Path(f"/data/{dataset_dir_name}/{ckpt_name}")
best_ckpt_path = Path(f"/data/{dataset_dir_name}/{best_ckpt_name}")

max_iters = int(os.getenv("MAX_ITERS", "100000"))
eval_interval = int(os.getenv("EVAL_INTERVAL", "100"))
ckpt_interval = int(os.getenv("CKPT_INTERVAL", "1000"))
batch_size = int(os.getenv("BATCH_SIZE", "64"))
val_batches = int(os.getenv("VAL_BATCHES", "1"))
early_stop_metric = os.getenv("EARLY_STOP_METRIC", "uniform").strip().lower()
early_stop_patience = int(os.getenv("EARLY_STOP_PATIENCE", "0"))
early_stop_min_steps = int(os.getenv("EARLY_STOP_MIN_STEPS", "0"))
early_stop_min_delta = float(os.getenv("EARLY_STOP_MIN_DELTA", "0.0"))

if early_stop_metric not in {"uniform", "weighted"}:
    raise ValueError(f"Unsupported EARLY_STOP_METRIC={early_stop_metric!r}")

print(
    f"run config: ckpt_name={ckpt_name} max_iters={max_iters} eval_interval={eval_interval} "
    f"ckpt_interval={ckpt_interval} batch_size={batch_size} val_batches={val_batches} "
    f"early_stop_metric={early_stop_metric} early_stop_patience={early_stop_patience} "
    f"early_stop_min_steps={early_stop_min_steps} early_stop_min_delta={early_stop_min_delta}"
)

start_step = 1
best_metric = float("inf")
best_step = 0
no_improve_evals = 0
if ckpt_path.exists():
    state = torch.load(ckpt_path, map_location=device)
    saved_config = state.get("meta", {}).get("model_config", {})
    if saved_config.get("n_layer") == config.n_layer and saved_config.get("n_embd") == config.n_embd:
        sd = {k.replace("_orig_mod.", ""): v for k, v in state["model"].items()}
        model.load_state_dict(sd)
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"] + 1
        best_metric = state.get("best_metric", float("inf"))
        best_step = state.get("best_step", 0)
        no_improve_evals = state.get("no_improve_evals", 0)
        print(
            f"Loaded checkpoint from {ckpt_path} at step {start_step} "
            f"(best_metric={best_metric:.4f}, best_step={best_step}, no_improve_evals={no_improve_evals})"
        )
    else:
        print("Config mismatch, starting fresh")
else:
    print("No checkpoint found, starting fresh")

if device.type == "cuda":
    model = torch.compile(model)

train_data = torch.from_numpy(train_tokens.astype(np.int64))
val_data = torch.from_numpy(val_tokens.astype(np.int64))
train_w = torch.from_numpy(train_weights)
val_w = torch.from_numpy(val_weights)
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

for step in range(start_step, max_iters + 1):
    xb_cpu, yb_cpu, wb_cpu = nanogpt_iter(train_data, train_w, context_length, batch_size)
    xb = xb_cpu.to(device, non_blocking=True)
    yb = yb_cpu.to(device, non_blocking=True)
    wb = wb_cpu.to(device, non_blocking=True)
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
        logits = model(xb)
        per_token = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction="none")
        loss = (per_token * wb.view(-1)).sum() / wb.sum()
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    should_stop = False
    if step % eval_interval == 0 or step == 1:
        val_loss_sum = 0.0
        val_loss_uniform_sum = 0.0
        for _ in range(val_batches):
            xb_val_cpu, yb_val_cpu, wb_val_cpu = nanogpt_iter(val_data, val_w, context_length, batch_size)
            xb_val = xb_val_cpu.to(device, non_blocking=True)
            yb_val = yb_val_cpu.to(device, non_blocking=True)
            wb_val = wb_val_cpu.to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                logits_val = model(xb_val)
                per_token_val = F.cross_entropy(logits_val.view(-1, logits_val.size(-1)), yb_val.view(-1), reduction="none")
                val_loss_batch = (per_token_val * wb_val.view(-1)).sum() / wb_val.sum()
                val_loss_uniform_batch = per_token_val.mean()
            val_loss_sum += val_loss_batch.item()
            val_loss_uniform_sum += val_loss_uniform_batch.item()

        val_loss = val_loss_sum / val_batches
        val_loss_uniform = val_loss_uniform_sum / val_batches
        metric_value = val_loss_uniform if early_stop_metric == "uniform" else val_loss
        improved = metric_value < (best_metric - early_stop_min_delta)
        if improved:
            best_metric = metric_value
            best_step = step
            no_improve_evals = 0
            best_checkpoint = build_checkpoint(model, optimizer, config, meta, step, best_metric, best_step, no_improve_evals)
            torch.save(best_checkpoint, best_ckpt_path)
            print(f"New best {early_stop_metric} val {best_metric:.4f} at step {step}, saved {best_ckpt_name}")
        elif early_stop_patience > 0 and step >= early_stop_min_steps:
            no_improve_evals += 1

        weighted_frac = (wb_cpu > 1).float().mean().item()
        avg_w = wb_cpu.mean().item()
        print(
            f"step {step:04d}/{max_iters} train {loss.item():.4f} "
            f"val {val_loss:.4f} (uniform {val_loss_uniform:.4f}) "
            f"best_{early_stop_metric} {best_metric:.4f}@{best_step} no_improve {no_improve_evals} "
            f"batch_w {avg_w:.3f} w_frac {weighted_frac:.4f}"
        )

        if early_stop_patience > 0 and step >= early_stop_min_steps and no_improve_evals >= early_stop_patience:
            should_stop = True

    if step % ckpt_interval == 0 or should_stop:
        checkpoint = build_checkpoint(model, optimizer, config, meta, step, best_metric, best_step, no_improve_evals)
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint to {ckpt_name} at step {step}")

    if should_stop:
        print(
            f"Early stopping at step {step}: best_{early_stop_metric}={best_metric:.4f} "
            f"from step {best_step}, no improvement for {no_improve_evals} evals"
        )
        break
