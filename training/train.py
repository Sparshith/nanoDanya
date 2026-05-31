import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig


CONFIG_KEYS = ("sequence_len", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd")


def config_mismatches(saved_config, config):
    current_config = config.__dict__
    return {
        key: (saved_config.get(key), current_config[key])
        for key in CONFIG_KEYS
        if saved_config.get(key) != current_config[key]
    }


def arch_name(config):
    if config.n_kv_head == config.n_head:
        return f"L{config.n_layer}_H{config.n_head}_E{config.n_embd}"
    return f"L{config.n_layer}_H{config.n_head}_KV{config.n_kv_head}_E{config.n_embd}"


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


dataset_dir_name = os.getenv("DATASET_DIR", "processed").strip() or "processed"
data_dir = Path(__file__).parent.parent / "data" / dataset_dir_name
meta = pickle.loads((data_dir / "meta.pkl").read_bytes())
vocab_size = meta["vocab_size"]
context_length = meta["context_length"]
train_tokens = np.fromfile(data_dir / "train.bin", dtype=np.uint16)
val_tokens = np.fromfile(data_dir / "val.bin", dtype=np.uint16)
print(f"dataset: {dataset_dir_name}")
print(f"train: {len(train_tokens):,} tokens, val: {len(val_tokens):,} tokens")


def nanogpt_iter(data, block_size, batch_size):
    max_start = data.size(0) - block_size - 1
    idx = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    return x, y


device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

config = GPTConfig(
    sequence_len=context_length,
    vocab_size=vocab_size,
    n_layer=int(os.getenv("N_LAYER", "12")),
    n_head=int(os.getenv("N_HEAD", "6")),
    n_kv_head=int(os.getenv("N_KV_HEAD", os.getenv("N_HEAD", "6"))),
    n_embd=int(os.getenv("N_EMBD", "768")),
)
model = GPT(config).to(device)
lr = float(os.getenv("LR", "1e-4"))
weight_decay = float(os.getenv("WEIGHT_DECAY", "0.0"))
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

default_ckpt_name = (
    f"chess_{arch_name(config)}.pt"
    if dataset_dir_name == "processed"
    else f"chess_{dataset_dir_name}_{arch_name(config)}.pt"
)
ckpt_name = os.getenv("CKPT_NAME", default_ckpt_name)
best_ckpt_name = ckpt_name.replace(".pt", "_best.pt")
if dataset_dir_name == "processed" and "CKPT_NAME" not in os.environ:
    ckpt_path = Path(f"/data/{ckpt_name}")
    best_ckpt_path = Path(f"/data/{best_ckpt_name}")
else:
    ckpt_path = Path("/data") / dataset_dir_name / ckpt_name
    best_ckpt_path = Path("/data") / dataset_dir_name / best_ckpt_name
ckpt_path.parent.mkdir(parents=True, exist_ok=True)

max_iters = int(os.getenv("MAX_ITERS", "50000"))
eval_interval = int(os.getenv("EVAL_INTERVAL", "100"))
ckpt_interval = int(os.getenv("CKPT_INTERVAL", "1000"))
batch_size = int(os.getenv("BATCH_SIZE", "32"))
eval_batch_size = int(os.getenv("EVAL_BATCH_SIZE", str(batch_size)))
grad_accum_steps = int(os.getenv("GRAD_ACCUM_STEPS", "1"))
val_batches = int(os.getenv("VAL_BATCHES", "1"))
early_stop_patience = int(os.getenv("EARLY_STOP_PATIENCE", "0"))
early_stop_min_steps = int(os.getenv("EARLY_STOP_MIN_STEPS", "0"))
early_stop_min_delta = float(os.getenv("EARLY_STOP_MIN_DELTA", "0.0"))

if batch_size <= 0 or eval_batch_size <= 0 or grad_accum_steps <= 0:
    raise ValueError("BATCH_SIZE, EVAL_BATCH_SIZE, and GRAD_ACCUM_STEPS must be positive")

print(
    f"run config: ckpt_name={ckpt_name} max_iters={max_iters} eval_interval={eval_interval} "
    f"ckpt_interval={ckpt_interval} batch_size={batch_size} eval_batch_size={eval_batch_size} "
    f"grad_accum_steps={grad_accum_steps} lr={lr:g} weight_decay={weight_decay:g} val_batches={val_batches} "
    f"arch={arch_name(config)} "
    f"early_stop_patience={early_stop_patience} early_stop_min_steps={early_stop_min_steps} "
    f"early_stop_min_delta={early_stop_min_delta}"
)

start_step = 1
best_metric = float("inf")
best_step = 0
no_improve_evals = 0
if ckpt_path.exists():
    state = torch.load(ckpt_path, map_location=device)
    saved_config = state.get("meta", {}).get("model_config", {})
    mismatches = config_mismatches(saved_config, config)
    if not mismatches:
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
        raise ValueError(f"checkpoint config mismatch in {ckpt_path}: {mismatches}")
else:
    print("No checkpoint found, starting fresh")

if device.type == "cuda":
    model = torch.compile(model)

train_data = torch.from_numpy(train_tokens.astype(np.int64))
val_data = torch.from_numpy(val_tokens.astype(np.int64))
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

for step in range(start_step, max_iters + 1):
    optimizer.zero_grad(set_to_none=True)
    train_loss_sum = 0.0
    for _ in range(grad_accum_steps):
        xb_cpu, yb_cpu = nanogpt_iter(train_data, context_length, batch_size)
        xb = xb_cpu.to(device, non_blocking=True)
        yb = yb_cpu.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        train_loss_sum += loss.item()

    train_loss = train_loss_sum / grad_accum_steps
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    should_stop = False
    if step % eval_interval == 0 or step == 1:
        val_loss_sum = 0.0
        for _ in range(val_batches):
            xb_val_cpu, yb_val_cpu = nanogpt_iter(val_data, context_length, eval_batch_size)
            xb_val = xb_val_cpu.to(device, non_blocking=True)
            yb_val = yb_val_cpu.to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                logits_val = model(xb_val)
                val_loss_batch = F.cross_entropy(logits_val.view(-1, logits_val.size(-1)), yb_val.view(-1))
            val_loss_sum += val_loss_batch.item()

        val_loss = val_loss_sum / val_batches
        improved = val_loss < (best_metric - early_stop_min_delta)
        if improved:
            best_metric = val_loss
            best_step = step
            no_improve_evals = 0
            best_checkpoint = build_checkpoint(model, optimizer, config, meta, step, best_metric, best_step, no_improve_evals)
            torch.save(best_checkpoint, best_ckpt_path)
            print(f"New best val {best_metric:.4f} at step {step}, saved {best_ckpt_name}")
        elif early_stop_patience > 0 and step >= early_stop_min_steps:
            no_improve_evals += 1

        print(
            f"step {step:04d}/{max_iters} train {train_loss:.4f} val {val_loss:.4f} "
            f"best {best_metric:.4f}@{best_step} no_improve {no_improve_evals}"
        )

        if early_stop_patience > 0 and step >= early_stop_min_steps and no_improve_evals >= early_stop_patience:
            should_stop = True

    if step % ckpt_interval == 0 or should_stop:
        checkpoint = build_checkpoint(model, optimizer, config, meta, step, best_metric, best_step, no_improve_evals)
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path} at step {step}")

    if should_stop:
        print(f"Early stopping at step {step}: best={best_metric:.4f} from step {best_step}")
        break
