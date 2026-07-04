import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT

from common import arch_name, config_from_env, get_device, sample_batch, train_loop

dataset_dir_name = os.getenv("DATASET_DIR", "processed").strip() or "processed"
data_dir = Path(__file__).parent.parent / "data" / dataset_dir_name
meta = pickle.loads((data_dir / "meta.pkl").read_bytes())
context_length = meta["context_length"]
train_data = torch.from_numpy(np.fromfile(data_dir / "train.bin", dtype=np.uint16).astype(np.int64))
val_data = torch.from_numpy(np.fromfile(data_dir / "val.bin", dtype=np.uint16).astype(np.int64))
print(f"dataset: {dataset_dir_name}")
print(f"train: {len(train_data):,} tokens, val: {len(val_data):,} tokens")

device = get_device()
config = config_from_env(context_length, meta["vocab_size"])
model = GPT(config).to(device)
lr = float(os.getenv("LR", "1e-4"))
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
print(f"arch {arch_name(config)} lr {lr:g}")

default_ckpt_name = (
    f"chess_{arch_name(config)}.pt"
    if dataset_dir_name == "processed"
    else f"chess_{dataset_dir_name}_{arch_name(config)}.pt"
)
ckpt_name = os.getenv("CKPT_NAME", default_ckpt_name)
if os.getenv("CKPT_DIR"):
    ckpt_dir = Path(os.environ["CKPT_DIR"])
elif dataset_dir_name == "processed" and "CKPT_NAME" not in os.environ:
    ckpt_dir = Path("/data")
else:
    ckpt_dir = Path("/data") / dataset_dir_name
ckpt_dir.mkdir(parents=True, exist_ok=True)

batch_size = int(os.getenv("BATCH_SIZE", "32"))


def train_step(m, xb, yb):
    logits = m(xb)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))


def val_step(m, xb, yb):
    return {"val": train_step(m, xb, yb).item()}


train_loop(
    model, optimizer, config, meta, device,
    ckpt_path=ckpt_dir / ckpt_name,
    best_ckpt_path=ckpt_dir / ckpt_name.replace(".pt", "_best.pt"),
    sample_train=lambda: sample_batch(train_data, context_length, batch_size),
    sample_val=lambda: sample_batch(val_data, context_length, batch_size),
    train_step=train_step,
    val_step=val_step,
    metric_key="val",
    max_iters=int(os.getenv("MAX_ITERS", "50000")),
    grad_accum_steps=int(os.getenv("GRAD_ACCUM_STEPS", "1")),
)
