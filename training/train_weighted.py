import pickle
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig

data_dir = Path(__file__).parent.parent / 'data/puzzle_weighted'
meta = pickle.loads((data_dir / 'meta.pkl').read_bytes())
vocab_size = meta['vocab_size']
context_length = meta['context_length']
train_tokens = np.fromfile(data_dir / 'train.bin', dtype=np.uint16)
val_tokens = np.fromfile(data_dir / 'val.bin', dtype=np.uint16)
train_weights = np.fromfile(data_dir / 'train_weights.bin', dtype=np.float32)
val_weights = np.fromfile(data_dir / 'val_weights.bin', dtype=np.float32)

assert len(train_tokens) == len(train_weights), f"token/weight mismatch: {len(train_tokens)} vs {len(train_weights)}"
assert len(val_tokens) == len(val_weights), f"token/weight mismatch: {len(val_tokens)} vs {len(val_weights)}"
print(f"train: {len(train_tokens)} tokens, val: {len(val_tokens)} tokens")
print(f"train weighted fraction: {(train_weights > 1).sum() / len(train_weights):.4f}")


def nanogpt_iter(data, weights, block_size, batch_size):
    max_start = data.size(0) - block_size - 1
    idx = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    w = torch.stack([weights[i + 1 : i + block_size + 1] for i in idx])
    return x, y, w


device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

# GPU optimizations
if device.type == 'cuda':
    torch.set_float32_matmul_precision('high')  # enable tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

config = GPTConfig(sequence_len=context_length, vocab_size=vocab_size, n_layer=12, n_head=6, n_kv_head=6, n_embd=768)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

ckpt_name = f"chess_weighted_L{config.n_layer}_H{config.n_head}_E{config.n_embd}.pt"

start_step = 1
for ckpt_path in (
    Path(f"/data/puzzle_weighted/{ckpt_name}"),
):
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        saved_config = state.get('meta', {}).get('model_config', {})
        if saved_config.get('n_layer') == config.n_layer and saved_config.get('n_embd') == config.n_embd:
            sd = {k.replace('_orig_mod.', ''): v for k, v in state['model'].items()}
            model.load_state_dict(sd)
            optimizer.load_state_dict(state['optimizer'])
            start_step = state['step'] + 1
            print(f"Loaded checkpoint from {ckpt_path}")
            break
        else:
            print("Config mismatch, starting fresh")
    else:
        print("No checkpoint found, starting fresh")

if device.type == 'cuda':
    model = torch.compile(model)

train_data = torch.from_numpy(train_tokens.astype(np.int64))
val_data = torch.from_numpy(val_tokens.astype(np.int64))
train_w = torch.from_numpy(train_weights)
val_w = torch.from_numpy(val_weights)
max_iters = 100000
eval_interval = 100
batch_size = 64
use_amp = device.type == 'cuda'
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

for step in range(start_step, max_iters + 1):
    xb_cpu, yb_cpu, wb_cpu = nanogpt_iter(train_data, train_w, context_length, batch_size)
    xb = xb_cpu.to(device, non_blocking=True)
    yb = yb_cpu.to(device, non_blocking=True)
    wb = wb_cpu.to(device, non_blocking=True)
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
        logits = model(xb)
        per_token = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction='none')
        loss = (per_token * wb.view(-1)).sum() / wb.sum()
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    if step % eval_interval == 0 or step == 1:
        xb_val_cpu, yb_val_cpu, wb_val_cpu = nanogpt_iter(val_data, val_w, context_length, batch_size)
        xb_val = xb_val_cpu.to(device, non_blocking=True)
        yb_val = yb_val_cpu.to(device, non_blocking=True)
        wb_val = wb_val_cpu.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            logits_val = model(xb_val)
            per_token_val = F.cross_entropy(logits_val.view(-1, logits_val.size(-1)), yb_val.view(-1), reduction='none')
            val_loss = (per_token_val * wb_val.view(-1)).sum() / wb_val.sum()
            val_loss_uniform = per_token_val.mean()
        weighted_frac = (wb_cpu > 1).float().mean().item()
        avg_w = wb_cpu.mean().item()
        print(f'step {step:04d}/{max_iters} train {loss.item():.4f} val {val_loss.item():.4f} (uniform {val_loss_uniform.item():.4f}) batch_w {avg_w:.3f} w_frac {weighted_frac:.4f}')

    if step % 1000 == 0:
        raw_sd = model.state_dict()
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in raw_sd.items()}
        checkpoint = {
            'model': clean_sd,
            'meta': {'model_config': config.__dict__, 'tokenizer': meta},
            'optimizer': optimizer.state_dict(),
            'step': step,
        }
        torch.save(checkpoint, f'/data/puzzle_weighted/{ckpt_name}')
        print(f'Saved checkpoint to {ckpt_name} at step {step}')
