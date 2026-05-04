import pickle
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig, norm

data_dir = Path(__file__).parent.parent / 'data/eval'
meta = pickle.loads((data_dir / 'meta.pkl').read_bytes())
vocab_size = meta['vocab_size']
context_length = meta['context_length']
train_tokens = np.fromfile(data_dir / 'train.bin', dtype=np.uint16)
val_tokens = np.fromfile(data_dir / 'val.bin', dtype=np.uint16)
train_evals_raw = np.fromfile(data_dir / 'train_evals.bin', dtype=np.float32)
val_evals_raw = np.fromfile(data_dir / 'val_evals.bin', dtype=np.float32)

assert len(train_tokens) == len(train_evals_raw)
assert len(val_tokens) == len(val_evals_raw)
print(f"train: {len(train_tokens)} tokens, val: {len(val_tokens)} tokens")

# normalize evals: clip to [-1500, 1500], scale to [-1, 1], keep NaN as-is
def normalize_evals(raw):
    e = raw.copy()
    mask = ~np.isnan(e)
    e[mask] = np.clip(e[mask], -1500, 1500) / 1500.0
    return e

train_evals = normalize_evals(train_evals_raw)
val_evals = normalize_evals(val_evals_raw)
nan_frac = np.isnan(train_evals).sum() / len(train_evals)
print(f"train eval NaN fraction: {nan_frac:.4f}")


def nanogpt_iter(data, evals, block_size, batch_size):
    max_start = data.size(0) - block_size - 1
    idx = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y_move = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    y_eval = torch.stack([evals[i : i + block_size] for i in idx])
    return x, y_move, y_eval


device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

if device.type == 'cuda':
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

config = GPTConfig(sequence_len=context_length, vocab_size=vocab_size, n_layer=12, n_head=6, n_kv_head=6, n_embd=768)
model = GPT(config).to(device)

# eval head: predict scalar eval from hidden states
eval_head = nn.Linear(config.n_embd, 1, bias=False).to(device)
nn.init.zeros_(eval_head.weight)

# load pretrained checkpoint
pretrained_path = Path(__file__).parent.parent / 'models' / 'chess_puzzle_weighted_L12_H6_E768.pt'
ckpt_name = f"chess_weighted_eval_ft_L{config.n_layer}_H{config.n_head}_E{config.n_embd}.pt"
resume_optimizer = None

ckpt_path = Path(f"/data/eval/{ckpt_name}")
if ckpt_path.exists():
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_config = state.get('meta', {}).get('model_config', {})
    if saved_config.get('n_layer') == config.n_layer and saved_config.get('n_embd') == config.n_embd:
        sd = {k.replace('_orig_mod.', ''): v for k, v in state['model'].items()}
        model.load_state_dict(sd)
        if 'eval_head' in state:
            eval_head.load_state_dict(state['eval_head'])
        resume_optimizer = state.get('optimizer')
        start_step = state['step'] + 1
        print(f"Resumed from {ckpt_path} at step {start_step}")
    else:
        print("Config mismatch in eval checkpoint, loading pretrained instead")
        ckpt_path = None
else:
    ckpt_path = None

if ckpt_path is None:
    start_step = 1
    for path in (pretrained_path, Path("/data/puzzle_weighted/chess_puzzle_weighted_L12_H6_E768.pt")):
        if path.exists():
            state = torch.load(path, map_location=device, weights_only=False)
            sd = {k.replace('_orig_mod.', ''): v for k, v in state['model'].items()}
            model.load_state_dict(sd)
            print(f"Loaded pretrained from {path}")
            break
    else:
        raise FileNotFoundError("No pretrained checkpoint found")

# freeze: wte + layers 0-8, unfreeze: layers 9-11 + lm_head + eval_head
for p in model.parameters():
    p.requires_grad = False
for i in [9, 10, 11]:
    for p in model.transformer.h[i].parameters():
        p.requires_grad = True
for p in model.lm_head.parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in eval_head.parameters())
total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in eval_head.parameters())
print(f"Trainable: {trainable/1e6:.1f}M / {total_params/1e6:.1f}M params")

all_params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(eval_head.parameters())
optimizer = torch.optim.AdamW(all_params, lr=1e-4)

if resume_optimizer is not None:
    optimizer.load_state_dict(resume_optimizer)

train_data = torch.from_numpy(train_tokens.astype(np.int64))
val_data = torch.from_numpy(val_tokens.astype(np.int64))
train_ev = torch.from_numpy(train_evals)
val_ev = torch.from_numpy(val_evals)

max_iters = 100000
eval_interval = 100
batch_size = 64
lambda_eval = 1.0
use_amp = device.type == 'cuda'
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
softcap = 15

for step in range(start_step, max_iters + 1):
    xb_cpu, yb_cpu, eb_cpu = nanogpt_iter(train_data, train_ev, context_length, batch_size)
    xb = xb_cpu.to(device, non_blocking=True)
    yb = yb_cpu.to(device, non_blocking=True)
    eb = eb_cpu.to(device, non_blocking=True)

    with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
        # forward through backbone (bypassing model.forward to get hidden states)
        B, T = xb.size()
        cos_sin = model.cos[:, :T], model.sin[:, :T]
        x = model.transformer.wte(xb)
        x = norm(x)
        for block in model.transformer.h:
            x = block(x, cos_sin, None)
        x = norm(x)

        # move prediction loss (shifted by 1)
        logits = model.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)
        logits = logits.float()
        move_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction='mean')

        # eval prediction loss (no shift, aligned with token position)
        eval_pred = eval_head(x).squeeze(-1)  # (B, T)
        eval_mask = ~torch.isnan(eb)
        if eval_mask.any():
            eval_loss = F.mse_loss(eval_pred[eval_mask], eb[eval_mask])
        else:
            eval_loss = torch.tensor(0.0, device=device)

        loss = move_loss + lambda_eval * eval_loss

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    scaler.step(optimizer)
    scaler.update()

    if step % eval_interval == 0 or step == 1:
        xb_val_cpu, yb_val_cpu, eb_val_cpu = nanogpt_iter(val_data, val_ev, context_length, batch_size)
        xb_val = xb_val_cpu.to(device, non_blocking=True)
        yb_val = yb_val_cpu.to(device, non_blocking=True)
        eb_val = eb_val_cpu.to(device, non_blocking=True)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            B, T = xb_val.size()
            cos_sin = model.cos[:, :T], model.sin[:, :T]
            x = model.transformer.wte(xb_val)
            x = norm(x)
            for block in model.transformer.h:
                x = block(x, cos_sin, None)
            x = norm(x)

            logits_val = model.lm_head(x)
            logits_val = softcap * torch.tanh(logits_val / softcap)
            logits_val = logits_val.float()
            val_move_loss = F.cross_entropy(logits_val.view(-1, logits_val.size(-1)), yb_val.view(-1), reduction='mean')

            eval_pred_val = eval_head(x).squeeze(-1)
            eval_mask_val = ~torch.isnan(eb_val)
            if eval_mask_val.any():
                val_eval_loss = F.mse_loss(eval_pred_val[eval_mask_val], eb_val[eval_mask_val])
            else:
                val_eval_loss = torch.tensor(0.0, device=device)

        print(f'step {step:05d}/{max_iters} train_move {move_loss.item():.4f} train_eval {eval_loss.item():.4f} val_move {val_move_loss.item():.4f} val_eval {val_eval_loss.item():.4f}')

    if step % 1000 == 0:
        raw_sd = model.state_dict()
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in raw_sd.items()}
        eval_head_sd = {k.replace('_orig_mod.', ''): v for k, v in eval_head.state_dict().items()}
        checkpoint = {
            'model': clean_sd,
            'eval_head': eval_head_sd,
            'meta': {'model_config': config.__dict__, 'tokenizer': meta},
            'optimizer': optimizer.state_dict(),
            'step': step,
        }
        torch.save(checkpoint, f'/data/eval/{ckpt_name}')
        print(f'Saved checkpoint to {ckpt_name} at step {step}')
