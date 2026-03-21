import pickle
import math
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.muon import Muon

data_dir = Path(__file__).parent.parent / 'data' / 'eval'
meta = pickle.loads((data_dir / 'meta.pkl').read_bytes())
vocab_size = meta['vocab_size']
context_length = meta['context_length']

train_tokens = np.fromfile(data_dir / 'train.bin', dtype=np.uint16)
val_tokens = np.fromfile(data_dir / 'val.bin', dtype=np.uint16)
train_evals_raw = np.fromfile(data_dir / 'train_evals.bin', dtype=np.float32)
val_evals_raw = np.fromfile(data_dir / 'val_evals.bin', dtype=np.float32)
train_weights = np.fromfile(data_dir / 'train_weights.bin', dtype=np.float32)
val_weights = np.fromfile(data_dir / 'val_weights.bin', dtype=np.float32)

assert len(train_tokens) == len(train_evals_raw) == len(train_weights)
assert len(val_tokens) == len(val_evals_raw) == len(val_weights)
print(f"train: {len(train_tokens):,} tokens, val: {len(val_tokens):,} tokens")


def normalize_evals(raw):
    e = raw.copy()
    mask = ~np.isnan(e)
    e[mask] = np.clip(e[mask], -1500, 1500) / 1500.0
    return e


train_evals = normalize_evals(train_evals_raw)
val_evals = normalize_evals(val_evals_raw)
nan_frac = np.isnan(train_evals).sum() / len(train_evals)
print(f"eval NaN fraction: {nan_frac:.4f}")
print(f"weight stats: mean={train_weights.mean():.3f} max={train_weights.max():.3f} frac>1={((train_weights > 1).sum() / len(train_weights)):.3f}")


def sample_batch(tokens, block_size, batch_size, weights, evals):
    max_start = tokens.size(0) - block_size - 1
    idx = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in idx])
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in idx])
    w = torch.stack([weights[i + 1 : i + block_size + 1] for i in idx])
    e = torch.stack([evals[i : i + block_size] for i in idx])
    return x, y, w, e


device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
if device.type == 'cuda':
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

config = GPTConfig(sequence_len=context_length, vocab_size=vocab_size, n_layer=12, n_head=6, n_kv_head=6, n_embd=768)
model = GPT(config).to(device)
model.init_weights()

eval_head = nn.Sequential(
    nn.Linear(config.n_embd, 256, bias=False),
    nn.ReLU(),
    nn.Linear(256, 1, bias=False),
).to(device)
nn.init.zeros_(eval_head[2].weight)

# Muon for transformer block weights, AdamW for embeddings + heads
lr_scale = (config.n_embd / 768) ** -0.5
adamw = torch.optim.AdamW([
    dict(params=list(model.lm_head.parameters()), lr=0.004 * lr_scale),
    dict(params=list(model.transformer.wte.parameters()), lr=0.2 * lr_scale),
    dict(params=list(eval_head.parameters()), lr=0.004 * lr_scale),
], betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
muon = Muon(model.transformer.h.parameters(), lr=0.02, momentum=0.95)
optimizers = [adamw, muon]
for opt in optimizers:
    for group in opt.param_groups:
        group['initial_lr'] = group['lr']

ckpt_name = f"chess_scratch_L{config.n_layer}_H{config.n_head}_E{config.n_embd}.pt"
start_step = 1
ckpt_path = Path(f"/data/eval/{ckpt_name}")
if ckpt_path.exists():
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = state.get('meta', {}).get('model_config', {})
    if saved_cfg.get('n_layer') == config.n_layer and saved_cfg.get('n_embd') == config.n_embd:
        sd = {k.replace('_orig_mod.', ''): v for k, v in state['model'].items()}
        model.load_state_dict(sd)
        if 'eval_head' in state:
            eval_head.load_state_dict(state['eval_head'])
        if 'optimizers' in state:
            for opt, opt_sd in zip(optimizers, state['optimizers']):
                opt.load_state_dict(opt_sd)
        start_step = state['step'] + 1
        best_val_loss = state.get('best_val_loss', float('inf'))
        print(f"Resumed from {ckpt_path} at step {start_step}, best_val_loss={best_val_loss:.4f}")
else:
    print("No checkpoint found, training from scratch")
    best_val_loss = float('inf')

max_iters = 100000
warmup_steps = 5000
batch_size = 64
grad_accum = 4
eval_interval = 100
lambda_eval = 10.0
softcap = 15
use_amp = device.type == 'cuda'


def get_lr_mult(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (max_iters - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


train_tok = torch.from_numpy(train_tokens.astype(np.int64))
val_tok = torch.from_numpy(val_tokens.astype(np.int64))
train_w = torch.from_numpy(train_weights)
val_w = torch.from_numpy(val_weights)
train_e = torch.from_numpy(train_evals)
val_e = torch.from_numpy(val_evals)


def forward_pass(xb, yb, wb, eb):
    B, T = xb.size()
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    x = model.transformer.wte(xb)
    x = norm(x)
    for block in model.transformer.h:
        x = block(x, cos_sin, None)
    x = norm(x)

    logits = model.lm_head(x)
    logits = softcap * torch.tanh(logits / softcap)
    logits = logits.float()

    per_token = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction='none')
    move_loss = (per_token * wb.view(-1)).sum() / wb.sum()

    eval_pred = torch.tanh(eval_head(x).squeeze(-1))
    eval_mask = ~torch.isnan(eb)
    if eval_mask.any():
        eval_loss = F.mse_loss(eval_pred[eval_mask], eb[eval_mask])
    else:
        eval_loss = torch.tensor(0.0, device=device)

    return move_loss, eval_loss, per_token.mean()


for step in range(start_step, max_iters + 1):
    lr_mult = get_lr_mult(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * lr_mult

    for opt in optimizers:
        opt.zero_grad()

    accum_move = 0.0
    accum_eval = 0.0
    for _ in range(grad_accum):
        xb, yb, wb, eb = sample_batch(train_tok, context_length, batch_size, train_w, train_e)
        xb, yb, wb, eb = xb.to(device), yb.to(device), wb.to(device), eb.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            move_loss, eval_loss, _ = forward_pass(xb, yb, wb, eb)
            loss = (move_loss + lambda_eval * eval_loss) / grad_accum

        loss.backward()
        accum_move += move_loss.item()
        accum_eval += eval_loss.item()

    all_params = list(model.parameters()) + list(eval_head.parameters())
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    for opt in optimizers:
        opt.step()

    if step % eval_interval == 0 or step == 1:
        xb_v, yb_v, wb_v, eb_v = sample_batch(val_tok, context_length, batch_size, val_w, val_e)
        xb_v, yb_v, wb_v, eb_v = xb_v.to(device), yb_v.to(device), wb_v.to(device), eb_v.to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            val_move, val_eval, val_uniform = forward_pass(xb_v, yb_v, wb_v, eb_v)
        val_total = val_move.item() + lambda_eval * val_eval.item()
        print(f'step {step:05d}/{max_iters} move {accum_move/grad_accum:.4f} eval {accum_eval/grad_accum:.4f} val_move {val_move.item():.4f} (uniform {val_uniform.item():.4f}) val_eval {val_eval.item():.4f} lr {lr_mult:.4f}')

        if val_total < best_val_loss:
            best_val_loss = val_total
            raw_sd = model.state_dict()
            clean_sd = {k.replace('_orig_mod.', ''): v for k, v in raw_sd.items()}
            best_ckpt = {
                'model': clean_sd,
                'eval_head': eval_head.state_dict(),
                'meta': {'model_config': config.__dict__, 'tokenizer': meta},
                'step': step,
                'val_loss': val_total,
            }
            best_name = ckpt_name.replace('.pt', '_best.pt')
            torch.save(best_ckpt, f'/data/eval/{best_name}')
            print(f'New best val loss {val_total:.4f} at step {step}, saved {best_name}')

    if step % 1000 == 0:
        raw_sd = model.state_dict()
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in raw_sd.items()}
        checkpoint = {
            'model': clean_sd,
            'eval_head': eval_head.state_dict(),
            'meta': {'model_config': config.__dict__, 'tokenizer': meta},
            'optimizers': [opt.state_dict() for opt in optimizers],
            'step': step,
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, f'/data/eval/{ckpt_name}')
        print(f'Saved checkpoint to {ckpt_name} at step {step}')
