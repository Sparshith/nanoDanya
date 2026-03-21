import pickle
import math
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.muon import Muon

data_root = Path(__file__).parent.parent / 'data'

# load puzzle_weighted dataset (5M games, with per-token weights)
pw_dir = data_root / 'puzzle_weighted'
pw_meta = pickle.loads((pw_dir / 'meta.pkl').read_bytes())
vocab_size = pw_meta['vocab_size']
context_length = pw_meta['context_length']
pw_train_tok = np.fromfile(pw_dir / 'train.bin', dtype=np.uint16)
pw_val_tok = np.fromfile(pw_dir / 'val.bin', dtype=np.uint16)
pw_train_w = np.fromfile(pw_dir / 'train_weights.bin', dtype=np.float32)
pw_val_w = np.fromfile(pw_dir / 'val_weights.bin', dtype=np.float32)

# load eval dataset (3M games, with per-token evals)
ev_dir = data_root / 'eval'
ev_meta = pickle.loads((ev_dir / 'meta.pkl').read_bytes())
assert ev_meta['vocab_size'] == vocab_size and ev_meta['context_length'] == context_length
ev_train_tok = np.fromfile(ev_dir / 'train.bin', dtype=np.uint16)
ev_val_tok = np.fromfile(ev_dir / 'val.bin', dtype=np.uint16)
ev_train_e = np.fromfile(ev_dir / 'train_evals.bin', dtype=np.float32)
ev_val_e = np.fromfile(ev_dir / 'val_evals.bin', dtype=np.float32)


def normalize_evals(raw):
    e = raw.copy()
    mask = ~np.isnan(e)
    e[mask] = np.clip(e[mask], -1500, 1500) / 1500.0
    return e


ev_train_e = normalize_evals(ev_train_e)
ev_val_e = normalize_evals(ev_val_e)

# concatenate: puzzle_weighted gets NaN evals, eval gets uniform weights
train_tokens = np.concatenate([pw_train_tok, ev_train_tok])
val_tokens = np.concatenate([pw_val_tok, ev_val_tok])
train_weights = np.concatenate([pw_train_w, np.ones(len(ev_train_tok), dtype=np.float32)])
val_weights = np.concatenate([pw_val_w, np.ones(len(ev_val_tok), dtype=np.float32)])
train_evals = np.concatenate([np.full(len(pw_train_tok), np.nan, dtype=np.float32), ev_train_e])
val_evals = np.concatenate([np.full(len(pw_val_tok), np.nan, dtype=np.float32), ev_val_e])

print(f"combined: {len(train_tokens):,} train, {len(val_tokens):,} val tokens")
print(f"eval coverage: {(~np.isnan(train_evals)).sum() / len(train_evals):.1%}")
print(f"weighted fraction: {(train_weights > 1).sum() / len(train_weights):.1%}")


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
eval_head = nn.Linear(config.n_embd, 1, bias=False).to(device)
nn.init.zeros_(eval_head.weight)

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

ckpt_name = f"chess_combined_L{config.n_layer}_H{config.n_head}_E{config.n_embd}.pt"
start_step = 1
for ckpt_path in (Path(f"/data/{ckpt_name}"),):
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
            print(f"Resumed from {ckpt_path} at step {start_step}")
            break
    print("No checkpoint found, starting fresh")

max_iters = 100000
warmup_steps = 5000
batch_size = 64
grad_accum = 4
eval_interval = 100
lambda_eval = 1.0
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

    eval_pred = eval_head(x).squeeze(-1)
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
        print(f'step {step:05d}/{max_iters} move {accum_move/grad_accum:.4f} eval {accum_eval/grad_accum:.4f} val_move {val_move.item():.4f} (uniform {val_uniform.item():.4f}) val_eval {val_eval.item():.4f} lr {lr_mult:.4f}')

    if step % 1000 == 0:
        raw_sd = model.state_dict()
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in raw_sd.items()}
        checkpoint = {
            'model': clean_sd,
            'eval_head': eval_head.state_dict(),
            'meta': {'model_config': config.__dict__, 'tokenizer': pw_meta},
            'optimizers': [opt.state_dict() for opt in optimizers],
            'step': step,
        }
        torch.save(checkpoint, f'/data/{ckpt_name}')
        print(f'Saved checkpoint to {ckpt_name} at step {step}')
