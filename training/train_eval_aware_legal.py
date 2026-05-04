import pickle
import math
import os
import sys
from time import perf_counter
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_registry import resolve_model_ref
from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.muon import Muon
try:
    from legality import build_legal_targets, build_normalized_token_id_map, legality_loss_from_logits
except ImportError:
    from training.legality import build_legal_targets, build_normalized_token_id_map, legality_loss_from_logits

data_dir = Path(__file__).parent.parent / 'data' / 'eval'
meta = pickle.loads((data_dir / 'meta.pkl').read_bytes())
vocab_size = meta['vocab_size']
context_length = meta['context_length']
bos_id = meta['stoi']['<bos>']
eos_id = meta['stoi']['<eos>']
itos = meta['itos']

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
    ew = torch.stack([weights[i : i + block_size] for i in idx])
    return x, y, w, e, ew, idx


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

init_model_ref = os.environ.get("INIT_MODEL_REF", "eval-aware/v2/best").strip()
ckpt_name = os.environ.get("CKPT_NAME", f"chess_eval_aware_v2_legal_L{config.n_layer}_H{config.n_head}_E{config.n_embd}.pt")
max_iters = int(os.environ.get("MAX_ITERS", "3000"))
warmup_steps = int(os.environ.get("WARMUP_STEPS", "200"))
batch_size = int(os.environ.get("BATCH_SIZE", "64"))
grad_accum = int(os.environ.get("GRAD_ACCUM", "4"))
eval_interval = int(os.environ.get("EVAL_INTERVAL", "100"))
ckpt_interval = int(os.environ.get("CKPT_INTERVAL", str(eval_interval)))
timing_steps = int(os.environ.get("TIMING_STEPS", "0"))
lambda_eval = float(os.environ.get("LAMBDA_EVAL", "1.0"))
lambda_legal = float(os.environ.get("LAMBDA_LEGAL", "0.1"))
legality_allow_eos = os.environ.get("LEGALITY_ALLOW_EOS", "1") != "0"
lm_head_lr = float(os.environ.get("LM_HEAD_LR", "5e-4"))
wte_lr = float(os.environ.get("WTE_LR", "1e-2"))
eval_head_lr = float(os.environ.get("EVAL_HEAD_LR", "5e-4"))
muon_lr = float(os.environ.get("MUON_LR", "5e-3"))

# Muon for transformer block weights, AdamW for embeddings + heads
lr_scale = (config.n_embd / 768) ** -0.5
adamw = torch.optim.AdamW([
    dict(params=list(model.lm_head.parameters()), lr=lm_head_lr * lr_scale),
    dict(params=list(model.transformer.wte.parameters()), lr=wte_lr * lr_scale),
    dict(params=list(eval_head.parameters()), lr=eval_head_lr * lr_scale),
], betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
muon = Muon(model.transformer.h.parameters(), lr=muon_lr, momentum=0.95)
optimizers = [adamw, muon]
for opt in optimizers:
    for group in opt.param_groups:
        group['initial_lr'] = group['lr']

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
        best_val_move = state.get('best_val_move', float('inf'))
        print(f"Resumed from {ckpt_path} at step {start_step}, best_val_loss={best_val_loss:.4f}, best_val_move={best_val_move:.4f}")
else:
    print("No legality checkpoint found, starting a new run")
    best_val_loss = float('inf')
    best_val_move = float('inf')
    if init_model_ref:
        init_candidate = Path(init_model_ref).expanduser()
        if init_candidate.exists():
            init_model_path = str(init_candidate)
            init_spec = None
        else:
            init_model_path, init_spec = resolve_model_ref(init_model_ref)
            if init_spec is not None and not Path(init_model_path).exists():
                for candidate in (
                    Path("/data") / init_spec.dataset / init_spec.filename,
                    Path("/data") / init_spec.filename,
                ):
                    if candidate.exists():
                        init_model_path = str(candidate)
                        break
        init_state = torch.load(init_model_path, map_location=device, weights_only=False)
        init_meta = init_state.get('meta', {})
        init_tok = init_meta.get('tokenizer', {})
        init_cfg = init_meta.get('model_config', {})
        if init_tok.get('vocab_size') != vocab_size:
            raise ValueError(
                f"Init checkpoint vocab mismatch: {init_tok.get('vocab_size')} vs current {vocab_size}"
            )
        if init_cfg.get('n_layer') != config.n_layer or init_cfg.get('n_embd') != config.n_embd:
            raise ValueError(
                f"Init checkpoint model mismatch: layers={init_cfg.get('n_layer')} embd={init_cfg.get('n_embd')}"
            )
        init_sd = {k.replace('_orig_mod.', ''): v for k, v in init_state['model'].items()}
        model.load_state_dict(init_sd)
        if 'eval_head' in init_state:
            eval_head.load_state_dict(init_state['eval_head'])
        init_label = init_spec.primary_alias if init_spec is not None else init_model_path
        print(f"Initialized legality run from {init_label} ({init_model_path})")

softcap = 15
use_amp = device.type == 'cuda'
print(
    f"run config: ckpt_name={ckpt_name} init_model_ref={init_model_ref or '<random-init>'} "
    f"max_iters={max_iters} warmup_steps={warmup_steps} batch_size={batch_size} grad_accum={grad_accum} "
    f"eval_interval={eval_interval} ckpt_interval={ckpt_interval} timing_steps={timing_steps} "
    f"lambda_eval={lambda_eval} lambda_legal={lambda_legal} "
    f"lrs(lm={lm_head_lr},wte={wte_lr},eval={eval_head_lr},muon={muon_lr})"
)


def cuda_sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()


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
train_bos_positions = np.flatnonzero(train_tokens == bos_id)
val_bos_positions = np.flatnonzero(val_tokens == bos_id)
san_to_ids = build_normalized_token_id_map(itos)


def forward_pass(xb, yb, wb, eb, ewb, legal_targets):
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
    bos_mask = (yb != bos_id).float()
    move_weights = wb * bos_mask
    move_loss = (per_token * move_weights.view(-1)).sum() / move_weights.sum()
    legal_loss = legality_loss_from_logits(logits, wb, legal_targets)

    eval_pred = torch.tanh(eval_head(x).squeeze(-1))
    eval_mask = ~torch.isnan(eb)
    if eval_mask.any():
        per_pos = (eval_pred[eval_mask] - eb[eval_mask]) ** 2
        eval_loss = (per_pos * ewb[eval_mask]).sum() / ewb[eval_mask].sum()
    else:
        eval_loss = torch.tensor(0.0, device=device)

    return move_loss, legal_loss, eval_loss, per_token.mean()


for step in range(start_step, max_iters + 1):
    do_timing = step <= timing_steps
    step_t0 = perf_counter() if do_timing else 0.0
    lr_mult = get_lr_mult(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * lr_mult

    for opt in optimizers:
        opt.zero_grad()

    accum_move = 0.0
    accum_legal = 0.0
    accum_eval = 0.0
    sample_s = 0.0
    train_legal_total_s = 0.0
    train_legal_replay_s = 0.0
    train_legal_ids_s = 0.0
    train_legal_advance_s = 0.0
    train_legal_tensorize_s = 0.0
    transfer_s = 0.0
    fwd_bwd_s = 0.0
    opt_s = 0.0
    val_sample_s = 0.0
    val_legal_total_s = 0.0
    val_legal_replay_s = 0.0
    val_legal_ids_s = 0.0
    val_legal_advance_s = 0.0
    val_legal_tensorize_s = 0.0
    val_transfer_s = 0.0
    val_fwd_s = 0.0
    for _ in range(grad_accum):
        sample_t0 = perf_counter() if do_timing else 0.0
        xb_cpu, yb_cpu, wb_cpu, eb_cpu, ewb_cpu, start_idx = sample_batch(train_tok, context_length, batch_size, train_w, train_e)
        if do_timing:
            sample_s += perf_counter() - sample_t0
            legal_targets, legal_stats = build_legal_targets(
                token_stream=train_tokens,
                start_indices=start_idx,
                target_batch=yb_cpu,
                itos=itos,
                bos_positions=train_bos_positions,
                san_to_ids=san_to_ids,
                bos_id=bos_id,
                eos_id=eos_id,
                device=device,
                allow_eos=legality_allow_eos,
                return_stats=True,
            )
            train_legal_total_s += legal_stats.total_s
            train_legal_replay_s += legal_stats.replay_s
            train_legal_ids_s += legal_stats.legal_ids_s
            train_legal_advance_s += legal_stats.target_advance_s
            train_legal_tensorize_s += legal_stats.tensorize_s
        else:
            legal_targets = build_legal_targets(
                token_stream=train_tokens,
                start_indices=start_idx,
                target_batch=yb_cpu,
                itos=itos,
                bos_positions=train_bos_positions,
                san_to_ids=san_to_ids,
                bos_id=bos_id,
                eos_id=eos_id,
                device=device,
                allow_eos=legality_allow_eos,
            )
        transfer_t0 = perf_counter() if do_timing else 0.0
        xb, yb, wb, eb, ewb = xb_cpu.to(device), yb_cpu.to(device), wb_cpu.to(device), eb_cpu.to(device), ewb_cpu.to(device)
        if do_timing:
            cuda_sync()
            transfer_s += perf_counter() - transfer_t0

        fwd_t0 = perf_counter() if do_timing else 0.0
        if do_timing:
            cuda_sync()
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            move_loss, legal_loss, eval_loss, _ = forward_pass(xb, yb, wb, eb, ewb, legal_targets)
            loss = (move_loss + lambda_legal * legal_loss + lambda_eval * eval_loss) / grad_accum

        loss.backward()
        if do_timing:
            cuda_sync()
            fwd_bwd_s += perf_counter() - fwd_t0
        accum_move += move_loss.item()
        accum_legal += legal_loss.item()
        accum_eval += eval_loss.item()

    opt_t0 = perf_counter() if do_timing else 0.0
    if do_timing:
        cuda_sync()
    all_params = list(model.parameters()) + list(eval_head.parameters())
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    for opt in optimizers:
        opt.step()
    if do_timing:
        cuda_sync()
        opt_s = perf_counter() - opt_t0

    if step % eval_interval == 0 or step == 1:
        val_sample_t0 = perf_counter() if do_timing else 0.0
        xb_v_cpu, yb_v_cpu, wb_v_cpu, eb_v_cpu, ewb_v_cpu, start_idx_v = sample_batch(val_tok, context_length, batch_size, val_w, val_e)
        if do_timing:
            val_sample_s = perf_counter() - val_sample_t0
            legal_targets_v, legal_stats_v = build_legal_targets(
                token_stream=val_tokens,
                start_indices=start_idx_v,
                target_batch=yb_v_cpu,
                itos=itos,
                bos_positions=val_bos_positions,
                san_to_ids=san_to_ids,
                bos_id=bos_id,
                eos_id=eos_id,
                device=device,
                allow_eos=legality_allow_eos,
                return_stats=True,
            )
            val_legal_total_s = legal_stats_v.total_s
            val_legal_replay_s = legal_stats_v.replay_s
            val_legal_ids_s = legal_stats_v.legal_ids_s
            val_legal_advance_s = legal_stats_v.target_advance_s
            val_legal_tensorize_s = legal_stats_v.tensorize_s
        else:
            legal_targets_v = build_legal_targets(
                token_stream=val_tokens,
                start_indices=start_idx_v,
                target_batch=yb_v_cpu,
                itos=itos,
                bos_positions=val_bos_positions,
                san_to_ids=san_to_ids,
                bos_id=bos_id,
                eos_id=eos_id,
                device=device,
                allow_eos=legality_allow_eos,
            )
        val_transfer_t0 = perf_counter() if do_timing else 0.0
        xb_v, yb_v, wb_v, eb_v, ewb_v = xb_v_cpu.to(device), yb_v_cpu.to(device), wb_v_cpu.to(device), eb_v_cpu.to(device), ewb_v_cpu.to(device)
        if do_timing:
            cuda_sync()
            val_transfer_s = perf_counter() - val_transfer_t0
            cuda_sync()
        val_fwd_t0 = perf_counter() if do_timing else 0.0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            val_move, val_legal, val_eval, val_uniform = forward_pass(xb_v, yb_v, wb_v, eb_v, ewb_v, legal_targets_v)
        if do_timing:
            cuda_sync()
            val_fwd_s = perf_counter() - val_fwd_t0
        val_total = val_move.item() + lambda_legal * val_legal.item() + lambda_eval * val_eval.item()
        print(
            f'step {step:05d}/{max_iters} move {accum_move/grad_accum:.4f} '
            f'legal {accum_legal/grad_accum:.4f} eval {accum_eval/grad_accum:.4f} '
            f'val_move {val_move.item():.4f} val_legal {val_legal.item():.4f} '
            f'(uniform {val_uniform.item():.4f}) val_eval {val_eval.item():.4f} lr {lr_mult:.4f}'
        )
        if do_timing:
            print(
                f"timing step {step:05d} train sample {sample_s:.3f}s legal {train_legal_total_s:.3f}s "
                f"(replay {train_legal_replay_s:.3f}s legal_ids {train_legal_ids_s:.3f}s "
                f"advance {train_legal_advance_s:.3f}s tensorize {train_legal_tensorize_s:.3f}s) "
                f"transfer {transfer_s:.3f}s fwd_bwd {fwd_bwd_s:.3f}s opt {opt_s:.3f}s"
            )
            print(
                f"timing step {step:05d} val sample {val_sample_s:.3f}s legal {val_legal_total_s:.3f}s "
                f"(replay {val_legal_replay_s:.3f}s legal_ids {val_legal_ids_s:.3f}s "
                f"advance {val_legal_advance_s:.3f}s tensorize {val_legal_tensorize_s:.3f}s) "
                f"transfer {val_transfer_s:.3f}s fwd {val_fwd_s:.3f}s "
                f"step_wall {perf_counter() - step_t0:.3f}s"
            )

        if val_total < best_val_loss:
            best_val_loss = val_total
            raw_sd = model.state_dict()
            clean_sd = {k.replace('_orig_mod.', ''): v for k, v in raw_sd.items()}
            best_ckpt = {
                'model': clean_sd,
                'eval_head': eval_head.state_dict(),
                'meta': {'model_config': config.__dict__, 'tokenizer': meta},
                'train_config': {
                    'init_model_ref': init_model_ref,
                    'lambda_eval': lambda_eval,
                    'lambda_legal': lambda_legal,
                    'legality_allow_eos': legality_allow_eos,
                },
                'step': step,
                'val_loss': val_total,
                'val_move_loss': val_move.item(),
                'val_legal_loss': val_legal.item(),
                'val_eval_loss': val_eval.item(),
            }
            best_name = ckpt_name.replace('.pt', '_best.pt')
            torch.save(best_ckpt, f'/data/eval/{best_name}')
            print(f'New best val loss {val_total:.4f} at step {step}, saved {best_name}')

        if val_move.item() < best_val_move:
            best_val_move = val_move.item()
            raw_sd = model.state_dict()
            clean_sd = {k.replace('_orig_mod.', ''): v for k, v in raw_sd.items()}
            best_move_ckpt = {
                'model': clean_sd,
                'eval_head': eval_head.state_dict(),
                'meta': {'model_config': config.__dict__, 'tokenizer': meta},
                'train_config': {
                    'init_model_ref': init_model_ref,
                    'lambda_eval': lambda_eval,
                    'lambda_legal': lambda_legal,
                    'legality_allow_eos': legality_allow_eos,
                },
                'step': step,
                'val_move_loss': best_val_move,
                'val_legal_loss': val_legal.item(),
                'val_total_loss': val_total,
            }
            best_move_name = ckpt_name.replace('.pt', '_best_move.pt')
            torch.save(best_move_ckpt, f'/data/eval/{best_move_name}')
            print(f'New best val_move {best_val_move:.4f} at step {step}, saved {best_move_name}')

    if step % ckpt_interval == 0:
        raw_sd = model.state_dict()
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in raw_sd.items()}
        checkpoint = {
            'model': clean_sd,
            'eval_head': eval_head.state_dict(),
            'meta': {'model_config': config.__dict__, 'tokenizer': meta},
            'train_config': {
                'init_model_ref': init_model_ref,
                'lambda_eval': lambda_eval,
                'lambda_legal': lambda_legal,
                'legality_allow_eos': legality_allow_eos,
            },
            'optimizers': [opt.state_dict() for opt in optimizers],
            'step': step,
            'best_val_loss': best_val_loss,
            'best_val_move': best_val_move,
        }
        torch.save(checkpoint, f'/data/eval/{ckpt_name}')
        print(f'Saved checkpoint to {ckpt_name} at step {step}')
