"""GRPO on puzzle first-moves, starting from a pretrained checkpoint.

Each step: forward a batch of puzzle positions, sample K moves from the last-position
distribution at TEMPERATURE, reward 1/0 against the stored correct set (solution move
plus mating alternates), advantage = reward minus the group mean (no std division),
loss = -advantage-weighted logprob of the draws + KL_BETA * exact KL to the frozen
init model. On-policy with a single update per rollout, so no PPO ratio/clip.

Val is analytic (no sampling): expected reward and pass@K from the probability mass
on correct tokens, illegal mass from per-position legal sets rebuilt from fen, plus
KL and entropy. train_loop's best-checkpoint metric is neg_reward.

Run: MODAL_GPU=A10G uv run modal run modal_train.py --datasets datasets/puzzles/grpo \
  --script training/train_grpo.py --env-overrides "DATASET_DIR=datasets/puzzles/grpo"
"""

import json
import os
import sys
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig

from common import forward_hidden, get_device, load_model_state, move_logits, train_loop

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_inference import token_for_id
from chess_token_utils import strip_san

dataset_dir_name = os.getenv("DATASET_DIR", "datasets/puzzles/grpo").strip()
data_dir = Path(__file__).parent.parent / "data" / dataset_dir_name

init_path = os.getenv("CKPT_INIT", "/data/checkpoints/plain/games-15m/l16_best.pt")
K = int(os.getenv("K", "8"))
temperature = float(os.getenv("TEMPERATURE", "0.8"))
kl_beta = float(os.getenv("KL_BETA", "0.05"))
lr = float(os.getenv("LR", "1e-5"))
batch_size = int(os.getenv("BATCH_SIZE", "64"))
val_size = int(os.getenv("VAL_SIZE", "2048"))
seed = int(os.getenv("SEED", "0"))
torch.manual_seed(seed)

device = get_device()
init_ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
config = GPTConfig(**init_ckpt["meta"]["model_config"])
meta = init_ckpt["meta"]["tokenizer"]
stoi = meta["stoi"]
vocab_size = config.vocab_size
seq_len = config.sequence_len

model = GPT(config).to(device)
load_model_state(model, init_ckpt)
ref_model = GPT(config).to(device).eval()
load_model_state(ref_model, init_ckpt)
for p in ref_model.parameters():
    p.requires_grad_(False)
del init_ckpt
print(f"init {init_path} (vocab {vocab_size}, seq_len {seq_len})")
print(f"K {K} temp {temperature} kl_beta {kl_beta} lr {lr:g} batch {batch_size}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# the vocab keeps +/# annotations, so one board move can be several tokens; grade
# and mask by stripped form, matching the benchmark's semantics
itos = meta["itos"]
strip_ids: dict[str, list[int]] = {}
for tok, tid in stoi.items():
    if not tok.startswith("<"):
        strip_ids.setdefault(strip_san(tok), []).append(tid)


def move_variants(tids) -> list[int]:
    return sorted({v for t in tids for v in strip_ids[strip_san(token_for_id(itos, t))]})


def load_positions(path: Path, limit: int = 0):
    prefixes, lengths, corrects, records = [], [], [], []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            prefixes.append(rec["prefix"])
            lengths.append(len(rec["prefix"]))
            corrects.append(move_variants(rec["correct"]))
            records.append((rec["fen"], rec["sol"]))
            if limit and len(records) >= limit:
                break
    n = len(records)
    prefix_arr = np.zeros((n, seq_len), dtype=np.int16)
    max_c = max(len(c) for c in corrects)
    correct_arr = np.full((n, max_c), -1, dtype=np.int64)
    for i, p in enumerate(prefixes):
        prefix_arr[i, : len(p)] = p
        correct_arr[i, : len(corrects[i])] = corrects[i]
    return (
        torch.from_numpy(prefix_arr),
        torch.tensor(lengths, dtype=torch.long) - 1,
        torch.from_numpy(correct_arr),
        records,
    )


def legal_multihot(records) -> torch.Tensor:
    mh = np.zeros((len(records), vocab_size), dtype=np.float32)
    for i, (fen, sol) in enumerate(records):
        board = chess.Board(fen)
        board.push(board.parse_uci(sol.split()[0]))
        for mv in board.legal_moves:
            for tid in strip_ids.get(strip_san(board.san(mv)), ()):
                mh[i, tid] = 1.0
    return torch.from_numpy(mh)


train_x, train_last, train_correct, _ = load_positions(data_dir / "train.jsonl")
val_x, val_last, val_correct, val_records = load_positions(data_dir / "val.jsonl", limit=val_size)
print(f"train positions: {len(train_x):,}, val positions: {len(val_x):,}")
val_legal = legal_multihot(val_records)
print("built val legal masks")

ckpt_dir = Path(os.getenv("CKPT_DIR", "/data/checkpoints/grpo/games-15m-l16"))
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt_name = os.getenv("CKPT_NAME", "grpo.pt")

val_gen = torch.Generator().manual_seed(seed)


def sample_train():
    idx = torch.randint(0, len(train_x), (batch_size,))
    last = train_last[idx]
    return [train_x[idx, : int(last.max()) + 1].long(), last, train_correct[idx]]


def sample_val():
    idx = torch.randint(0, len(val_x), (batch_size,), generator=val_gen)
    last = val_last[idx]
    return [val_x[idx, : int(last.max()) + 1].long(), last, val_correct[idx], val_legal[idx]]


def last_logprobs(m, x, last):
    h = forward_hidden(m, x)
    h_last = h[torch.arange(x.size(0), device=x.device), last]
    return F.log_softmax(move_logits(m, h_last) / temperature, dim=-1)


def train_step(m, x, last, correct):
    logp = last_logprobs(m, x, last)
    with torch.no_grad():
        draws = torch.multinomial(logp.exp().float(), K, replacement=True)
        ref_logp = last_logprobs(ref_model, x, last)
    r = (draws.unsqueeze(2) == correct.unsqueeze(1)).any(dim=2).float()
    adv = r - r.mean(dim=1, keepdim=True)
    pg = -(adv * logp.gather(1, draws)).mean()
    kl = (logp.exp() * (logp - ref_logp)).sum(dim=-1).mean()
    return pg + kl_beta * kl


def val_step(m, x, last, correct, legal):
    logp = last_logprobs(m, x, last)
    ref_logp = last_logprobs(ref_model, x, last)
    probs = logp.exp()
    p_correct = (probs.gather(1, correct.clamp(min=0)) * (correct >= 0)).sum(dim=-1)
    return {
        "neg_reward": -p_correct.mean().item(),
        "pass_k": (1 - (1 - p_correct) ** K).mean().item(),
        "illegal": (1 - (probs * legal).sum(dim=-1)).mean().item(),
        "kl": (probs * (logp - ref_logp)).sum(dim=-1).mean().item(),
        "entropy": -(probs * logp).sum(dim=-1).mean().item(),
    }


train_loop(
    model, optimizer, config, meta, device,
    ckpt_path=ckpt_dir / ckpt_name,
    best_ckpt_path=ckpt_dir / ckpt_name.replace(".pt", "_best.pt"),
    sample_train=sample_train,
    sample_val=sample_val,
    train_step=train_step,
    val_step=val_step,
    metric_key="neg_reward",
    max_iters=int(os.getenv("MAX_ITERS", "1000")),
    grad_accum_steps=int(os.getenv("GRAD_ACCUM_STEPS", "1")),
)
