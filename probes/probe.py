import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import chess
from nanochat.gpt import GPT, GPTConfig

# --- config ---
model_path = sys.argv[1] if len(sys.argv) > 1 else "models/chess_L12_H6_E768.pt"
data_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/val.bin"
meta_path = sys.argv[3] if len(sys.argv) > 3 else "data/processed/meta.pkl"
probe_layers = [0, 3, 6, 9, 11]
n_train_games = 400
n_val_games = 100
probe_epochs = 20
probe_lr = 1e-3
batch_size = 256

# --- load model ---
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
config = GPTConfig(**ckpt["meta"]["model_config"])
model = GPT(config)
model.load_state_dict(ckpt["model"])
model = model.to(device).eval()
for p in model.parameters():
    p.requires_grad = False
n_embd = config.n_embd
print(f"Model: {config.n_layer}L {config.n_head}H {n_embd}E on {device}")

# --- load tokenizer ---
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
itos = meta["itos"]
stoi = meta["stoi"]
vocab_size = meta["vocab_size"]
bos_id = stoi["<bos>"]
eos_id = stoi["<eos>"]

# --- extract games from val.bin ---
raw = np.memmap(data_path, dtype=np.uint16, mode='r')
bos_positions = np.where(raw == bos_id)[0]
print(f"val.bin: {len(raw)} tokens, {len(bos_positions)} games")

games = []
for i in range(len(bos_positions)):
    start = bos_positions[i]
    end = bos_positions[i + 1] if i + 1 < len(bos_positions) else len(raw)
    tokens = list(raw[start:end])
    if tokens[-1] == eos_id:
        tokens = tokens[:-1]
    if len(tokens) >= 5:
        games.append(tokens)

train_games = games[:n_train_games]
val_games = games[n_train_games:n_train_games + n_val_games]
print(f"Using {len(train_games)} train, {len(val_games)} val games")

# --- board labeling ---
def board_labels(board):
    pieces = np.zeros(64, dtype=np.int64)
    for sq in range(64):
        p = board.piece_at(sq)
        if p is not None:
            pieces[sq] = p.piece_type + (6 if p.color == chess.BLACK else 0)
    turn = float(board.turn)  # WHITE=1, BLACK=0
    in_check = float(board.is_check())
    legal_ids = []
    for mv in board.legal_moves:
        san = board.san(mv)
        if san in stoi:
            legal_ids.append(stoi[san])
    return pieces, turn, in_check, np.array(legal_ids, dtype=np.int32)

# --- collect hidden states ---
_hook_out = {}
hooks = []

def make_hook(li):
    def fn(module, inp, out):
        _hook_out[li] = out.detach().cpu()
    return fn

for li in probe_layers:
    hooks.append(model.transformer.h[li].register_forward_hook(make_hook(li)))

def collect(game_list, label):
    H = {l: [] for l in probe_layers}
    pieces_all, turn_all, check_all, legal_all = [], [], [], []

    for gi, tokens in enumerate(game_list):
        board = chess.Board()
        labels = [board_labels(board)]
        for tid in tokens[1:]:
            try:
                board.push_san(itos[tid])
            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                break
            labels.append(board_labels(board))

        n = len(labels)
        with torch.no_grad():
            model(torch.tensor([tokens[:n]], dtype=torch.long, device=device))

        for li in probe_layers:
            H[li].append(_hook_out[li].squeeze(0)[:n])

        for pieces, turn, check, legal_ids in labels:
            pieces_all.append(pieces)
            turn_all.append(turn)
            check_all.append(check)
            legal_all.append(legal_ids)

        if (gi + 1) % 100 == 0:
            n_pos = sum(h.shape[0] for h in H[probe_layers[0]])
            print(f"  [{label}] {gi+1}/{len(game_list)} games, {n_pos} positions")

    n_pos = sum(h.shape[0] for h in H[probe_layers[0]])
    print(f"  [{label}] done: {n_pos} positions from {len(game_list)} games")

    return (
        {l: torch.cat(H[l]) for l in probe_layers},
        torch.from_numpy(np.stack(pieces_all)),
        torch.tensor(turn_all, dtype=torch.float32).unsqueeze(1),
        torch.tensor(check_all, dtype=torch.float32).unsqueeze(1),
        legal_all,  # list of int32 arrays (sparse, memory-efficient)
    )

print("\nCollecting hidden states...")
train_h, train_pieces, train_turn, train_check, train_legal = collect(train_games, "train")
val_h, val_pieces, val_turn, val_check, val_legal = collect(val_games, "val")

for h in hooks:
    h.remove()
del model, ckpt

print(f"Train: {train_pieces.shape[0]} positions, Val: {val_pieces.shape[0]} positions")
for li in probe_layers:
    print(f"  Layer {li} hidden: train {train_h[li].shape}, val {val_h[li].shape}")

# --- probe training ---
def make_legal_batch(legal_list, idx):
    batch = torch.zeros(len(idx), vocab_size)
    for i, j in enumerate(idx):
        ids = legal_list[j]
        if len(ids) > 0:
            batch[i, ids] = 1.0
    return batch

def train_probe(name, probe, train_x, train_y, val_x, val_y, loss_fn, epochs=probe_epochs):
    opt = torch.optim.AdamW(probe.parameters(), lr=probe_lr)
    n = train_x.shape[0]

    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(n)
        total_loss, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = train_x[idx]
            yb = train_y(idx) if callable(train_y) else train_y[idx]
            loss = loss_fn(probe(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch == 0 or epoch == epochs - 1:
            probe.eval()
            with torch.no_grad():
                val_yb = val_y(torch.arange(val_x.shape[0])) if callable(val_y) else val_y
                val_loss = loss_fn(probe(val_x), val_yb)
            print(f"    {name} epoch {epoch:2d}: train={total_loss / n_batches:.4f} val={val_loss.item():.4f}")

    return probe

def piece_loss(logits, targets):
    return F.cross_entropy(logits.view(-1, 13), targets.view(-1))

def bce_loss(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets)

# --- baselines ---
print("\n--- Baselines ---")
empty_frac = (val_pieces == 0).float().mean().item()
print(f"Piece majority (predict empty): {empty_frac:.4f}")
turn_majority = max(val_turn.mean().item(), 1 - val_turn.mean().item())
print(f"Turn majority: {turn_majority:.4f}")
check_majority = 1 - val_check.mean().item()
print(f"Check majority (predict no-check): {check_majority:.4f}")

# --- train probes per layer ---
results = {}
for li in probe_layers:
    print(f"\n=== Layer {li} ===")
    th, vh = train_h[li], val_h[li]

    # piece placement
    probe = train_probe(
        "pieces", nn.Linear(n_embd, 64 * 13), th, train_pieces, vh, val_pieces, piece_loss
    )
    with torch.no_grad():
        pred = probe(vh).view(-1, 64, 13).argmax(dim=2)
        all_acc = (pred == val_pieces).float().mean().item()
        occ_mask = val_pieces != 0
        occ_acc = (pred[occ_mask] == val_pieces[occ_mask]).float().mean().item() if occ_mask.any() else 0.0
    print(f"    -> all={all_acc:.4f} occupied={occ_acc:.4f}")
    del probe

    # turn
    probe = train_probe("turn", nn.Linear(n_embd, 1), th, train_turn, vh, val_turn, bce_loss)
    with torch.no_grad():
        turn_acc = ((probe(vh).sigmoid() > 0.5).float() == val_turn).float().mean().item()
    print(f"    -> acc={turn_acc:.4f}")
    del probe

    # check
    probe = train_probe("check", nn.Linear(n_embd, 1), th, train_check, vh, val_check, bce_loss)
    with torch.no_grad():
        check_acc = ((probe(vh).sigmoid() > 0.5).float() == val_check).float().mean().item()
    print(f"    -> acc={check_acc:.4f}")
    del probe

    # legal moves
    train_legal_fn = lambda idx, ll=train_legal: make_legal_batch(ll, idx.tolist())
    val_legal_fn = lambda idx, ll=val_legal: make_legal_batch(ll, idx.tolist())
    probe = train_probe(
        "legal", nn.Linear(n_embd, vocab_size), th, train_legal_fn, vh, val_legal_fn, bce_loss
    )
    with torch.no_grad():
        tp, pp, ap = 0.0, 0.0, 0.0
        for i in range(0, vh.shape[0], batch_size):
            pred = (probe(vh[i:i + batch_size]).sigmoid() > 0.5).float()
            target = make_legal_batch(val_legal, list(range(i, min(i + batch_size, vh.shape[0]))))
            tp += (pred * target).sum().item()
            pp += pred.sum().item()
            ap += target.sum().item()
        precision = tp / pp if pp > 0 else 0
        recall = tp / ap if ap > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"    -> F1={f1:.4f} (P={precision:.4f} R={recall:.4f})")
    del probe

    results[li] = dict(pieces_all=all_acc, pieces_occ=occ_acc, turn=turn_acc, check=check_acc, legal_f1=f1)

# --- results table ---
print("\n" + "=" * 78)
print(f"{'Layer':>5} | {'Pieces (all)':>12} | {'Pieces (occ)':>12} | {'Turn':>7} | {'Check':>7} | {'Legal F1':>8}")
print("-" * 78)
for li in probe_layers:
    r = results[li]
    print(f"{li:>5} | {r['pieces_all']:>12.4f} | {r['pieces_occ']:>12.4f} | {r['turn']:>7.4f} | {r['check']:>7.4f} | {r['legal_f1']:>8.4f}")
print("-" * 78)
print(f"{'base':>5} | {empty_frac:>12.4f} | {'n/a':>12} | {turn_majority:>7.4f} | {check_majority:>7.4f} | {'n/a':>8}")
print("=" * 78)
