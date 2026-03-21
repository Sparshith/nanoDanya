import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import chess
from pathlib import Path
from nanochat.gpt import GPT, GPTConfig

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/chess_weighted_L12_H6_E768.pt"
data_path = "data/processed/val.bin"
meta_path = "data/processed/meta.pkl"
output_path = Path(__file__).parent / "probe_viz.html"
probe_layer = 11
n_train_games = 400
n_val_games = 100
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

with open(meta_path, "rb") as f:
    meta = pickle.load(f)
itos = meta["itos"]
stoi = meta["stoi"]
bos_id = stoi["<bos>"]
eos_id = stoi["<eos>"]

raw = np.memmap(data_path, dtype=np.uint16, mode='r')
bos_positions = np.where(raw == bos_id)[0]
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

def board_labels(board):
    pieces = np.zeros(64, dtype=np.int64)
    for sq in range(64):
        p = board.piece_at(sq)
        if p is not None:
            pieces[sq] = p.piece_type + (6 if p.color == chess.BLACK else 0)
    return pieces

_hook_out = {}
hook = model.transformer.h[probe_layer].register_forward_hook(
    lambda mod, inp, out: _hook_out.update({0: out.detach().cpu()})
)

def collect(game_list, label):
    H, pieces_all = [], []
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
        H.append(_hook_out[0].squeeze(0)[:n])
        for p in labels:
            pieces_all.append(p)
        if (gi + 1) % 100 == 0:
            print(f"  [{label}] {gi+1}/{len(game_list)} games")
    print(f"  [{label}] done")
    return torch.cat(H), torch.from_numpy(np.stack(pieces_all))

print("Collecting hidden states (layer 11 only)...")
train_h, train_pieces = collect(train_games, "train")
val_h, val_pieces = collect(val_games, "val")
hook.remove()
del model, ckpt

# --- train piece probe ---
print("Training piece probe...")
n_embd = config.n_embd
probe = nn.Linear(n_embd, 64 * 13)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
n = train_h.shape[0]
for epoch in range(20):
    probe.train()
    perm = torch.randperm(n)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        logits = probe(train_h[idx])
        loss = F.cross_entropy(logits.view(-1, 13), train_pieces[idx].view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
probe.eval()

# --- generate boards for visualization ---
PIECE_CHARS = {
    0: '', 1: '\u2659', 2: '\u2658', 3: '\u2657', 4: '\u2656', 5: '\u2655', 6: '\u2654',
    7: '\u265F', 8: '\u265E', 9: '\u265D', 10: '\u265C', 11: '\u265B', 12: '\u265A',
}

positions_to_show = [(0, 0), (0, 10), (0, 20), (0, 30), (1, 15), (2, 20), (3, 25), (4, 40)]
boards_html = []

for game_idx, move_idx in positions_to_show:
    if game_idx >= len(val_games):
        continue
    tokens = val_games[game_idx]
    board = chess.Board()
    moves_played = []
    for tid in tokens[1:move_idx + 1]:
        try:
            board.push_san(itos[tid])
            moves_played.append(itos[tid])
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
            break

    count = 0
    for gi in range(game_idx):
        b = chess.Board()
        nn_ = 1
        for tid in val_games[gi][1:]:
            try:
                b.push_san(itos[tid])
                nn_ += 1
            except:
                break
        count += nn_
    pos_idx = count + len(moves_played)

    if pos_idx >= val_h.shape[0]:
        continue

    truth = val_pieces[pos_idx].numpy()
    with torch.no_grad():
        pred = probe(val_h[pos_idx:pos_idx + 1]).view(64, 13).argmax(dim=1).numpy()

    correct_occ = sum(1 for sq in range(64) if truth[sq] != 0 and truth[sq] == pred[sq])
    total_occ = sum(1 for sq in range(64) if truth[sq] != 0)
    last_moves = ' '.join(moves_played[-5:]) if moves_played else '(starting position)'
    title = f"Game {game_idx}, move {len(moves_played)}: {last_moves}"
    subtitle = f"{correct_occ}/{total_occ} occupied squares correct"

    rows = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            sq = rank * 8 + file
            is_light = (rank + file) % 2 == 1
            t, p = int(truth[sq]), int(pred[sq])
            piece_char = PIECE_CHARS[p]

            if t == p:
                if t == 0:
                    bg = '#f0d9b5' if is_light else '#b58863'
                else:
                    bg = '#a8e6a0' if is_light else '#6abf69'
            else:
                bg = '#f5a0a0' if is_light else '#d66'
                if p == 0:
                    piece_char = PIECE_CHARS[t]

            cells.append((bg, piece_char, t, p))
        rows.append((rank, cells))

    boards_html.append((title, subtitle, rows))

# --- write HTML ---
html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Linear Probe: Piece Placement (Layer 11)</title>
<style>
body { font-family: system-ui, -apple-system, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 40px; }
h1 { color: #fff; font-size: 24px; margin-bottom: 4px; }
.subtitle { color: #888; font-size: 14px; margin-bottom: 30px; }
.legend { display: flex; gap: 24px; margin-bottom: 36px; font-size: 14px; }
.legend-item { display: flex; align-items: center; gap: 8px; }
.legend-swatch { width: 20px; height: 20px; border-radius: 4px; border: 1px solid #555; }
.boards { display: flex; flex-wrap: wrap; gap: 32px; }
.board-container { background: #16213e; border-radius: 12px; padding: 20px; }
.board-title { font-size: 13px; font-weight: 600; margin-bottom: 2px; color: #ccc; }
.board-subtitle { font-size: 12px; margin-bottom: 12px; color: #888; }
.board { display: grid; grid-template-columns: 20px repeat(8, 52px); grid-template-rows: repeat(8, 52px) 20px; gap: 0; }
.rank-label { display: flex; align-items: center; justify-content: center; font-size: 12px; color: #666; }
.file-label { display: flex; align-items: center; justify-content: center; font-size: 12px; color: #666; }
.cell { display: flex; align-items: center; justify-content: center; font-size: 32px; line-height: 1; }
.tooltip { position: relative; cursor: default; }
.tooltip .tip { visibility: hidden; background: #333; color: #fff; text-align: center; padding: 4px 8px;
  border-radius: 4px; font-size: 11px; position: absolute; z-index: 1; bottom: 110%; left: 50%;
  transform: translateX(-50%); white-space: nowrap; }
.tooltip:hover .tip { visibility: visible; }
</style></head><body>
<h1>Linear Probe: Piece Placement</h1>
<div class="subtitle">Layer 11 of chess_weighted_L12_H6_E768 &mdash; probing what the model "knows" about the board</div>
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#a8e6a0"></div> Correct piece</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f5a0a0"></div> Wrong prediction</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#f0d9b5"></div> Correct empty</div>
</div>
<div class="boards">
"""

PIECE_NAMES = {
    0: 'empty', 1: 'white pawn', 2: 'white knight', 3: 'white bishop',
    4: 'white rook', 5: 'white queen', 6: 'white king',
    7: 'black pawn', 8: 'black knight', 9: 'black bishop',
    10: 'black rook', 11: 'black queen', 12: 'black king',
}
FILES = 'abcdefgh'

for title, subtitle, rows in boards_html:
    html += f'<div class="board-container"><div class="board-title">{title}</div>'
    html += f'<div class="board-subtitle">{subtitle}</div><div class="board">'
    for rank, cells in rows:
        html += f'<div class="rank-label">{rank + 1}</div>'
        for file_idx, (bg, piece_char, t, p) in enumerate(cells):
            sq_name = f"{FILES[file_idx]}{rank + 1}"
            if t == p:
                tip = f"{sq_name}: {PIECE_NAMES[t]}"
            else:
                tip = f"{sq_name}: predicted {PIECE_NAMES[p]}, actual {PIECE_NAMES[t]}"
            html += f'<div class="cell tooltip" style="background:{bg}"><span class="tip">{tip}</span>{piece_char}</div>'
    # file labels
    html += '<div></div>'
    for f in FILES:
        html += f'<div class="file-label">{f}</div>'
    html += '</div></div>\n'

html += '</div></body></html>'

with open(output_path, 'w') as f:
    f.write(html)
print(f"\nWrote {output_path}")
