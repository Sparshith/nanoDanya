import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import chess
from pathlib import Path
from collections import defaultdict
from nanochat.gpt import GPT, GPTConfig

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/chess_weighted_L12_H6_E768.pt"
data_path = "data/processed/val.bin"
meta_path = "data/processed/meta.pkl"
output_path = Path(__file__).parent / "probe_analysis.html"
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

# --- collect hidden states + metadata per position ---
def board_info(board):
    pieces = np.zeros(64, dtype=np.int64)
    n_pieces = 0
    for sq in range(64):
        p = board.piece_at(sq)
        if p is not None:
            pieces[sq] = p.piece_type + (6 if p.color == chess.BLACK else 0)
            n_pieces += 1
    return pieces, n_pieces, board.is_check()

_hook_out = {}
hook = model.transformer.h[probe_layer].register_forward_hook(
    lambda mod, inp, out: _hook_out.update({0: out.detach().cpu()})
)

def collect(game_list, label):
    H, pieces_all, move_nums, piece_counts, check_flags = [], [], [], [], []
    for gi, tokens in enumerate(game_list):
        board = chess.Board()
        p, nc, chk = board_info(board)
        labels = [(p, nc, chk)]
        for tid in tokens[1:]:
            try:
                board.push_san(itos[tid])
            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                break
            p, nc, chk = board_info(board)
            labels.append((p, nc, chk))
        n = len(labels)
        with torch.no_grad():
            model(torch.tensor([tokens[:n]], dtype=torch.long, device=device))
        H.append(_hook_out[0].squeeze(0)[:n])
        for mi, (p, nc, chk) in enumerate(labels):
            pieces_all.append(p)
            move_nums.append(mi)
            piece_counts.append(nc)
            check_flags.append(chk)
        if (gi + 1) % 100 == 0:
            print(f"  [{label}] {gi+1}/{len(game_list)} games")
    print(f"  [{label}] done")
    return (
        torch.cat(H), torch.from_numpy(np.stack(pieces_all)),
        np.array(move_nums), np.array(piece_counts), np.array(check_flags),
    )

print("Collecting hidden states...")
train_h, train_pieces, _, _, _ = collect(train_games, "train")
val_h, val_pieces, val_moves, val_pcounts, val_checks = collect(val_games, "val")
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

# --- compute per-position accuracy ---
print("Computing per-position accuracy...")
with torch.no_grad():
    all_pred = []
    for i in range(0, val_h.shape[0], batch_size):
        pred = probe(val_h[i:i + batch_size]).view(-1, 64, 13).argmax(dim=2)
        all_pred.append(pred)
    all_pred = torch.cat(all_pred)

per_pos_correct = (all_pred == val_pieces).float()  # (N, 64)
per_pos_all_acc = per_pos_correct.mean(dim=1).numpy()  # accuracy across all 64 squares
occ_mask = val_pieces != 0  # (N, 64)
per_pos_occ_acc = []
for i in range(len(val_pieces)):
    mask = occ_mask[i]
    if mask.any():
        per_pos_occ_acc.append(per_pos_correct[i][mask].mean().item())
    else:
        per_pos_occ_acc.append(1.0)
per_pos_occ_acc = np.array(per_pos_occ_acc)

# --- bin by move number ---
def bin_accuracy(values, acc, bin_size):
    bins = defaultdict(list)
    for v, a in zip(values, acc):
        b = (int(v) // bin_size) * bin_size
        bins[b].append(a)
    result = []
    for b in sorted(bins.keys()):
        vals = bins[b]
        if len(vals) >= 10:
            result.append((b, b + bin_size, np.mean(vals), len(vals)))
    return result

move_bins = bin_accuracy(val_moves, per_pos_occ_acc, 5)
pcount_bins = bin_accuracy(val_pcounts, per_pos_occ_acc, 2)
check_acc = {
    'in_check': np.mean(per_pos_occ_acc[val_checks]) if val_checks.any() else 0,
    'not_check': np.mean(per_pos_occ_acc[~val_checks]) if (~val_checks).any() else 0,
    'n_check': int(val_checks.sum()),
    'n_no_check': int((~val_checks).sum()),
}

print(f"Positions: {len(val_moves)}")
print(f"Overall occupied acc: {per_pos_occ_acc.mean():.4f}")

# --- write HTML ---
html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Probe Accuracy Analysis</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 40px; }
h1 { color: #fff; font-size: 24px; margin-bottom: 4px; }
.subtitle { color: #888; font-size: 14px; margin-bottom: 36px; }
.charts { display: flex; flex-wrap: wrap; gap: 40px; }
.chart-box { background: #16213e; border-radius: 12px; padding: 24px; width: 560px; }
.chart-title { font-size: 15px; font-weight: 600; margin-bottom: 4px; }
.chart-desc { font-size: 12px; color: #888; margin-bottom: 16px; }
canvas { display: block; }
.stat-box { background: #16213e; border-radius: 12px; padding: 24px; margin-top: 32px; display: inline-flex; gap: 48px; }
.stat { text-align: center; }
.stat-val { font-size: 28px; font-weight: 700; color: #a8e6a0; }
.stat-label { font-size: 12px; color: #888; margin-top: 4px; }
</style></head><body>
<h1>When does the model lose board state?</h1>
<div class="subtitle">Layer 11 piece probe accuracy (occupied squares only) on validation set</div>

<div class="stat-box">
  <div class="stat"><div class="stat-val">OVERALL_ACC</div><div class="stat-label">overall accuracy</div></div>
  <div class="stat"><div class="stat-val">N_POS</div><div class="stat-label">positions evaluated</div></div>
  <div class="stat"><div class="stat-val">CHECK_ACC</div><div class="stat-label">accuracy when in check (n=N_CHECK)</div></div>
  <div class="stat"><div class="stat-val">NO_CHECK_ACC</div><div class="stat-label">accuracy when not in check (n=N_NO_CHECK)</div></div>
</div>

<div class="charts" style="margin-top: 36px;">
  <div class="chart-box">
    <div class="chart-title">Accuracy by move number</div>
    <div class="chart-desc">Does the model lose track of the board as the game goes on?</div>
    <canvas id="moveChart" width="512" height="300"></canvas>
  </div>
  <div class="chart-box">
    <div class="chart-title">Accuracy by piece count</div>
    <div class="chart-desc">Is the board harder to track with more pieces?</div>
    <canvas id="pcountChart" width="512" height="300"></canvas>
  </div>
</div>

<script>
const moveData = MOVE_DATA;
const pcountData = PCOUNT_DATA;

function drawChart(canvasId, data, xLabel, xKey) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = { top: 20, right: 20, bottom: 44, left: 52 };
  const cw = W - pad.left - pad.right, ch = H - pad.top - pad.bottom;

  ctx.fillStyle = '#16213e';
  ctx.fillRect(0, 0, W, H);

  const xs = data.map(d => d[xKey]);
  const ys = data.map(d => d.acc);
  const ns = data.map(d => d.n);

  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(0.5, Math.min(...ys) - 0.05);
  const yMax = 1.0;

  function tx(x) { return pad.left + (x - xMin) / (xMax - xMin) * cw; }
  function ty(y) { return pad.top + (1 - (y - yMin) / (yMax - yMin)) * ch; }

  // grid lines
  ctx.strokeStyle = '#2a3a5e';
  ctx.lineWidth = 1;
  for (let y = yMin; y <= yMax; y += 0.1) {
    ctx.beginPath(); ctx.moveTo(pad.left, ty(y)); ctx.lineTo(W - pad.right, ty(y)); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '11px system-ui'; ctx.textAlign = 'right';
    ctx.fillText((y * 100).toFixed(0) + '%', pad.left - 8, ty(y) + 4);
  }

  // bars showing sample count (faint)
  const maxN = Math.max(...ns);
  const barW = cw / data.length * 0.6;
  ctx.fillStyle = 'rgba(168, 230, 160, 0.08)';
  data.forEach((d, i) => {
    const bh = (d.n / maxN) * ch * 0.3;
    ctx.fillRect(tx(d[xKey]) - barW/2, pad.top + ch - bh, barW, bh);
  });

  // line
  ctx.strokeStyle = '#a8e6a0';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  data.forEach((d, i) => {
    const x = tx(d[xKey]), y = ty(d.acc);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // dots
  data.forEach((d, i) => {
    const x = tx(d[xKey]), y = ty(d.acc);
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#a8e6a0'; ctx.fill();
    ctx.strokeStyle = '#16213e'; ctx.lineWidth = 1.5; ctx.stroke();
  });

  // x labels
  ctx.fillStyle = '#666'; ctx.font = '11px system-ui'; ctx.textAlign = 'center';
  data.forEach((d, i) => {
    if (i % 2 === 0 || data.length <= 20)
      ctx.fillText(d[xKey], tx(d[xKey]), H - pad.bottom + 16);
  });
  ctx.fillStyle = '#888'; ctx.font = '12px system-ui';
  ctx.fillText(xLabel, pad.left + cw / 2, H - 4);

  // n labels on hover area
  ctx.fillStyle = '#555'; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  data.forEach((d, i) => {
    ctx.fillText('n=' + d.n, tx(d[xKey]), ty(d.acc) - 10);
  });
}

drawChart('moveChart', moveData, 'Move number', 'x');
drawChart('pcountChart', pcountData, 'Pieces on board', 'x');
</script>
</body></html>"""

move_json = json.dumps([{'x': int(b), 'acc': round(a, 4), 'n': int(n)} for b, _, a, n in move_bins])
pcount_json = json.dumps([{'x': int(b), 'acc': round(a, 4), 'n': int(n)} for b, _, a, n in pcount_bins])

html = html.replace('MOVE_DATA', move_json)
html = html.replace('PCOUNT_DATA', pcount_json)
html = html.replace('OVERALL_ACC', f"{per_pos_occ_acc.mean():.1%}")
html = html.replace('N_POS', str(len(val_moves)))
html = html.replace('NO_CHECK_ACC', f"{check_acc['not_check']:.1%}")
html = html.replace('CHECK_ACC', f"{check_acc['in_check']:.1%}")
html = html.replace('N_CHECK', str(check_acc['n_check']))
html = html.replace('N_NO_CHECK', str(check_acc['n_no_check']))

with open(output_path, 'w') as f:
    f.write(html)
print(f"Wrote {output_path}")
