import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import chess
import chess.engine
from pathlib import Path
from collections import defaultdict
from nanochat.gpt import GPT, GPTConfig

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/chess_weighted_L12_H6_E768.pt"
data_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/val.bin"
meta_path = sys.argv[3] if len(sys.argv) > 3 else "data/processed/meta.pkl"
output_path = Path(__file__).parent / "probe_eval.html"
stockfish_path = "/opt/homebrew/bin/stockfish"
probe_layers = [0, 3, 6, 9, 11]
n_train_games = 400
n_val_games = 100
probe_epochs = 20
probe_lr = 1e-3
batch_size = 256
sf_depth = 8

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
print(f"Using {len(train_games)} train, {len(val_games)} val games")

# --- collect hidden states + board positions ---
_hook_out = {}
hooks = []
for li in probe_layers:
    def make_hook(l):
        def fn(mod, inp, out):
            _hook_out[l] = out.detach().cpu()
        return fn
    hooks.append(model.transformer.h[li].register_forward_hook(make_hook(li)))

def collect_boards(game_list, label):
    H = {l: [] for l in probe_layers}
    boards_all = []
    move_nums = []
    for gi, tokens in enumerate(game_list):
        board = chess.Board()
        board_states = [board.copy()]
        for tid in tokens[1:]:
            try:
                board.push_san(itos[tid])
            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                break
            board_states.append(board.copy())
        n = len(board_states)
        with torch.no_grad():
            model(torch.tensor([tokens[:n]], dtype=torch.long, device=device))
        for li in probe_layers:
            H[li].append(_hook_out[li].squeeze(0)[:n])
        for mi, b in enumerate(board_states):
            boards_all.append(b)
            move_nums.append(mi)
        if (gi + 1) % 100 == 0:
            print(f"  [{label}] {gi+1}/{len(game_list)} games, {len(boards_all)} positions")
    print(f"  [{label}] done: {len(boards_all)} positions")
    return {l: torch.cat(H[l]) for l in probe_layers}, boards_all, np.array(move_nums)

print("\nCollecting hidden states...")
train_h, train_boards, train_moves = collect_boards(train_games, "train")
val_h, val_boards, val_moves = collect_boards(val_games, "val")
for h in hooks:
    h.remove()
del model, ckpt

# --- run stockfish evals ---
def eval_positions(boards, label):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    evals = []
    for i, board in enumerate(boards):
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                cp = 10000
            elif result == "0-1":
                cp = -10000
            else:
                cp = 0
        else:
            info = engine.analyse(board, chess.engine.Limit(depth=sf_depth))
            score = info['score'].white()
            cp = score.score(mate_score=10000)
        evals.append(cp)
        if (i + 1) % 5000 == 0:
            print(f"  [{label}] {i+1}/{len(boards)} positions evaluated")
    engine.quit()
    print(f"  [{label}] done: {len(evals)} evals")
    return np.array(evals, dtype=np.float32)

print("\nRunning Stockfish evaluations (depth 8)...")
train_evals = eval_positions(train_boards, "train")
val_evals = eval_positions(val_boards, "val")
del train_boards, val_boards

print(f"Train eval range: [{train_evals.min():.0f}, {train_evals.max():.0f}] cp")
print(f"Val eval range: [{val_evals.min():.0f}, {val_evals.max():.0f}] cp")

# --- build labels ---
# binary: white winning (eval > 50cp) vs black winning (eval < -50cp), skip draws
DRAW_MARGIN = 50
train_side = np.sign(train_evals)  # -1, 0, 1
val_side = np.sign(val_evals)

# binary classification: white advantage vs black advantage (skip near-equal)
train_winning_mask = np.abs(train_evals) > DRAW_MARGIN
val_winning_mask = np.abs(val_evals) > DRAW_MARGIN
train_white_winning = (train_evals[train_winning_mask] > 0).astype(np.float32)
val_white_winning = (val_evals[val_winning_mask] > 0).astype(np.float32)

# eval buckets: losing (<-200), slight disadv (-200 to -50), equal (-50 to 50), slight adv (50 to 200), winning (>200)
def eval_to_bucket(evals):
    buckets = np.zeros(len(evals), dtype=np.int64)
    buckets[evals < -200] = 0
    buckets[(evals >= -200) & (evals < -50)] = 1
    buckets[(evals >= -50) & (evals <= 50)] = 2
    buckets[(evals > 50) & (evals <= 200)] = 3
    buckets[evals > 200] = 4
    return buckets

train_buckets = eval_to_bucket(train_evals)
val_buckets = eval_to_bucket(val_evals)
bucket_names = ["losing (<-200)", "slight disadv", "equal", "slight adv", "winning (>200)"]
print(f"Val bucket distribution: {[f'{bucket_names[i]}: {(val_buckets==i).sum()}' for i in range(5)]}")

# clipped regression target (centipawns, clipped to +/- 1500)
train_cp = np.clip(train_evals, -1500, 1500) / 1500.0  # normalize to [-1, 1]
val_cp = np.clip(val_evals, -1500, 1500) / 1500.0

# --- train probes ---
def train_probe(name, probe, train_x, train_y, val_x, val_y, loss_fn, epochs=probe_epochs):
    opt = torch.optim.AdamW(probe.parameters(), lr=probe_lr)
    n = train_x.shape[0]
    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(n)
        total_loss, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(probe(train_x[idx]), train_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch == 0 or epoch == epochs - 1:
            probe.eval()
            with torch.no_grad():
                vl = loss_fn(probe(val_x), val_y)
            print(f"    {name} epoch {epoch:2d}: train={total_loss/n_batches:.4f} val={vl.item():.4f}")
    return probe

results = {}
for li in probe_layers:
    print(f"\n=== Layer {li} ===")
    th, vh = train_h[li], val_h[li]

    # 1. who's winning (binary, skip draws)
    th_w = th[train_winning_mask]
    vh_w = vh[val_winning_mask]
    ty_w = torch.tensor(train_white_winning).unsqueeze(1)
    vy_w = torch.tensor(val_white_winning).unsqueeze(1)
    probe = train_probe("winning", nn.Linear(n_embd, 1), th_w, ty_w, vh_w, vy_w,
                        lambda o, t: F.binary_cross_entropy_with_logits(o, t))
    with torch.no_grad():
        pred = (probe(vh_w).sigmoid() > 0.5).float()
        winning_acc = (pred == vy_w).float().mean().item()
    print(f"    -> who's winning acc: {winning_acc:.4f}")
    del probe

    # 2. eval buckets (5-class)
    ty_b = torch.tensor(train_buckets)
    vy_b = torch.tensor(val_buckets)
    probe = train_probe("buckets", nn.Linear(n_embd, 5), th, ty_b, vh, vy_b,
                        lambda o, t: F.cross_entropy(o, t))
    with torch.no_grad():
        pred = probe(vh).argmax(dim=1)
        bucket_acc = (pred == vy_b).float().mean().item()
    print(f"    -> eval bucket acc: {bucket_acc:.4f}")
    del probe

    # 3. centipawn regression
    ty_cp = torch.tensor(train_cp).unsqueeze(1)
    vy_cp = torch.tensor(val_cp).unsqueeze(1)
    probe = train_probe("cp_regr", nn.Linear(n_embd, 1), th, ty_cp, vh, vy_cp,
                        lambda o, t: F.mse_loss(o, t))
    with torch.no_grad():
        pred_cp = probe(vh)
        mse = F.mse_loss(pred_cp, vy_cp).item()
        # correlation
        p = pred_cp.squeeze().numpy()
        t = vy_cp.squeeze().numpy()
        corr = np.corrcoef(p, t)[0, 1]
    print(f"    -> regression MSE: {mse:.4f}, correlation: {corr:.4f}")

    # accuracy by move number for winning probe
    move_accs = defaultdict(list)
    val_moves_w = val_moves[val_winning_mask]
    with torch.no_grad():
        # retrain quickly just for this
        pass
    # use pred from above
    pred_flat = pred.squeeze().numpy()
    truth_flat = vy_w.squeeze().numpy()
    for mi, p_, t_ in zip(val_moves_w, pred_flat, truth_flat):
        move_accs[(mi // 10) * 10].append(float(p_ == t_))

    del probe

    results[li] = dict(winning=winning_acc, buckets=bucket_acc, mse=mse, corr=corr,
                        move_accs={k: (np.mean(v), len(v)) for k, v in sorted(move_accs.items()) if len(v) >= 10})

# --- results table ---
print("\n" + "=" * 72)
print(f"{'Layer':>5} | {'Who winning':>11} | {'Eval bucket':>11} | {'Regr MSE':>8} | {'Corr':>6}")
print("-" * 72)
majority = max(val_white_winning.mean(), 1 - val_white_winning.mean())
bucket_majority = max(np.bincount(val_buckets) / len(val_buckets))
for li in probe_layers:
    r = results[li]
    print(f"{li:>5} | {r['winning']:>11.4f} | {r['buckets']:>11.4f} | {r['mse']:>8.4f} | {r['corr']:>6.4f}")
print("-" * 72)
print(f"{'base':>5} | {majority:>11.4f} | {bucket_majority:>11.4f} | {'n/a':>8} | {'n/a':>6}")
print("=" * 72)

# --- write HTML ---
html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Eval Probe Analysis</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 40px; }
h1 { color: #fff; font-size: 24px; margin-bottom: 4px; }
.subtitle { color: #888; font-size: 14px; margin-bottom: 30px; }
table { border-collapse: collapse; margin-bottom: 32px; }
th, td { padding: 8px 16px; text-align: right; border-bottom: 1px solid #2a3a5e; }
th { color: #888; font-size: 12px; text-transform: uppercase; }
td { font-size: 14px; font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: 2px solid #2a3a5e; }
.highlight { color: #a8e6a0; font-weight: 600; }
.chart-section { margin-top: 36px; }
.chart-title { font-size: 16px; font-weight: 600; margin-bottom: 8px; }
.chart-desc { font-size: 12px; color: #888; margin-bottom: 16px; }
.charts { display: flex; flex-wrap: wrap; gap: 24px; }
.chart-box { background: #16213e; border-radius: 12px; padding: 20px; width: 480px; }
.chart-box-title { font-size: 13px; font-weight: 600; margin-bottom: 12px; }
canvas { display: block; }
</style></head><body>
<h1>Does the model know who's winning?</h1>
<div class="subtitle">Linear probes predicting Stockfish evaluation (depth 8) from hidden states</div>

<table>
<tr><th>Layer</th><th>Who's winning</th><th>Eval buckets (5)</th><th>Regression MSE</th><th>Correlation</th></tr>
"""

for li in probe_layers:
    r = results[li]
    best_w = max(results[l]['winning'] for l in probe_layers)
    best_b = max(results[l]['buckets'] for l in probe_layers)
    best_c = max(results[l]['corr'] for l in probe_layers)
    w_cls = ' class="highlight"' if r['winning'] == best_w else ''
    b_cls = ' class="highlight"' if r['buckets'] == best_b else ''
    c_cls = ' class="highlight"' if r['corr'] == best_c else ''
    html += f'<tr><td>{li}</td><td{w_cls}>{r["winning"]:.4f}</td><td{b_cls}>{r["buckets"]:.4f}</td>'
    html += f'<td>{r["mse"]:.4f}</td><td{c_cls}>{r["corr"]:.4f}</td></tr>\n'

html += f'<tr><td>base</td><td>{majority:.4f}</td><td>{bucket_majority:.4f}</td><td>-</td><td>-</td></tr>'
html += '</table>\n'

# per-layer winning accuracy by move number
html += '<div class="chart-section"><div class="chart-title">Who\'s winning accuracy by move number</div>'
html += '<div class="chart-desc">Can the model tell who\'s ahead at different stages of the game?</div>'
html += '<div class="charts">\n'

chart_data = {}
for li in probe_layers:
    pts = []
    for mv, (acc, n) in sorted(results[li]['move_accs'].items()):
        pts.append({'x': int(mv), 'acc': round(acc, 4), 'n': int(n)})
    chart_data[li] = pts

for li in probe_layers:
    html += f'<div class="chart-box"><div class="chart-box-title">Layer {li}</div>'
    html += f'<canvas id="chart{li}" width="440" height="220"></canvas></div>\n'

html += '</div></div>\n'

html += '<script>\nconst chartData = ' + json.dumps(chart_data) + ';\n'
html += """
function drawChart(canvasId, data) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = { top: 16, right: 16, bottom: 36, left: 44 };
  const cw = W - pad.left - pad.right, ch = H - pad.top - pad.bottom;
  ctx.fillStyle = '#16213e'; ctx.fillRect(0, 0, W, H);
  if (!data.length) return;
  const xs = data.map(d => d.x), ys = data.map(d => d.acc);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(0.5, Math.min(...ys) - 0.05), yMax = 1.0;
  function tx(x) { return pad.left + (x - xMin) / (xMax - xMin || 1) * cw; }
  function ty(y) { return pad.top + (1 - (y - yMin) / (yMax - yMin)) * ch; }
  ctx.strokeStyle = '#2a3a5e'; ctx.lineWidth = 1;
  for (let y = 0.5; y <= 1.0; y += 0.1) {
    ctx.beginPath(); ctx.moveTo(pad.left, ty(y)); ctx.lineTo(W-pad.right, ty(y)); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
    ctx.fillText((y*100).toFixed(0)+'%', pad.left-6, ty(y)+3);
  }
  // 50% baseline
  ctx.strokeStyle = '#d66'; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(pad.left, ty(0.5)); ctx.lineTo(W-pad.right, ty(0.5)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = '#a8e6a0'; ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((d,i) => { const x=tx(d.x), y=ty(d.acc); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
  ctx.stroke();
  data.forEach(d => {
    ctx.beginPath(); ctx.arc(tx(d.x), ty(d.acc), 3, 0, Math.PI*2);
    ctx.fillStyle='#a8e6a0'; ctx.fill();
  });
  ctx.fillStyle='#666'; ctx.font='10px system-ui'; ctx.textAlign='center';
  data.forEach((d,i) => { if(i%2===0) ctx.fillText(d.x, tx(d.x), H-pad.bottom+14); });
  ctx.fillStyle='#888'; ctx.font='11px system-ui';
  ctx.fillText('Move number', pad.left+cw/2, H-2);
}
Object.entries(chartData).forEach(([li, data]) => drawChart('chart'+li, data));
</script></body></html>"""

with open(output_path, 'w') as f:
    f.write(html)
print(f"\nWrote {output_path}")
