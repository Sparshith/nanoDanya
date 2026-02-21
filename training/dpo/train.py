import torch
import torch.nn.functional as F
import sys
import json
import random
from pathlib import Path
from copy import deepcopy

sys.path.append(str((Path(__file__).parent.parent.parent / "nanochat").resolve()))
from nanochat.gpt import GPT, GPTConfig
import chess

# --- config ---
model_path = "models/chess_L12_H6_E768.pt"
data_path = str(Path(__file__).parent / "games/12L_dpo_data.jsonl")
output_path = "models/chess_L12_H6_E768_dpo.pt"
lr = 1e-5
beta = 0.5
max_steps = 200
eval_interval = 50

# --- load model ---
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
config = GPTConfig(**ckpt["meta"]["model_config"])
stoi = ckpt["meta"]["tokenizer"]["stoi"]
itos = ckpt["meta"]["tokenizer"]["itos"]

model = GPT(config)
model.load_state_dict(ckpt["model"])
model = model.to(device).train()

ref_model = deepcopy(model)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# --- load games ---
games = []
with open(data_path) as f:
    for line in f:
        g = json.loads(line)
        games.append(g)
random.shuffle(games)
print(f"Loaded {len(games)} games")

# --- training loop ---
total_pairs = 0
skipped = 0

for step in range(1, max_steps + 1):
    game = games[(step - 1) % len(games)]
    moves = game["moves"]
    prompt_move_count = game["prompt_move_count"]

    token_ids = [stoi["<bos>"]] + [stoi[m] for m in moves]
    x = torch.tensor([token_ids], device=device)

    with torch.no_grad():
        ref_logits = ref_model(x)[0]

    # find positions where ref model's argmax is illegal
    board = chess.Board()
    for m in moves[:prompt_move_count]:
        board.push_san(m)

    pairs = []
    for i in range(prompt_move_count, len(moves)):
        if i >= ref_logits.size(0):
            break
        top1 = itos[torch.argmax(ref_logits[i]).item()]
        legal_san = {board.san(mv) for mv in board.legal_moves}
        if top1 not in legal_san and top1 != "<eos>":
            pairs.append((i, stoi[moves[i]], stoi[top1]))
        board.push_san(moves[i])

    if not pairs:
        skipped += 1
        continue

    train_logits = model(x)[0]

    loss = torch.tensor(0.0, device=device)
    for pos, chosen_id, rejected_id in pairs:
        train_lp = F.log_softmax(train_logits[pos], dim=-1)
        ref_lp = F.log_softmax(ref_logits[pos], dim=-1)
        delta = (train_lp[chosen_id] - ref_lp[chosen_id]) - (train_lp[rejected_id] - ref_lp[rejected_id])
        loss = loss - F.logsigmoid(beta * delta)
    loss = loss / len(pairs)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    total_pairs += len(pairs)

    if step % eval_interval == 0 or step == 1:
        print(f'step {step:04d}/{max_steps} loss {loss.item():.4f} pairs {len(pairs)} total_pairs {total_pairs} skipped {skipped}')

    if step % 1000 == 0 or step == max_steps:
        torch.save({
            'model': model.state_dict(),
            'meta': ckpt["meta"],
            'step': step,
        }, output_path)
        print(f'Saved checkpoint to {output_path} at step {step}')
