import torch
import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent.parent / "nanochat").resolve()))
from nanochat.gpt import GPT, GPTConfig
from kv_cache import KVCache
import chess
import chess.pgn

ckpt = torch.load("models/chess_L12_H6_E768.pt", map_location="cpu", weights_only=False)
config = GPTConfig(**ckpt["meta"]["model_config"])
model = GPT(config).eval()
model.load_state_dict(ckpt["model"])
stoi = ckpt["meta"]["tokenizer"]["stoi"]
itos = ckpt["meta"]["tokenizer"]["itos"]
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)


def sample_next_token(logits, board, stoi, itos, forbid_eos=False):
    legal_san = {board.san(mv) for mv in board.legal_moves}
    mask = torch.full((len(itos),), float("-inf"), device=logits.device)
    for token, idx in stoi.items():
        if token == "<bos>":
            continue
        elif token == "<eos>":
            if forbid_eos:
                continue
            mask[idx] = logits[idx]
        elif token in legal_san:
            mask[idx] = logits[idx]
    filtered = torch.softmax(mask, dim=-1)
    next_id = torch.multinomial(filtered, num_samples=1)
    next_token = itos[next_id.item()]
    if next_token not in {"<bos>", "<eos>"}:
        board.push_san(next_token)
    return next_id, next_token


prompt_text = "<bos> d4 d5"
prompt_tokens = prompt_text.strip().split()
prompt_ids = [stoi[token] for token in prompt_tokens]

board = chess.Board()
for token in prompt_tokens:
    if token in {"<bos>", "<eos>"}:
        continue
    board.push_san(token)

kv_cache = KVCache(
    batch_size=1,
    num_heads=config.n_kv_head,
    seq_len=config.sequence_len,
    head_dim=config.n_embd // config.n_head,
    num_layers=config.n_layer,
)

x = torch.tensor([prompt_ids], device=device)
logits = model(x, kv_cache=kv_cache)

generated = prompt_tokens[:]
eos_cnt = 0
max_eos_cnt = 5

for _ in range(200):
    next_id, next_token = sample_next_token(logits[:, -1, :].squeeze(0), board, stoi, itos, forbid_eos=eos_cnt >= max_eos_cnt)
    generated.append(next_token)

    if next_token == "<eos>":
        eos_cnt += 1
        if eos_cnt >= max_eos_cnt:
            break

    x = next_id.view(1, 1).to(device)
    logits = model(x, kv_cache=kv_cache)

game = chess.pgn.Game()
node = game
board = game.board()
for token in generated:
    if token in {"<bos>", "<eos>"}:
        continue
    move = board.parse_san(token)
    node = node.add_variation(move)
    board.push(move)
exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
print(game.accept(exporter))
