import torch
import sys
import json
import uuid
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.append(str((Path(__file__).parent / "nanochat").resolve()))
from nanochat.gpt import GPT, GPTConfig
from inference.kv_cache import KVCache
import chess
import chess.pgn


@dataclass
class MoveStats:
    raw_top1: str
    raw_top1_prob: float
    raw_top1_legal: bool


@dataclass
class GameRecord:
    id: str
    model_path: str
    timestamp: str
    prompt: str
    temperature: float
    moves: list[str]
    prompt_move_count: int
    move_stats: list[dict]
    result: str
    termination: str
    num_moves: int
    pgn: str


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = GPTConfig(**ckpt["meta"]["model_config"])
    model = GPT(config).eval()
    model.load_state_dict(ckpt["model"])
    stoi = ckpt["meta"]["tokenizer"]["stoi"]
    itos = ckpt["meta"]["tokenizer"]["itos"]
    model = model.to(device)
    return model, config, stoi, itos


def sample_with_tracking(logits, board, stoi, itos, temperature, forbid_eos=False):
    raw_top1_idx = torch.argmax(logits).item()
    raw_probs = torch.softmax(logits, dim=-1)
    raw_top1 = itos[raw_top1_idx]
    raw_top1_prob = raw_probs[raw_top1_idx].item()

    legal_san = {board.san(mv) for mv in board.legal_moves}
    raw_top1_legal = raw_top1 in legal_san or raw_top1 == "<eos>"

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

    filtered = torch.softmax(mask / temperature, dim=-1)
    next_id = torch.multinomial(filtered, num_samples=1)
    next_token = itos[next_id.item()]

    if next_token not in {"<bos>", "<eos>"}:
        board.push_san(next_token)

    stats = MoveStats(
        raw_top1=raw_top1,
        raw_top1_prob=round(raw_top1_prob, 4),
        raw_top1_legal=raw_top1_legal,
    )
    return next_id, next_token, stats


def board_result(board):
    if board.is_checkmate():
        return ("0-1" if board.turn == chess.WHITE else "1-0"), "checkmate"
    if board.is_stalemate():
        return "1/2-1/2", "stalemate"
    if board.is_insufficient_material():
        return "1/2-1/2", "insufficient_material"
    if board.is_fifty_moves():
        return "1/2-1/2", "fifty_moves"
    if board.can_claim_threefold_repetition():
        return "1/2-1/2", "threefold_repetition"
    return "1/2-1/2", "draw"


def build_pgn(moves):
    game = chess.pgn.Game()
    node = game
    board = game.board()
    for san in moves:
        move = board.parse_san(san)
        node = node.add_variation(move)
        board.push(move)
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    return game.accept(exporter).strip()


@torch.inference_mode()
def play_games(model, config, stoi, itos, device, model_path, n_games, batch_size,
               prompt="<bos>", temperature=0.8, max_moves=200, max_eos=5):
    all_records = []
    prompt_tokens = prompt.strip().split()
    prompt_ids = [stoi[token] for token in prompt_tokens]
    prompt_moves = [t for t in prompt_tokens if t not in {"<bos>", "<eos>"}]
    prompt_move_count = len(prompt_moves)
    eos_id = stoi["<eos>"]

    for chunk_start in range(0, n_games, batch_size):
        chunk_size = min(batch_size, n_games - chunk_start)

        boards = []
        for _ in range(chunk_size):
            board = chess.Board()
            for m in prompt_moves:
                board.push_san(m)
            boards.append(board)

        kv_cache = KVCache(
            batch_size=chunk_size,
            num_heads=config.n_kv_head,
            seq_len=config.sequence_len,
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer,
        )

        x = torch.tensor([prompt_ids] * chunk_size, device=device)
        logits = model(x, kv_cache=kv_cache)

        generated = [prompt_tokens[:] for _ in range(chunk_size)]
        move_stats = [[] for _ in range(chunk_size)]
        eos_counts = [0] * chunk_size
        finished = [False] * chunk_size
        terminations = [""] * chunk_size
        results = ["*"] * chunk_size

        for step in range(max_moves):
            if all(finished):
                break

            next_ids = []
            for i in range(chunk_size):
                if finished[i]:
                    next_ids.append(eos_id)
                    continue

                if boards[i].is_game_over():
                    results[i], terminations[i] = board_result(boards[i])
                    finished[i] = True
                    next_ids.append(eos_id)
                    continue

                game_logits = logits[i, -1, :]
                next_id, next_token, stats = sample_with_tracking(
                    game_logits, boards[i], stoi, itos, temperature,
                    forbid_eos=(eos_counts[i] >= max_eos)
                )

                move_stats[i].append(asdict(stats))
                generated[i].append(next_token)

                if next_token == "<eos>":
                    eos_counts[i] += 1
                    if eos_counts[i] >= max_eos:
                        finished[i] = True
                        terminations[i] = "eos_limit"

                next_ids.append(next_id.item())

            x = torch.tensor(next_ids, device=device).unsqueeze(1)
            logits = model(x, kv_cache=kv_cache)

        for i in range(chunk_size):
            if not finished[i]:
                finished[i] = True
                terminations[i] = "max_moves"

        now = datetime.now().isoformat()
        for i in range(chunk_size):
            game_moves = [t for t in generated[i] if t not in {"<bos>", "<eos>"}]
            record = GameRecord(
                id=str(uuid.uuid4()),
                model_path=model_path,
                timestamp=now,
                prompt=prompt,
                temperature=temperature,
                moves=game_moves,
                prompt_move_count=prompt_move_count,
                move_stats=move_stats[i],
                result=results[i],
                termination=terminations[i],
                num_moves=len(game_moves),
                pgn=build_pgn(game_moves),
            )
            all_records.append(record)

        print(f"  completed {min(chunk_start + chunk_size, n_games)}/{n_games} games")

    return all_records


def save_games(games, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for game in games:
            d = asdict(game)
            f.write(json.dumps(d, default=str) + "\n")


def load_games(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(GameRecord(**json.loads(line)))
    return records


def print_summary(games):
    n = len(games)
    print(f"\n{'='*50}")
    print(f"Summary ({n} games)")
    print(f"{'='*50}")

    total_stats = 0
    illegal_count = 0
    for g in games:
        for s in g.move_stats:
            total_stats += 1
            if not s["raw_top1_legal"]:
                illegal_count += 1
    if total_stats > 0:
        print(f"Illegal move rate: {illegal_count/total_stats*100:.1f}% ({illegal_count}/{total_stats})")

    avg_len = sum(g.num_moves for g in games) / n
    print(f"Avg game length: {avg_len:.1f} moves")

    print("\nTerminations:")
    terms = {}
    for g in games:
        terms[g.termination] = terms.get(g.termination, 0) + 1
    for t, count in sorted(terms.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count} ({count/n*100:.0f}%)")

    print("\nResults:")
    res = {}
    for g in games:
        res[g.result] = res.get(g.result, 0) + 1
    labels = {"1-0": "White wins", "0-1": "Black wins", "1/2-1/2": "Draws", "*": "Unfinished"}
    for r in ["1-0", "0-1", "1/2-1/2", "*"]:
        count = res.get(r, 0)
        if count > 0:
            print(f"  {labels[r]}: {count} ({count/n*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run chess games with nanoDanya model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-games", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--prompt", default="<bos>")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-moves", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from {args.model}...")
    model, config, stoi, itos = load_model(args.model, device)

    print(f"Playing {args.n_games} games (batch_size={args.batch_size}, temp={args.temperature})...")
    t0 = time.time()
    games = play_games(
        model, config, stoi, itos, device,
        model_path=args.model,
        n_games=args.n_games,
        batch_size=args.batch_size,
        prompt=args.prompt,
        temperature=args.temperature,
        max_moves=args.max_moves,
    )
    elapsed = time.time() - t0
    print(f"\nGenerated {len(games)} games in {elapsed:.1f}s ({elapsed/len(games):.2f}s/game)")

    save_games(games, args.output)
    print(f"Saved to {args.output}")
    print_summary(games)


if __name__ == "__main__":
    main()
