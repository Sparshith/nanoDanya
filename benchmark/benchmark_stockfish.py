import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import chess
import chess.engine
import torch

from game_runner import load_model, sample_with_tracking
from inference.kv_cache import KVCache


@dataclass
class BenchmarkGame:
    game_num: int
    model_color: str
    moves: list[str]
    result: str
    outcome: str
    termination: str
    num_moves: int


@torch.inference_mode()
def play_vs_stockfish(model, config, stoi, itos, device, sf_path, elo,
                      model_color=chess.WHITE, temperature=0.8, max_moves=200, game_num=0):
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})

    board = chess.Board()
    kv_cache = KVCache(1, config.n_kv_head, config.sequence_len,
                       config.n_embd // config.n_head, config.n_layer)

    x = torch.tensor([[stoi["<bos>"]]], device=device)
    logits = model(x, kv_cache=kv_cache)
    moves = []
    termination = ""
    for _ in range(max_moves):
        if board.is_game_over():
            break

        if board.turn == model_color:
            next_id, next_token, _ = sample_with_tracking(
                logits[0, -1, :], board, stoi, itos, temperature,
                forbid_eos=True
            )
            moves.append(next_token)
            x = next_id.view(1, 1).to(device)
            logits = model(x, kv_cache=kv_cache)
        else:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            san = board.san(result.move)
            board.push(result.move)
            moves.append(san)
            token_id = stoi.get(san)
            if token_id is None:
                termination = "oov_stockfish"
                break
            x = torch.tensor([[token_id]], device=device)
            logits = model(x, kv_cache=kv_cache)

    engine.quit()

    if not termination:
        if board.is_game_over():
            termination = board.outcome().termination.name
        else:
            termination = "max_moves"

    board_result = board.result()
    if board_result == "1-0":
        outcome = "win" if model_color == chess.WHITE else "loss"
    elif board_result == "0-1":
        outcome = "loss" if model_color == chess.WHITE else "win"
    else:
        outcome = "draw"

    return BenchmarkGame(
        game_num=game_num,
        model_color="white" if model_color == chess.WHITE else "black",
        moves=moves,
        result=board_result,
        outcome=outcome,
        termination=termination,
        num_moves=len(moves),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/chess_weighted_L12_H6_E768.pt")
    parser.add_argument("--stockfish", default="/opt/homebrew/bin/fairy-stockfish")
    parser.add_argument("--elo", type=int, default=500)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output", default="benchmark/benchmark_results.jsonl")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, config, stoi, itos = load_model(args.model, device)
    print(f"Loaded {args.model} on {device}")
    print(f"Playing {args.games} games vs fairy-stockfish Elo {args.elo} ({args.workers} workers)")

    t0 = time.time()
    results = {"win": 0, "loss": 0, "draw": 0}
    terminations = {}
    all_games = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for i in range(args.games):
            color = chess.WHITE if i % 2 == 0 else chess.BLACK
            f = pool.submit(
                play_vs_stockfish, model, config, stoi, itos, device,
                args.stockfish, args.elo, color, args.temperature, 200, i
            )
            futures[f] = i

        for f in as_completed(futures):
            game = f.result()
            all_games.append(game)
            results[game.outcome] += 1
            terminations[game.termination] = terminations.get(game.termination, 0) + 1
            done = len(all_games)
            wr = (results["win"] + 0.5 * results["draw"]) / done
            print(f"[{done}/{args.games}] Game {game.game_num} [{game.model_color[0].upper()}]: {game.outcome} ({game.termination}, {game.num_moves} moves) | W:{results['win']} D:{results['draw']} L:{results['loss']} WR:{wr:.1%}")

    elapsed = time.time() - t0
    total = args.games
    wr = (results["win"] + 0.5 * results["draw"]) / total

    print(f"\nFinal: {results['win']}W {results['draw']}D {results['loss']}L ({wr:.1%}) vs Stockfish {args.elo}")
    print(f"Terminations: {terminations}")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.1f}s/game)")

    all_games.sort(key=lambda g: g.game_num)
    with open(args.output, "w") as f:
        for g in all_games:
            f.write(json.dumps(asdict(g)) + "\n")
    print(f"Saved {len(all_games)} games to {args.output}")


if __name__ == "__main__":
    main()
