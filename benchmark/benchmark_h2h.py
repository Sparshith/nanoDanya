import argparse
import time
import torch
import chess

from game_runner import load_model, sample_with_tracking
from inference.kv_cache import KVCache


@torch.inference_mode()
def play_h2h(white_model, white_config, white_stoi, white_itos,
             black_model, black_config, black_stoi, black_itos,
             device, temperature=0.8, max_moves=200, game_num=0):
    board = chess.Board()

    white_cache = KVCache(1, white_config.n_kv_head, white_config.sequence_len,
                          white_config.n_embd // white_config.n_head, white_config.n_layer)
    black_cache = KVCache(1, black_config.n_kv_head, black_config.sequence_len,
                          black_config.n_embd // black_config.n_head, black_config.n_layer)

    # feed bos to both
    bos_w = torch.tensor([[white_stoi["<bos>"]]], device=device)
    bos_b = torch.tensor([[black_stoi["<bos>"]]], device=device)
    logits_w = white_model(bos_w, kv_cache=white_cache)
    logits_b = black_model(bos_b, kv_cache=black_cache)

    moves = []
    for _ in range(max_moves):
        if board.is_game_over():
            break

        if board.turn == chess.WHITE:
            next_id, next_token, _ = sample_with_tracking(
                logits_w[0, -1, :], board, white_stoi, white_itos,
                temperature, forbid_eos=True
            )
            moves.append(next_token)
            # feed to white
            x_w = next_id.view(1, 1).to(device)
            logits_w = white_model(x_w, kv_cache=white_cache)
            # feed to black
            tid = black_stoi.get(next_token)
            if tid is None:
                break
            x_b = torch.tensor([[tid]], device=device)
            logits_b = black_model(x_b, kv_cache=black_cache)
        else:
            next_id, next_token, _ = sample_with_tracking(
                logits_b[0, -1, :], board, black_stoi, black_itos,
                temperature, forbid_eos=True
            )
            moves.append(next_token)
            # feed to black
            x_b = next_id.view(1, 1).to(device)
            logits_b = black_model(x_b, kv_cache=black_cache)
            # feed to white
            tid = white_stoi.get(next_token)
            if tid is None:
                break
            x_w = torch.tensor([[tid]], device=device)
            logits_w = white_model(x_w, kv_cache=white_cache)

    result = board.result()
    if board.is_game_over():
        termination = board.outcome().termination.name
    else:
        termination = "max_moves"

    return {"game": game_num, "moves": len(moves), "result": result, "termination": termination}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default="models/chess_weighted_L12_H6_E768.pt")
    parser.add_argument("--model-b", default="models/chess_eval_L12_H6_E768.pt")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_a, config_a, stoi_a, itos_a = load_model(args.model_a, device)
    model_b, config_b, stoi_b, itos_b = load_model(args.model_b, device)
    print(f"Model A: {args.model_a}")
    print(f"Model B: {args.model_b}")
    print(f"Playing {args.games} games (alternating colors, temp={args.temperature})")

    a_wins, b_wins, draws = 0, 0, 0
    t0 = time.time()

    for i in range(args.games):
        if i % 2 == 0:
            # A plays white
            g = play_h2h(model_a, config_a, stoi_a, itos_a,
                         model_b, config_b, stoi_b, itos_b,
                         device, args.temperature, game_num=i)
            if g["result"] == "1-0":
                a_wins += 1
            elif g["result"] == "0-1":
                b_wins += 1
            else:
                draws += 1
            color_info = "A=white"
        else:
            # B plays white
            g = play_h2h(model_b, config_b, stoi_b, itos_b,
                         model_a, config_a, stoi_a, itos_a,
                         device, args.temperature, game_num=i)
            if g["result"] == "1-0":
                b_wins += 1
            elif g["result"] == "0-1":
                a_wins += 1
            else:
                draws += 1
            color_info = "B=white"

        done = i + 1
        print(f"[{done}/{args.games}] {color_info} result={g['result']} ({g['termination']}, {g['moves']} moves) | A:{a_wins} D:{draws} B:{b_wins}")

    elapsed = time.time() - t0
    total = args.games
    a_score = a_wins + 0.5 * draws
    b_score = b_wins + 0.5 * draws
    print(f"\nModel A ({args.model_a}): {a_wins}W {draws}D {b_wins}L  score={a_score}/{total}")
    print(f"Model B ({args.model_b}): {b_wins}W {draws}D {a_wins}L  score={b_score}/{total}")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.1f}s/game)")


if __name__ == "__main__":
    main()
