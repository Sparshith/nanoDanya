import argparse
import json
import pickle
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm


def choose_sample_lines(games_path: Path, sample: int, mode: str, seed: int) -> set[int] | None:
    if sample <= 0:
        return None
    if mode == "head":
        return set(range(sample))
    if mode != "random":
        raise ValueError(f"Unsupported sample mode: {mode}")

    rng = random.Random(seed)
    reservoir: list[int] = []
    seen = 0
    with open(games_path) as f:
        for line_num, line in enumerate(tqdm(f, desc="Sampling games")):
            if not line.strip():
                continue
            seen += 1
            if len(reservoir) < sample:
                reservoir.append(line_num)
            else:
                j = rng.randrange(seen)
                if j < sample:
                    reservoir[j] = line_num
    print(f"Selected {len(reservoir)} line numbers from {seen} candidate games using mode={mode}, seed={seed}")
    return set(reservoir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=0, help='process only first N games')
    parser.add_argument('--sample-mode', choices=('head', 'random'), default='head', help='how to choose sampled games')
    parser.add_argument('--sample-seed', type=int, default=0, help='seed used when --sample-mode=random')
    parser.add_argument('--weight', type=float, default=5.0, help='weight for puzzle solution moves')
    parser.add_argument('--games', type=str, default='puzzle_games_ndjson.txt')
    parser.add_argument('--metadata', type=str, default='puzzle_metadata_full.txt')
    parser.add_argument('--vocab', type=str, default='data/processed/meta.pkl')
    parser.add_argument('--out', type=str, default='data/puzzle_weighted')
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    games_path = root / args.games
    meta_path = root / args.metadata
    vocab_path = root / args.vocab
    out_dir = root / args.out
    selected_lines = choose_sample_lines(games_path, args.sample, args.sample_mode, args.sample_seed)

    meta = pickle.loads(vocab_path.read_bytes())
    stoi = dict(meta['stoi'])
    itos = list(meta['itos'])
    bos_id = stoi['<bos>']
    eos_id = stoi['<eos>']

    print("Loading puzzle metadata...")
    with open(meta_path) as f:
        puzzles_by_game = json.load(f)
    print(f"Loaded puzzles for {len(puzzles_by_game)} games")

    # first pass: collect OOV tokens so we can extend vocab
    print("Scanning for OOV tokens...")
    new_tokens = set()
    with open(games_path) as f:
        for line_num, line in enumerate(tqdm(f, desc="OOV scan")):
            if selected_lines is not None and line_num not in selected_lines:
                continue
            line = line.strip()
            if not line:
                continue
            moves = json.loads(line).get('moves', '').split()
            for m in moves:
                if m not in stoi:
                    new_tokens.add(m)
    if new_tokens:
        for tok in sorted(new_tokens):
            stoi[tok] = len(itos)
            itos.append(tok)
        print(f"Added {len(new_tokens)} new tokens, vocab: {meta['vocab_size']} -> {len(itos)}")

    all_tokens = []
    all_weights = []
    skipped_oov = 0
    skipped_no_moves = 0
    total_weighted = 0
    total_tokens = 0
    n_games = 0
    n_with_puzzles = 0
    printed_examples = 0

    with open(games_path) as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing games")):
            if selected_lines is not None and line_num not in selected_lines:
                continue

            line = line.strip()
            if not line:
                continue

            game = json.loads(line)
            game_id = game['id']
            moves_str = game.get('moves', '')
            if not moves_str:
                skipped_no_moves += 1
                continue

            san_moves = moves_str.split()

            token_ids = [bos_id]
            oov = False
            for m in san_moves:
                if m not in stoi:
                    oov = True
                    break
                token_ids.append(stoi[m])
            if oov:
                skipped_oov += 1
                continue
            token_ids.append(eos_id)

            weights = [1.0] * len(token_ids)
            game_weighted = 0

            if game_id in puzzles_by_game:
                n_with_puzzles += 1
                for puzzle in puzzles_by_game[game_id]:
                    move_num = puzzle.get('move_num')
                    if move_num is None:
                        continue
                    puzzle_uci = puzzle['moves'].split()
                    # solution = puzzle_uci[1:], at SAN indices move_num .. move_num+len(puzzle_uci)-2
                    # token positions = SAN index + 1 (for <bos>)
                    for k in range(1, len(puzzle_uci)):
                        san_idx = move_num + k - 1
                        token_pos = san_idx + 1
                        if 0 < token_pos < len(token_ids):
                            weights[token_pos] = args.weight
                            game_weighted += 1

                if printed_examples < 3 and game_weighted > 0:
                    printed_examples += 1
                    print(f"\nExample: game {game_id} ({len(san_moves)} moves, {game_weighted} weighted)")
                    for puzzle in puzzles_by_game[game_id]:
                        move_num = puzzle.get('move_num')
                        if move_num is None:
                            continue
                        puzzle_uci = puzzle['moves'].split()
                        print(f"  Puzzle {puzzle['puzzle_id']} (rating {puzzle['rating']}, move_num={move_num}):")
                        setup_idx = move_num - 1
                        if 0 <= setup_idx < len(san_moves):
                            print(f"    Setup: moves[{setup_idx}] = {san_moves[setup_idx]}")
                        for k in range(1, len(puzzle_uci)):
                            san_idx = move_num + k - 1
                            token_pos = san_idx + 1
                            san = san_moves[san_idx] if san_idx < len(san_moves) else '??'
                            tok = itos[token_ids[token_pos]] if token_pos < len(token_ids) else '??'
                            w = weights[token_pos] if token_pos < len(weights) else '??'
                            print(f"    Solution: moves[{san_idx}] = {san} (token: {tok}, weight: {w})")

            all_tokens.extend(token_ids)
            all_weights.extend(weights)
            total_weighted += game_weighted
            total_tokens += len(token_ids)
            n_games += 1

    print(f"\nProcessed {n_games} games ({n_with_puzzles} with puzzles)")
    print(f"Skipped: {skipped_oov} OOV, {skipped_no_moves} no moves")
    print(f"Total tokens: {total_tokens}, weighted: {total_weighted} ({100*total_weighted/max(total_tokens,1):.3f}%)")
    avg_weight = (total_tokens - total_weighted + total_weighted * args.weight) / max(total_tokens, 1)
    print(f"Average weight: {avg_weight:.4f}")

    tokens = np.array(all_tokens, dtype=np.uint16)
    weights = np.array(all_weights, dtype=np.float32)

    split = int(len(tokens) * 0.9)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens[:split].tofile(out_dir / 'train.bin')
    tokens[split:].tofile(out_dir / 'val.bin')
    weights[:split].tofile(out_dir / 'train_weights.bin')
    weights[split:].tofile(out_dir / 'val_weights.bin')
    out_meta = {
        'vocab_size': len(itos),
        'context_length': meta['context_length'],
        'itos': itos,
        'stoi': stoi,
    }
    with open(out_dir / 'meta.pkl', 'wb') as f:
        pickle.dump(out_meta, f)

    print(f"\nSaved to {out_dir}/")
    print(f"  train: {split} tokens ({tokens[:split].nbytes/1e6:.1f}MB tokens, {weights[:split].nbytes/1e6:.1f}MB weights)")
    print(f"  val: {len(tokens)-split} tokens ({tokens[split:].nbytes/1e6:.1f}MB tokens, {weights[split:].nbytes/1e6:.1f}MB weights)")


if __name__ == '__main__':
    main()
