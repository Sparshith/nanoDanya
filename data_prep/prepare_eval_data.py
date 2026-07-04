import json
import pickle
import numpy as np
from pathlib import Path

data_dir = Path(__file__).parent.parent / 'data'
input_path = data_dir / 'puzzle_games_ndjson.txt'
output_dir = data_dir / 'eval'
output_dir.mkdir(exist_ok=True)

context_length = 512
alpha = 4.0

min_freq = 5


def strip_annotations(move):
    return move.rstrip('+#')


# pass 1: build vocab from scratch, count move frequencies
print("Pass 1: scanning for unique moves...")
from collections import Counter
move_counts = Counter()
n_games = 0
with open(input_path) as f:
    for line in f:
        game = json.loads(line)
        move_counts.update(strip_annotations(m) for m in game['moves'].split())
        n_games += 1
        if n_games % 500000 == 0:
            print(f"  scanned {n_games} games, {len(move_counts)} unique moves so far")

rare_moves = {m for m, c in move_counts.items() if c < min_freq}
sorted_moves = sorted(m for m in move_counts if m not in rare_moves)
stoi = {'<bos>': 0, '<eos>': 1}
for i, m in enumerate(sorted_moves):
    stoi[m] = i + 2
itos = {v: k for k, v in stoi.items()}
vocab_size = len(stoi)
bos_id = stoi['<bos>']
eos_id = stoi['<eos>']
print(f"Total unique moves: {len(move_counts)}, pruned {len(rare_moves)} with freq < {min_freq}")
print(f"Vocab: {vocab_size} tokens ({len(sorted_moves)} moves + bos/eos)")

# pass 2: tokenize, extract evals, compute weights
print("\nPass 2: tokenizing and extracting evals...")
all_tokens = []
all_evals = []
all_weights = []
skipped_oov = 0
skipped_mismatch = 0
skipped_incomplete = 0
total = 0

with open(input_path) as f:
    for line in f:
        total += 1
        game = json.loads(line)
        raw_moves = game['moves'].split()
        moves = [strip_annotations(m) for m in raw_moves]
        analysis = game['analysis']

        token_ids = [bos_id]
        has_oov = False
        for m in moves:
            if m not in stoi:
                has_oov = True
                break
            token_ids.append(stoi[m])
        if has_oov:
            skipped_oov += 1
            continue
        token_ids.append(eos_id)

        n_moves = len(moves)
        n_analysis = len(analysis)

        evals = [float('nan')]  # bos
        incomplete = False
        if n_analysis == n_moves or n_analysis == n_moves - 1:
            for a in analysis:
                if 'mate' in a:
                    evals.append(np.sign(a['mate']) * 10000.0)
                elif 'eval' in a:
                    evals.append(float(a['eval']))
                else:
                    incomplete = True
                    break
            if incomplete:
                skipped_incomplete += 1
                continue
            if n_analysis == n_moves - 1:
                evals.append(float('nan'))
        else:
            skipped_mismatch += 1
            continue
        evals.append(float('nan'))  # eos

        assert len(token_ids) == len(evals)

        # normalize evals for weight computation
        norm_evals = np.array(evals, dtype=np.float32)
        mask = ~np.isnan(norm_evals)
        norm_evals[mask] = np.clip(norm_evals[mask], -1500, 1500) / 1500.0

        # weights from eval deltas: weight[t] = 1 + alpha * |eval[t] - eval[t-1]|
        # endgame multiplier: ramp up after move 30 (token index 31 = move 30)
        weights = np.ones(len(evals), dtype=np.float32)
        for t in range(1, len(evals)):
            if mask[t] and mask[t - 1]:
                weights[t] = 1.0 + alpha * abs(norm_evals[t] - norm_evals[t - 1])
            move_num = t - 1
            weights[t] *= 1.0 + max(0, (move_num - 30)) * 0.05

        all_tokens.extend(token_ids)
        all_evals.extend(evals)
        all_weights.extend(weights)

        if total % 500000 == 0:
            print(f"  processed {total} games, kept {total - skipped_oov - skipped_mismatch - skipped_incomplete}")

print(f"\nTotal: {total} games")
print(f"Skipped OOV (rare moves): {skipped_oov} ({skipped_oov/total*100:.2f}%)")
print(f"Skipped mismatch: {skipped_mismatch} ({skipped_mismatch/total*100:.2f}%)")
print(f"Skipped incomplete: {skipped_incomplete} ({skipped_incomplete/total*100:.2f}%)")
kept = total - skipped_oov - skipped_mismatch - skipped_incomplete
print(f"Kept: {kept} games")
print(f"Total tokens: {len(all_tokens)}")

tokens = np.array(all_tokens, dtype=np.uint16)
evals = np.array(all_evals, dtype=np.float32)
weights = np.array(all_weights, dtype=np.float32)

# 90/10 split on game boundaries
bos_positions = np.where(tokens == bos_id)[0]
n_games = len(bos_positions)
n_train = int(n_games * 0.9)

split_idx = bos_positions[n_train]
train_tokens = tokens[:split_idx]
val_tokens = tokens[split_idx:]
train_evals = evals[:split_idx]
val_evals = evals[split_idx:]
train_weights = weights[:split_idx]
val_weights = weights[split_idx:]

print(f"\nTrain: {len(train_tokens)} tokens ({n_train} games)")
print(f"Val: {len(val_tokens)} tokens ({n_games - n_train} games)")

train_tokens.tofile(output_dir / 'train.bin')
val_tokens.tofile(output_dir / 'val.bin')
train_evals.tofile(output_dir / 'train_evals.bin')
val_evals.tofile(output_dir / 'val_evals.bin')
train_weights.tofile(output_dir / 'train_weights.bin')
val_weights.tofile(output_dir / 'val_weights.bin')

out_meta = {
    'vocab_size': vocab_size,
    'context_length': context_length,
    'itos': itos,
    'stoi': stoi,
}
with open(output_dir / 'meta.pkl', 'wb') as f:
    pickle.dump(out_meta, f)

print(f"\nWrote files to {output_dir}")
print(f"  train.bin: {train_tokens.nbytes / 1e6:.1f} MB")
print(f"  val.bin: {val_tokens.nbytes / 1e6:.1f} MB")
print(f"  train_evals.bin: {train_evals.nbytes / 1e6:.1f} MB")
print(f"  val_evals.bin: {val_evals.nbytes / 1e6:.1f} MB")
print(f"  train_weights.bin: {train_weights.nbytes / 1e6:.1f} MB")
print(f"  val_weights.bin: {val_weights.nbytes / 1e6:.1f} MB")

# weight stats
print(f"\nWeight stats (train):")
print(f"  mean: {train_weights.mean():.3f}, max: {train_weights.max():.3f}")
print(f"  fraction > 1.0: {(train_weights > 1.0).sum() / len(train_weights):.3f}")

# spot check
print("\nSpot check first game:")
first_eos = np.where(train_tokens == eos_id)[0][0]
toks = train_tokens[:first_eos + 1]
evs = train_evals[:first_eos + 1]
wts = train_weights[:first_eos + 1]
for i in range(min(10, len(toks))):
    print(f"  {itos[toks[i]]:>10s}  eval={evs[i]:>8.1f}  weight={wts[i]:.3f}")
print(f"  ... ({len(toks)} tokens total)")
