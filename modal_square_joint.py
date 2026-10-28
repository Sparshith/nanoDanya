"""Balanced square-loss continuation, fresh readouts and a fixed chess benchmark."""
import json
from pathlib import Path

import modal

from modal_board_benchmark import image, run_evaluation

image = image.add_local_file('modal_board_benchmark.py', '/root/project/modal_board_benchmark.py')
app = modal.App('nanodanya-square-joint')
volume = modal.Volume.from_name('nanodanya-data')
SOURCE = '/data/experiments/involved-pieces-20260921'
BASE = '/data/experiments/board-input-20260922'
RUN = '/data/experiments/square-joint-20260922'
INITIAL = f'{BASE}/oracle/final.pt'
INITIAL_SHA = 'b07f4d6ac0a7330faad2937954e9dee6f4dca4a0251191cda073196da34828be'
HEAD = f'{BASE}/square-readouts/oracle/head.pt'
MODELS = {arm: f'{RUN}/{arm}/final.pt' for arm in ('control', 'joint')}


def setup():
    import os
    import sys
    os.chdir('/root/project')
    sys.path[:0] = ['/root/project', '/root/project/nanochat']


@app.function(image=image, gpu='H100', cpu=8, memory=32768, max_containers=2,
              volumes={'/data': volume}, timeout=7200)
def train(arm):
    setup()
    import time
    import torch
    from data_prep.prepare_involved_pieces import file_hash
    from training.train_square_joint import train as fit
    volume.reload()
    assert arm in MODELS and file_hash(INITIAL) == INITIAL_SHA
    output = Path(RUN) / arm
    if (output / 'result.json').exists():
        result = json.loads((output / 'result.json').read_text())
        assert result['protocol']['initial_sha256'] == INITIAL_SHA
        assert result['protocol']['initial_head_sha256'] == file_hash(HEAD)
        assert result['protocol']['steps'] == 1000 and result['protocol']['board_weight'] == 0.1
        assert result['protocol']['joint'] == (arm == 'joint')
        assert file_hash(MODELS[arm]) == result['sha256']
        return {'arm': arm, **result}
    torch.set_num_threads(8)
    started = time.monotonic()
    try:
        result = fit(INITIAL, HEAD, f'{SOURCE}/dataset', f'{BASE}/boards', output, arm == 'joint')
        result['seconds'] = time.monotonic() - started
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        return {'arm': arm, **result}
    finally:
        volume.commit()


@app.function(image=image, gpu='H100', cpu=8, memory=32768, max_containers=2,
              volumes={'/data': volume}, timeout=7200)
def readout(arm, digest):
    setup()
    import torch
    from data_prep.prepare_involved_pieces import file_hash
    from model_loading import load_model
    from probes.board_input_readout import FrozenReadout, run_readout
    from probes.square_readout import evaluate_squares
    volume.reload()
    assert file_hash(MODELS[arm]) == digest
    torch.set_num_threads(8)
    output = Path(RUN) / arm / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        result = json.loads((output / 'result.json').read_text())
        assert result['fresh']['protocol']['checkpoint_sha256'] == digest
        return {'arm': arm, **result}
    pgn = (Path(SOURCE) / 'game.pgn').read_text()
    try:
        player, _, stoi, itos = load_model(MODELS[arm], 'cuda')
        head = FrozenReadout(player, classes_per_piece=13, slots=64)
        del player
        checkpoint = torch.load(MODELS[arm], map_location='cpu', weights_only=False)
        head.head.load_state_dict(checkpoint['square_head'])
        del checkpoint
        attached_dir = output / 'attached'
        attached_dir.mkdir(exist_ok=True)
        attached = evaluate_squares(head, f'{SOURCE}/dataset', pgn, stoi, itos, attached_dir, 'cuda')
        del head
        torch.cuda.empty_cache()
        fresh = run_readout(MODELS[arm], f'{SOURCE}/dataset', f'{BASE}/boards', pgn,
                            output / 'fresh', square_board=True)
        result = {'attached': attached, 'fresh': fresh}
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        return {'arm': arm, **result}
    finally:
        volume.commit()


@app.function(image=image, gpu='H100', cpu=4, memory=32768, max_containers=4,
              volumes={'/data': volume}, timeout=7200)
def benchmark(name, command, flags, hashes):
    return run_evaluation(name, command, flags, hashes, f'{RUN}/benchmarks')


@app.function(image=image, gpu='H100', cpu=4, memory=16384,
              volumes={'/data': volume}, timeout=1200)
def trace(hashes):
    setup()
    import chess
    import torch
    from chess_inference import choose_move_from_logits, legal_token_ids, token_for_id
    from chess_token_utils import strip_san
    from data_prep.prepare_involved_pieces import file_hash
    from model_loading import load_model
    from probes.inspect_game import game_position
    volume.reload()
    torch.set_num_threads(4)
    pgn = (Path(SOURCE) / 'game.pgn').read_text()
    result = {}
    for arm, path in MODELS.items():
        assert file_hash(path) == hashes[arm]
        player, _, stoi, itos = load_model(path, 'cuda')
        position = game_position(pgn, stoi, 71)
        board = chess.Board(position['fen'][0])
        with torch.no_grad():
            logits = player(torch.tensor(position['prefixes'], device='cuda'))[0, -1].float()
        legal = legal_token_ids(stoi, board, allow_eos=False)
        probs = (logits[legal] / 0.2).softmax(-1)
        _, greedy = choose_move_from_logits(logits, board, stoi, itos, temperature=0,
                                             allow_eos=False, legal_mask=True)
        result[arm] = {'fen': board.fen(), 'greedy': greedy, 'temperature': 0.2,
                       'g5_probability': sum(float(probs[i]) for i, token in enumerate(legal)
                                             if strip_san(token_for_id(itos, token)) == 'g5')}
        del player
    return result


@app.function(image=image, cpu=0.25, memory=4096, timeout=21600)
def pipeline():
    training = {r['arm']: r for r in train.map(list(MODELS), order_outputs=False)}
    hashes = {arm: r['sha256'] for arm, r in training.items()}
    readers = {arm: readout.spawn(arm, digest) for arm, digest in hashes.items()}
    jobs = []
    for offset in range(0, 2000, 50):
        jobs.append((f'joint-vs-control-{offset:04d}', 'games', {
            'game_mode': 'h2h', 'model': MODELS['joint'], 'opponent_model': MODELS['control'],
            'games': 50, 'game_offset': offset, 'seed': 20280922 + offset, 'batch_size': 50,
            'temperature': 0.2, 'max_plies': 200, 'kv_cache': True,
        }, {'model': hashes['joint'], 'opponent_model': hashes['control']}))
    for arm in MODELS:
        jobs.append((f'{arm}-move-quality', 'move-quality', {
            'model': MODELS[arm], 'data_dir': f'{SOURCE}/dataset', 'split': 'probe',
            'max_games': 200, 'max_ply': 140, 'batch_size': 64, 'max_positions': 1024,
            'stockfish': '/usr/games/stockfish', 'stockfish_depth': 10, 'write_positions': True,
        }, {'model': hashes[arm]}))
    benchmarks = []
    for row in benchmark.starmap(jobs, order_outputs=False):
        benchmarks.append(row)
        print(json.dumps({'finished': row['name'], 'completed_jobs': len(benchmarks), 'total_jobs': len(jobs)}), flush=True)
    return {'training': training, 'readouts': {arm: handle.get() for arm, handle in readers.items()},
            'benchmarks': benchmarks, 'trace': trace.remote(hashes),
            'protocol': {'board_weight': 0.1, 'steps': 1000, 'games': 2000, 'temperature': 0.2,
                         'seed_base': 20280922, 'stopping_rule': 'fixed sample, no interim selection'}}


@app.local_entrypoint()
def main():
    result = pipeline.remote()
    output = Path('benchmark/results/square-joint-20260922')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'training': {arm: r['metrics'][-1] for arm, r in result['training'].items()},
                      'readouts': {arm: r['fresh']['probe']['game_average'] for arm, r in result['readouts'].items()},
                      'trace': result['trace']}, indent=2))
