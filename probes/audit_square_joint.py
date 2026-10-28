"""Audit matched continuation, reconstruction predictions, and completed chess games."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import chess
import numpy as np

from probes.audit_square_readout import check, position_counts


def audit_readout(result, path, base):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result['probe']['predictions_sha256']
    data = np.load(path)
    for key in ('actual', 'game_id', 'ply'):
        np.testing.assert_array_equal(data[key], base[key])
    assert np.isin(data['predicted'], np.arange(13)).all()
    rows = [position_counts(a, p) for a, p in zip(data['actual'], data['predicted'])]
    plies = data['ply']
    bands = {'initial': plies == 0, 'moves_1_10': (plies > 0) & (plies <= 20),
             'moves_11_30': (plies > 20) & (plies <= 60), 'moves_31_plus': plies > 60}
    choose = lambda mask: [r for r, keep in zip(rows, mask) if keep]
    check(result['probe']['aggregate'], choose(plies > 0))
    for band, mask in bands.items():
        check(result['probe']['by_stage'][band], choose(mask))
    assert result['probe']['games'] == 200 and result['probe']['truncated_games'] == 0
    games = result['probe']['per_game']
    assert [g['game_id'] for g in games] == list(range(200))
    for game in games:
        mask = data['game_id'] == game['game_id']
        check(game['aggregate'], choose(mask & (plies > 0)))
        for band, bmask in bands.items():
            check(game['by_stage'][band], choose(mask & bmask))
    for band, score in result['probe']['game_average'].items():
        group = [g['aggregate'] if band == 'all_moves' else g['by_stage'][band] for g in games]
        group = [g for g in group if g['counts'].get('positions', 0)]
        assert score['games'] == len(group)
        assert score['exact_board_rate'] == float(np.mean([g['exact_board_rate'] for g in group]))
    losses = data['loss_parts'][plies > 0]
    assert result['probe']['loss']['balanced_loss'] == float(losses.mean())
    assert len(result['trace']['timeline']) == 74
    for position in result['trace']['timeline']:
        board = chess.Board(position['fen'])
        truth = np.zeros(64, dtype=int)
        for square, piece in board.piece_map().items():
            truth[square] = piece.piece_type + (0 if piece.color else 6)
        np.testing.assert_array_equal(position['actual'], truth)
        assert position['counts'] == position_counts(position['actual'], position['predicted'])
    return result['probe']['game_average']


def audit(root):
    result = json.loads((root / 'result.json').read_text())
    assert result['protocol'] == {'board_weight': 0.1, 'steps': 1000, 'games': 2000,
                                 'temperature': 0.2, 'seed_base': 20280922,
                                 'stopping_rule': 'fixed sample, no interim selection'}
    training = result['training']
    assert set(training) == {'control', 'joint'}
    c, j = (training[a]['protocol'] for a in ('control', 'joint'))
    assert {k: v for k, v in c.items() if k != 'joint'} == {k: v for k, v in j.items() if k != 'joint'}
    assert not c['joint'] and j['joint']
    assert c['steps'] == 1000 and c['board_weight'] == .1
    assert training['control']['baseline'] == training['joint']['baseline']
    base_probe = json.loads(Path('data/probes/board-input-20260922/square-readout-result.json').read_text())['oracle']
    assert c['initial_sha256'] == base_probe['protocol']['checkpoint_sha256']
    assert c['initial_backbone_state_hash'] == base_probe['frozen_state_hash']
    for arm, tr in training.items():
        assert [m['step'] for m in tr['metrics']] == [1, *range(100, 1001, 100)]
        assert tr['final_backbone_state_hash'] != tr['protocol']['initial_backbone_state_hash']
        for row in tr['metrics']:
            assert np.isclose(row['train_total_loss'], row['train_move_loss'] + .1 * row['train_board_loss'])
    base = np.load('data/probes/board-input-20260922/square-oracle-predictions.npz')
    representation = {}
    fresh_protocols = []
    for arm, readout in result['readouts'].items():
        representation[arm] = {}
        for kind in ('attached', 'fresh'):
            r = readout[kind]
            representation[arm][kind] = audit_readout(r, root / f'{arm}-{kind}-predictions.npz', base)
        fresh = readout['fresh']
        assert fresh['backbone_bitwise_unchanged']
        assert fresh['frozen_state_hash'] == training[arm]['final_backbone_state_hash']
        assert fresh['protocol']['checkpoint_sha256'] == training[arm]['sha256']
        assert fresh['protocol']['steps'] == fresh['validation']['step'] == 3000
        fresh_protocols.append({k: v for k, v in fresh['protocol'].items() if k not in ('checkpoint', 'checkpoint_sha256')})
    assert fresh_protocols[0] == fresh_protocols[1]
    rng = np.random.default_rng(42)
    comparisons = {}
    for band in ('moves_1_10', 'moves_11_30', 'moves_31_plus'):
        paired = []
        for control, joint in zip(result['readouts']['control']['fresh']['probe']['per_game'],
                                  result['readouts']['joint']['fresh']['probe']['per_game']):
            ca, ja = control['by_stage'][band], joint['by_stage'][band]
            if ca['counts'].get('positions', 0):
                paired.append(ja['exact_board_rate'] - ca['exact_board_rate'])
        delta = np.array(paired)
        draws = rng.integers(len(delta), size=(20000, len(delta)))
        comparisons[band] = {'games': len(delta), 'joint_minus_control_pp': float(delta.mean() * 100),
                              'paired_game_95ci_pp': (np.quantile(delta[draws].mean(1), [.025, .975]) * 100).tolist()}
    jobs = [b for b in result['benchmarks'] if b['request']['subcommand'] == 'games']
    assert len(jobs) == len({b['name'] for b in jobs}) == 40
    hashes = {'model': training['joint']['sha256'], 'opponent_model': training['control']['sha256']}
    games = []
    for job in jobs:
        assert job['request']['model_hashes'] == hashes
        flags = job['request']['flags']
        offset = flags['game_offset']
        assert offset in range(0, 2000, 50) and flags['seed'] == 20280922 + offset
        config = next(r['config'] for r in job['records'] if r['type'] == 'run')
        for key, value in {'games': 50, 'batch_size': 50, 'temperature': .2, 'max_plies': 200,
                           'kv_cache': True, 'legal_mask': True, 'allow_eos': False, 'prompt': ''}.items():
            assert config[key] == value
        shard = [r for r in job['records'] if r['type'] == 'game']
        assert sorted(g['game_num'] for g in shard) == list(range(offset, offset + 50))
        for game in shard:
            assert game['model'] == training['joint']['checkpoint']
            assert game['opponent_model'] == training['control']['checkpoint']
            assert game['model_color'] == ('white' if game['game_num'] % 2 == 0 else 'black')
            assert game['num_plies'] == len(game['moves'])
            board = chess.Board()
            for san in game['moves']:
                board.push_san(san)
            assert game['result'] == board.result()
            if board.is_game_over():
                assert game['termination'] == board.outcome().termination.name
            else:
                assert game['termination'] == 'max_plies' and game['num_plies'] == 200
            outcome = 'draw'
            if board.result() in ('1-0', '0-1'):
                winner = 'white' if board.result() == '1-0' else 'black'
                outcome = 'win' if winner == game['model_color'] else 'loss'
            assert game['outcome'] == outcome
        summary = next(r for r in job['records'] if r['type'] == 'games_summary')
        assert summary['outcomes'] == dict(Counter(g['outcome'] for g in shard))
        games.extend(shard)
    assert len(games) == len({g['game_id'] for g in games}) == 2000
    assert sorted(g['game_num'] for g in games) == list(range(2000))
    rng = np.random.default_rng(20260922)
    bootstrap = np.zeros(20000)
    colors = {}
    for color in ('white', 'black'):
        outcomes = Counter(g['outcome'] for g in games if g['model_color'] == color)
        assert sum(outcomes.values()) == 1000
        draws = rng.multinomial(1000, np.array([outcomes[k] for k in ('win', 'draw', 'loss')]) / 1000, size=20000)
        bootstrap += draws[:, 0] + .5 * draws[:, 1]
        colors[color] = dict(outcomes) | {'score': (outcomes['win'] + .5 * outcomes['draw']) / 1000}
    outcomes = Counter(g['outcome'] for g in games)
    terminations = Counter(g['termination'] for g in games)
    score = (outcomes['win'] + .5 * outcomes['draw']) / 2000
    quality = {}
    qjobs = [b for b in result['benchmarks'] if b['request']['subcommand'] == 'move-quality']
    assert len(qjobs) == 2 and qjobs[0]['engine_sha256'] == qjobs[1]['engine_sha256']
    samples = {}
    for job in qjobs:
        arm = job['name'].split('-')[0]
        assert job['request']['model_hashes']['model'] == training[arm]['sha256']
        rows = [r for r in job['records'] if r['type'] == 'move_quality_position']
        summary = next(r for r in job['records'] if r['type'] == 'move_quality_summary')
        assert len(rows) == summary['scored_positions'] == 1024 and summary['failures'] == 0
        losses = []
        for row in rows:
            board = chess.Board(row['fen'])
            assert board.parse_san(row['played']) in board.legal_moves
            loss = max(0, row['before_cp'] - row['after_cp'])
            assert row['cp_loss'] == loss
            losses.append(loss)
        assert summary['median_cp_loss'] == float(np.median(losses))
        assert np.isclose(summary['avg_cp_loss'], np.mean(losses))
        assert summary['blunder_rate_cp_300'] == sum(l >= 300 for l in losses) / 1024
        samples[arm] = [(r['fen'], r['ply']) for r in rows]
        quality[arm] = summary
    assert samples['control'] == samples['joint']
    output = {'passed': True, 'representation': representation, 'fresh_probe_differences': comparisons,
              'chess': {'games': 2000, 'outcomes': dict(outcomes), 'score': score,
                        'score_95ci': np.quantile(bootstrap / 2000, [.025, .975]).tolist(),
                        'by_color': colors, 'terminations': dict(terminations),
                        'score_if_all_caps_lost_or_won': [score - terminations['max_plies'] / 4000,
                                                        score + terminations['max_plies'] / 4000],
                        'unique_move_sequences': len({tuple(g['moves']) for g in games})},
              'move_quality': quality, 'trace': result['trace']}
    (root / 'audit.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('benchmark/results/square-joint-20260922'))
    audit(parser.parse_args().root)
