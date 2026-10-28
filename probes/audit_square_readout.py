"""Independent set-based audit of saved direct-square probe predictions."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import chess
import numpy as np


def position_counts(actual, predicted):
    truth = {(int(piece), square) for square, piece in enumerate(actual) if piece}
    guess = {(int(piece), square) for square, piece in enumerate(predicted) if piece}
    matched = len(truth & guess)
    missing, extra = len(truth - guess), len(guess - truth)
    errors = max(missing, extra)
    kings = lambda pieces: {p for p in pieces if p[0] in (6, 12)}
    return {'positions': 1, 'exact_boards': int(truth == guess), 'actual': len(truth),
            'predicted': len(guess), 'matched': matched, 'missing': missing, 'extra': extra,
            'errors': errors, 'errors_0': int(errors == 0), 'errors_1': int(errors == 1),
            'errors_2': int(errors == 2), 'errors_3_plus': int(errors >= 3),
            'both_kings_correct': int(kings(truth) == kings(guess))}


def check(summary, rows):
    total = Counter()
    for row in rows:
        total.update(row)
    assert dict(total) == summary['counts']
    if not rows:
        assert summary['exact_board_rate'] is None
        return
    n = total['positions']
    assert summary['exact_board_rate'] == total['exact_boards'] / n
    assert summary['mean_errors'] == total['errors'] / n
    assert summary['both_kings_correct_rate'] == total['both_kings_correct'] / n
    assert sum(total['errors_' + k] for k in ('0', '1', '2', '3_plus')) == n
    for k in ('0', '1', '2', '3_plus'):
        assert summary['error_distribution'][k] == total['errors_' + k] / n


def audit(root):
    result = json.loads((root / 'square-readout-result.json').read_text())['oracle']
    previous = json.loads((root / 'full-board-readout-result.json').read_text())['oracle']
    assert result['backbone_bitwise_unchanged']
    assert result['frozen_state_hash'] == previous['frozen_state_hash']
    assert result['protocol']['checkpoint_sha256'] == previous['protocol']['checkpoint_sha256']
    assert result['protocol']['version'] == 3 and result['protocol']['steps'] == 3000
    assert not result['protocol']['material_weighting']
    path = root / 'square-oracle-predictions.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result['probe']['predictions_sha256']
    arrays = np.load(path)
    prior = np.load(root / 'full-board-oracle-predictions.npz')
    actual, predicted = arrays['actual'], arrays['predicted']
    assert actual.shape == predicted.shape == (len(arrays['ply']), 64)
    assert np.isin(predicted, np.arange(13)).all()
    assert result['probe']['source_sha256'] == previous['probe']['source_sha256']
    for key in ('game_id', 'ply'):
        np.testing.assert_array_equal(arrays[key], prior[key])
    for features, squares in zip(prior['actual'], actual):
        expected = np.zeros(64, dtype=int)
        for identity, (square, kind) in enumerate(features):
            if square < 64:
                expected[square] = kind + (6 if identity >= 16 else 0)
        np.testing.assert_array_equal(squares, expected)
    rows = [position_counts(a, p) for a, p in zip(actual, predicted)]
    plies = arrays['ply']
    masks = {'initial': plies == 0, 'moves_1_10': (plies > 0) & (plies <= 20),
             'moves_11_30': (plies > 20) & (plies <= 60), 'moves_31_plus': plies > 60}
    def selected(mask):
        return [r for r, keep in zip(rows, mask) if keep]
    check(result['probe']['aggregate'], selected(plies > 0))
    for name, mask in masks.items():
        check(result['probe']['by_stage'][name], selected(mask))
    games = result['probe']['per_game']
    assert [g['game_id'] for g in games] == list(range(200))
    assert result['probe']['games'] == 200 and result['probe']['truncated_games'] == 0
    for game in games:
        mask = arrays['game_id'] == game['game_id']
        np.testing.assert_array_equal(plies[mask], np.arange(mask.sum()))
        check(game['aggregate'], selected(mask & (plies > 0)))
        for name, stage_mask in masks.items():
            check(game['by_stage'][name], selected(mask & stage_mask))
    for name, aggregate in result['probe']['game_average'].items():
        group = [g['aggregate'] if name == 'all_moves' else g['by_stage'][name] for g in games]
        group = [g for g in group if g['counts'].get('positions', 0)]
        assert aggregate['games'] == len(group)
        assert aggregate['positions'] == sum(g['counts']['positions'] for g in group)
        assert aggregate['exact_board_rate'] == float(np.mean([g['exact_board_rate'] for g in group]))
        assert aggregate['mean_errors'] == float(np.mean([g['mean_errors'] for g in group]))
    parts = arrays['loss_parts'][plies > 0]
    assert np.isfinite(parts).all() and (parts >= 0).all()
    assert result['probe']['loss']['balanced_loss'] == float(parts.mean())
    assert result['probe']['loss']['occupied_loss'] == float(parts[:, 0].mean())
    assert result['probe']['loss']['empty_loss'] == float(parts[:, 1].mean())
    timeline = result['trace']['timeline']
    assert len(timeline) == 74
    for position in timeline:
        truth = np.zeros(64, dtype=int)
        for square, piece in chess.Board(position['fen']).piece_map().items():
            truth[square] = piece.piece_type + (0 if piece.color else 6)
        np.testing.assert_array_equal(position['actual'], truth)
        assert position['counts'] == position_counts(position['actual'], position['predicted'])
    check(result['trace']['aggregate'], [p['counts'] for p in timeline if p['ply'] > 0])
    curve = result['learning_curve']
    assert [r['step'] for r in curve] == [0, 1, 500, 1000, 1500, 2000, 2500, 3000]
    assert result['validation'] == curve[-1]
    for row in curve:
        assert row['positions'] == curve[0]['positions']
        assert row['balanced_loss'] == (row['occupied_loss'] + row['empty_loss']) / 2
    output = {'passed': True, 'games': len(games), 'post_move_positions': int((plies > 0).sum()),
              'backbone_matches_previous_frozen_checkpoint': True,
              'initial_validation_loss': curve[0]['balanced_loss'],
              'final_validation_loss': curve[-1]['balanced_loss'],
              'game_average_by_stage': result['probe']['game_average'],
              'trace_before_g5': timeline[71]['counts']}
    (root / 'square-readout-audit.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('data/probes/board-input-20260922'))
    audit(parser.parse_args().root)
