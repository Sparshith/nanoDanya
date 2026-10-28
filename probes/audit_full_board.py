"""Recompute full-board metrics from exported predictions using independent histograms."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def histogram(features):
    square, kind = features[..., 0].astype(int), features[..., 1].astype(int)
    assert ((square >= 0) & (square <= 64)).all()
    assert np.array_equal(square == 64, kind == 0)
    assert ((kind >= 0) & (kind <= 6)).all()
    color = np.arange(32) >= 16
    code = (color * 6 + kind - 1) * 64 + square
    alive = square < 64
    rows = np.broadcast_to(np.arange(len(features))[:, None], square.shape)
    result = np.zeros((len(features), 768), dtype=np.int32)
    np.add.at(result, (rows[alive], code[alive]), 1)
    return result.reshape(-1, 2, 6, 64)


def audit(root):
    results = json.loads((root / 'full-board-readout-result.json').read_text())
    previous = json.loads((root / 'readout-result.json').read_text())
    assert set(results) == {'champion', 'control', 'oracle'}
    reference = results['champion']
    protocol = lambda r: {k: v for k, v in r['protocol'].items() if k not in ('checkpoint', 'checkpoint_sha256')}
    reference_arrays, rows_by_arm, critical = None, {}, {}
    for arm, result in results.items():
        assert protocol(result) == protocol(reference)
        assert result['metric_protocol'] == reference['metric_protocol']
        assert result['backbone_bitwise_unchanged']
        assert result['frozen_state_hash'] == previous[arm]['frozen_state_hash']
        assert result['protocol']['checkpoint_sha256'] == previous[arm]['protocol']['checkpoint_sha256']
        assert result['protocol']['steps'] == result['validation']['step'] == 3000
        path = root / f'full-board-{arm}-predictions.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == result['probe']['predictions_sha256']
        arrays = np.load(path)
        assert arrays['actual'].shape == arrays['predicted'].shape
        if reference_arrays is None:
            reference_arrays = {key: arrays[key] for key in ('actual', 'game_id', 'ply')}
        for key, values in reference_arrays.items():
            np.testing.assert_array_equal(arrays[key], values)
        assert result['probe']['source_sha256'] == previous[arm]['probe']['source_sha256']
        assert result['probe']['games'] == 200 and result['probe']['truncated_games'] == 0
        truth, predicted = histogram(arrays['actual']), histogram(arrays['predicted'])
        assert (truth.sum((1, 2)) <= 1).all()
        matched = np.minimum(truth, predicted)
        columns = {'positions': np.ones(len(truth), dtype=int)}
        for name, values in (('actual', truth), ('predicted', predicted), ('matched', matched)):
            pieces = values.sum((1, 3))
            columns[name] = pieces.sum(1)
            columns[f'{name}_material'] = pieces @ np.array([1, 3, 3, 5, 9, 0])
            for i, label in enumerate(('pawn', 'knight', 'bishop', 'rook', 'queen', 'king')):
                columns[f'{name}_{label}'] = pieces[:, i]
        columns['missing'] = columns['actual'] - columns['matched']
        columns['extra'] = columns['predicted'] - columns['matched']
        columns['errors'] = np.maximum(columns['missing'], columns['extra'])
        columns['exact_boards'] = (truth == predicted).all((1, 2, 3)).astype(int)
        columns['overlap_positions'] = (predicted.sum((1, 2)) > 1).any(1).astype(int)
        columns['both_kings_correct'] = (truth[:, :, 5] == predicted[:, :, 5]).all((1, 2)).astype(int)
        for k in (0, 1, 2):
            columns[f'errors_{k}'] = (columns['errors'] == k).astype(int)
        columns['errors_3_plus'] = (columns['errors'] >= 3).astype(int)
        p = arrays['ply']
        bands = {'initial': p == 0, 'moves_1_10': (p > 0) & (p <= 20),
                 'moves_11_30': (p > 20) & (p <= 60), 'moves_31_plus': p > 60}

        def check(summary, mask):
            expected = {key: int(value[mask].sum()) for key, value in columns.items()}
            if not mask.any():
                assert not summary['counts']
                return
            assert expected == summary['counts']
            ratio = lambda a, b: a / b if b else None
            ratios = {'exact_board_rate': (expected['exact_boards'], expected['positions']),
                      'mean_errors': (expected['errors'], expected['positions']),
                      'piece_f1': (2 * expected['matched'], expected['actual'] + expected['predicted']),
                      'material_f1': (2 * expected['matched_material'], expected['actual_material'] + expected['predicted_material']),
                      'both_kings_correct_rate': (expected['both_kings_correct'], expected['positions'])}
            for key, args in ratios.items():
                assert summary[key] == ratio(*args)
            assert sum(expected[f'errors_{k}'] for k in ('0', '1', '2', '3_plus')) == expected['positions']

        check(result['probe']['aggregate'], p > 0)
        for band, mask in bands.items():
            check(result['probe']['by_stage'][band], mask)
        rows = result['probe']['per_game']
        assert [r['game_id'] for r in rows] == list(range(200))
        for row in rows:
            game_mask = arrays['game_id'] == row['game_id']
            assert np.array_equal(p[game_mask], np.arange(game_mask.sum()))
            check(row['aggregate'], game_mask & (p > 0))
            for band, mask in bands.items():
                check(row['by_stage'][band], game_mask & mask)
        rows_by_arm[arm] = rows
        critical[arm] = result['trace']['timeline'][71]
    rng = np.random.default_rng(42)
    indices = rng.integers(200, size=(20000, 200))
    comparisons = {}
    for other in ('champion', 'control'):
        comparisons[other] = {}
        for metric, num, den, scale in (('exact_board_rate', 'exact_boards', 'positions', 100),
                                        ('mean_errors', 'errors', 'positions', 1),
                                        ('material_f1', 'matched_material', None, 100)):
            samples, observed = {}, {}
            for arm in ('oracle', other):
                counts = [r['aggregate']['counts'] for r in rows_by_arm[arm]]
                numerators = np.array([r[num] for r in counts]) * (2 if den is None else 1)
                denominators = np.array([r[den] if den else r['actual_material'] + r['predicted_material'] for r in counts])
                samples[arm] = numerators[indices].sum(1) / denominators[indices].sum(1)
                observed[arm] = numerators.sum() / denominators.sum()
            comparisons[other][metric] = {'delta': scale * (observed['oracle'] - observed[other]),
                                           'paired_game_bootstrap_95ci': (scale * np.quantile(samples['oracle'] - samples[other], [.025, .975])).tolist(),
                                           'unit': 'percentage points' if scale == 100 else 'errors per position'}
    audit_result = {'passed': True, 'bootstrap_seed': 42, 'resamples': 20000,
                    'new_minus_comparator': comparisons,
                    'trace_before_g5': {arm: row['counts'] for arm, row in critical.items()}}
    (root / 'full-board-readout-audit.json').write_text(json.dumps(audit_result, indent=2) + '\n')
    print(json.dumps(audit_result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('data/probes/board-input-20260922'))
    audit(parser.parse_args().root)
