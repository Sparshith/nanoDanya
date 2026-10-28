"""Direct-square linear probe with per-position occupied/empty balanced loss."""
import io
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
from torch.nn import functional as F

from chess_token_utils import resolve_token_id
from data_prep.prepare_involved_pieces import file_hash
from probes.build_positions import load_games
from probes.full_board_readout import stage

BANDS = ('initial', 'moves_1_10', 'moves_11_30', 'moves_31_plus')
METRIC_PROTOCOL = {
    'primary': 'Exact colored piece placement by move band, averaging within each game then across games',
    'secondary': 'Pooled position accuracy and 0/1/2/3+ piece-error distribution',
    'errors': 'max(missing, extra) after exact color/type/square matches; relocation counts once',
    'classes': '0 empty; 1..6 white P N B R Q K; 7..12 black P N B R Q K',
    'decoding': 'independent argmax per square; no truth filtering or chess-legality correction',
    'move_bands': {'moves_1_10': [1, 20], 'moves_11_30': [21, 60], 'moves_31_plus': [61, None]},
    'initial_positions': 'reported separately, excluded from primary scores',
    'material_weighting': False,
    'scope': 'piece placement only, not turn, castling rights, en passant or legality',
}


def square_targets(boards):
    locations, kinds = boards.unbind(-1)
    colors = (torch.arange(32, device=boards.device) >= 16).long() * 6
    values = (kinds + colors) * (locations < 64)
    squares = torch.zeros((*boards.shape[:-2], 65), dtype=torch.long, device=boards.device)
    squares.scatter_add_(-1, locations, values)
    return squares[..., :64]


def square_loss_parts(logits, tokens, boards, stoi):
    truth = square_targets(boards)
    occupied = truth != 0
    valid = ((tokens == stoi['<bos>']).cumsum(1) > 0) & (tokens != stoi['<eos>'])
    ce = F.cross_entropy(logits.flatten(0, 2), truth.flatten(), reduction='none').view_as(truth)
    occupied_loss = (ce * occupied).sum(-1) / occupied.sum(-1).clamp_min(1)
    empty_loss = (ce * ~occupied).sum(-1) / (~occupied).sum(-1).clamp_min(1)
    return occupied_loss, empty_loss, valid


def square_loss(logits, tokens, boards, stoi):
    occupied, empty, valid = square_loss_parts(logits, tokens, boards, stoi)
    return ((occupied + empty) * 0.5 * valid).sum() / valid.sum().clamp_min(1)


def score_positions(actual, predicted):
    matched = (actual == predicted) & (actual != 0)
    actual_count = (actual != 0).sum(-1)
    predicted_count = (predicted != 0).sum(-1)
    correct = matched.sum(-1)
    missing, extra = actual_count - correct, predicted_count - correct
    errors = np.maximum(missing, extra)
    true_kings = np.where((actual == 6) | (actual == 12), actual, 0)
    pred_kings = np.where((predicted == 6) | (predicted == 12), predicted, 0)
    columns = {'positions': np.ones(len(actual), dtype=int), 'exact_boards': (actual == predicted).all(-1),
               'actual': actual_count, 'predicted': predicted_count, 'matched': correct,
               'missing': missing, 'extra': extra, 'errors': errors,
               'errors_0': errors == 0, 'errors_1': errors == 1, 'errors_2': errors == 2,
               'errors_3_plus': errors >= 3, 'both_kings_correct': (true_kings == pred_kings).all(-1)}
    return [{key: int(value[i]) for key, value in columns.items()} for i in range(len(actual))]


def summarize(rows):
    counts = Counter()
    for row in rows:
        counts.update(row)
    ratio = lambda a, b: a / b if b else None
    n = counts['positions']
    return {'counts': dict(counts), 'exact_board_rate': ratio(counts['exact_boards'], n),
            'mean_errors': ratio(counts['errors'], n),
            'error_distribution': {k: ratio(counts['errors_' + k], n) for k in ('0', '1', '2', '3_plus')},
            'precision': ratio(counts['matched'], counts['predicted']),
            'recall': ratio(counts['matched'], counts['actual']),
            'both_kings_correct_rate': ratio(counts['both_kings_correct'], n)}


def report(records):
    return {'aggregate': summarize(r['counts'] for r in records if r['ply'] > 0),
            'by_stage': {band: summarize(r['counts'] for r in records if stage(r['ply']) == band)
                         for band in BANDS}}


def game_average(per_game):
    result = {}
    rng = np.random.default_rng(42)
    for band in ('all_moves', *BANDS):
        groups = [g['aggregate'] if band == 'all_moves' else g['by_stage'][band] for g in per_game]
        groups = [g for g in groups if g['counts'].get('positions', 0)]
        if not groups:
            result[band] = {'games': 0, 'positions': 0, 'exact_board_rate': None, 'mean_errors': None,
                            'exact_board_95ci': None, 'error_distribution': None}
            continue
        exact = np.array([g['exact_board_rate'] for g in groups])
        draws = rng.integers(len(groups), size=(10000, len(groups)))
        result[band] = {'games': len(groups), 'positions': sum(g['counts']['positions'] for g in groups),
                        'exact_board_rate': float(exact.mean()),
                        'exact_board_95ci': np.quantile(exact[draws].mean(1), [.025, .975]).tolist(),
                        'mean_errors': float(np.mean([g['mean_errors'] for g in groups])),
                        'error_distribution': {k: float(np.mean([g['error_distribution'][k] for g in groups]))
                                               for k in ('0', '1', '2', '3_plus')}}
    return result


@torch.no_grad()
def evaluate_squares(model, data_dir, pgn, stoi, itos, output, device):
    from probes.board_input_readout import sequence_batch

    games = load_games(Path(data_dir) / 'probe.bin', stoi['<bos>'], stoi['<eos>'], 200)
    truncated = sum(len(game) > model.config.sequence_len for game in games)
    games = [game[:model.config.sequence_len] for game in games]
    actuals, guesses, game_ids, plies, records, per_game, losses = [], [], [], [], [], [], []
    for offset in range(0, len(games), 8):
        batch = games[offset:offset + 8]
        x, boards = sequence_batch(batch, itos, stoi['<eos>'], device)
        logits = model(x, boards)
        predictions = logits.argmax(-1).cpu().numpy().astype(np.uint8)
        truth = square_targets(boards).cpu().numpy().astype(np.uint8)
        occupied, empty, _ = square_loss_parts(logits, x, boards, stoi)
        occupied, empty = occupied.cpu().numpy(), empty.cpu().numpy()
        for i, game in enumerate(batch):
            actual, predicted = truth[i, :len(game)], predictions[i, :len(game)]
            counts = score_positions(actual, predicted)
            rows = [{'ply': ply, 'counts': c} for ply, c in enumerate(counts)]
            records.extend(rows)
            per_game.append({'game_id': offset + i, **report(rows)})
            actuals.extend(actual)
            guesses.extend(predicted)
            game_ids.extend([offset + i] * len(game))
            plies.extend(range(len(game)))
            losses.extend(zip(occupied[i, :len(game)].tolist(), empty[i, :len(game)].tolist()))
    np.savez_compressed(Path(output) / 'probe-predictions.npz', actual=np.array(actuals),
                        predicted=np.array(guesses), game_id=np.array(game_ids), ply=np.array(plies),
                        loss_parts=np.array(losses))
    scored_losses = np.array(losses)[np.array(plies) > 0]
    probe = {**report(records), 'game_average': game_average(per_game), 'per_game': per_game,
             'games': len(games), 'truncated_games': truncated,
             'loss': {'balanced_loss': float(scored_losses.mean()),
                      'occupied_loss': float(scored_losses[:, 0].mean()),
                      'empty_loss': float(scored_losses[:, 1].mean())},
             'source_sha256': file_hash(Path(data_dir) / 'probe.bin'),
             'predictions_sha256': file_hash(Path(output) / 'probe-predictions.npz')}

    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    sequence, metadata = [stoi['<bos>']], [{'ply': 0, 'fen': board.fen(), 'last_move': None}]
    for ply, move in enumerate(game.mainline_moves(), 1):
        san = board.san(move)
        token = resolve_token_id(stoi, san)
        if token is None:
            raise ValueError(f'Trace move missing from tokenizer: {san}')
        sequence.append(token)
        board.push(move)
        metadata.append({'ply': ply, 'fen': board.fen(), 'last_move': san})
    assert len(sequence) <= model.config.sequence_len
    x, boards = sequence_batch([sequence], itos, stoi['<eos>'], device)
    predicted = model(x, boards)[0].argmax(-1).cpu().numpy().astype(np.uint8)
    truth = square_targets(boards)[0].cpu().numpy().astype(np.uint8)
    counts = score_positions(truth, predicted)
    timeline = [position | {'counts': c, 'actual': a.tolist(), 'predicted': p.tolist(),
                            'next_move': metadata[i + 1]['last_move'] if i + 1 < len(metadata) else None}
                for i, (position, c, a, p) in enumerate(zip(metadata, counts, truth, predicted))]
    return {'metric_protocol': METRIC_PROTOCOL, 'probe': probe,
            'trace': {**report(timeline), 'timeline': timeline}}
