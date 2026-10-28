"""Full piece-placement scoring, including captures, promotions and extra pieces."""
import io
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
from torch.nn import functional as F

from chess_token_utils import resolve_token_id
from data_prep.prepare_involved_pieces import INITIAL_SQUARES, file_hash
from probes.build_positions import load_games

MATERIAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
PIECES = ('pawn', 'knight', 'bishop', 'rook', 'queen', 'king')
METRIC_PROTOCOL = {
    'primary': 'Exact colored piece placement, including absence and piece kind; identities interchangeable',
    'errors': 'max(missing pieces, extra pieces); a misplaced piece counts once',
    'material_weights': MATERIAL,
    'material_score': '2 * matched material / (actual material + predicted material); kings scored separately',
    'stage_bands': {'moves_1_10': [1, 20], 'moves_11_30': [21, 60], 'moves_31_plus': [61, None]},
    'initial_positions': 'reported separately, excluded from primary aggregate',
    'decoding': 'independent argmax location and kind; no collision correction or truth-based filtering',
    'scope': 'piece placement only, not side to move, castling rights or en passant rights',
}


def full_loss(logits, tokens, boards, stoi):
    valid = ((tokens == stoi['<bos>']).cumsum(1) > 0) & (tokens != stoi['<eos>'])
    locations = boards[..., 0]
    alive = valid.unsqueeze(-1) & (locations < 64)
    location_loss = F.cross_entropy(logits[..., :65].flatten(0, 2), locations.flatten(), reduction='none')
    kind_loss = F.cross_entropy(logits[..., 65:].flatten(0, 2),
                                (boards[..., 1] - 1).clamp_min(0).flatten(), reduction='none')
    return ((location_loss.view_as(locations) * valid.unsqueeze(-1)).sum() / (valid.sum() * 32).clamp_min(1)
            + (kind_loss.view_as(locations) * alive).sum() / alive.sum().clamp_min(1))


def decode(logits):
    locations = logits[..., :65].argmax(-1)
    kinds = (logits[..., 65:].argmax(-1) + 1).masked_fill(locations == 64, 0)
    return torch.stack((locations, kinds), -1)


def placement(features):
    return Counter((identity < 16, int(kind), int(square))
                   for identity, (square, kind) in enumerate(features) if square < 64)


def score_position(actual, predicted):
    truth, guess = placement(actual), placement(predicted)
    matched = truth & guess
    missing, extra = (truth - matched).total(), (guess - matched).total()
    errors = max(missing, extra)
    overlaps = Counter(square for (_, _, square), count in guess.items() for _ in range(count))
    kings = lambda pieces: Counter({key: count for key, count in pieces.items() if key[1] == chess.KING})
    row = {'positions': 1, 'exact_boards': int(truth == guess), 'missing': missing, 'extra': extra,
           'errors': errors, 'errors_0': int(errors == 0), 'errors_1': int(errors == 1),
           'errors_2': int(errors == 2), 'errors_3_plus': int(errors >= 3),
           'overlap_positions': int(any(n > 1 for n in overlaps.values())),
           'both_kings_correct': int(kings(truth) == kings(guess))}
    for name, pieces in (('actual', truth), ('predicted', guess), ('matched', matched)):
        row[name] = pieces.total()
        row[f'{name}_material'] = sum(MATERIAL[kind] * count for (_, kind, _), count in pieces.items())
        for kind, label in enumerate(PIECES, 1):
            row[f'{name}_{label}'] = sum(count for (_, k, _), count in pieces.items() if k == kind)
    return row


def summarize(rows):
    total = Counter()
    for row in rows:
        total.update(row)
    n = total['positions']
    ratio = lambda a, b: a / b if b else None
    return {'counts': dict(total), 'exact_board_rate': ratio(total['exact_boards'], n),
            'mean_errors': ratio(total['errors'], n),
            'error_distribution': {key: ratio(total[f'errors_{key}'], n) for key in ('0', '1', '2', '3_plus')},
            'precision': ratio(total['matched'], total['predicted']),
            'recall': ratio(total['matched'], total['actual']),
            'piece_f1': ratio(2 * total['matched'], total['predicted'] + total['actual']),
            'material_f1': ratio(2 * total['matched_material'], total['predicted_material'] + total['actual_material']),
            'both_kings_correct_rate': ratio(total['both_kings_correct'], n),
            'overlap_position_rate': ratio(total['overlap_positions'], n),
            'by_piece': {label: {'precision': ratio(total[f'matched_{label}'], total[f'predicted_{label}']),
                                  'recall': ratio(total[f'matched_{label}'], total[f'actual_{label}'])}
                         for label in PIECES}}


def stage(ply):
    if ply == 0:
        return 'initial'
    if ply <= 20:
        return 'moves_1_10'
    return 'moves_11_30' if ply <= 60 else 'moves_31_plus'


def report(records):
    return {'aggregate': summarize(r['counts'] for r in records if r['ply'] > 0),
            'by_stage': {band: summarize(r['counts'] for r in records if stage(r['ply']) == band)
                         for band in ('initial', 'moves_1_10', 'moves_11_30', 'moves_31_plus')}}


@torch.no_grad()
def evaluate_full(model, data_dir, pgn, stoi, itos, output, device):
    from probes.board_input_readout import sequence_batch

    games = load_games(Path(data_dir) / 'probe.bin', stoi['<bos>'], stoi['<eos>'], 200)
    truncated = sum(len(game) > model.config.sequence_len for game in games)
    games = [game[:model.config.sequence_len] for game in games]
    actuals, predictions, game_ids, plies, records, per_game = [], [], [], [], [], []
    for offset in range(0, len(games), 8):
        batch = games[offset:offset + 8]
        x, boards = sequence_batch(batch, itos, stoi['<eos>'], device)
        predicted = decode(model(x, boards)).cpu().numpy().astype(np.uint8)
        boards = boards.cpu().numpy().astype(np.uint8)
        for i, game in enumerate(batch):
            actual, guessed = boards[i, :len(game)], predicted[i, :len(game)]
            rows = [{'ply': ply, 'counts': score_position(a, p)} for ply, (a, p) in enumerate(zip(actual, guessed))]
            records.extend(rows)
            per_game.append({'game_id': offset + i, **report(rows)})
            actuals.extend(actual)
            predictions.extend(guessed)
            game_ids.extend([offset + i] * len(game))
            plies.extend(range(len(game)))
    np.savez_compressed(Path(output) / 'probe-predictions.npz', actual=np.array(actuals),
                        predicted=np.array(predictions), game_id=np.array(game_ids), ply=np.array(plies))
    probe = {**report(records), 'per_game': per_game, 'games': len(games), 'truncated_games': truncated,
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
    guessed = decode(model(x, boards))[0].cpu().tolist()
    actuals = boards[0].cpu().tolist()
    timeline = []
    for i, (position, actual, predicted) in enumerate(zip(metadata, actuals, guessed)):
        pieces = []
        for identity, ((square, kind), (pred_square, pred_kind)) in enumerate(zip(actual, predicted)):
            pieces.append({'identity': chess.square_name(INITIAL_SQUARES[identity]),
                           'actual': chess.square_name(square) if square < 64 else None,
                           'piece': chess.Piece(kind, identity < 16).symbol() if kind else None,
                           'predicted': chess.square_name(pred_square) if pred_square < 64 else None,
                           'predicted_piece': chess.Piece(pred_kind, identity < 16).symbol() if pred_kind else None})
        timeline.append(position | {'counts': score_position(actual, predicted), 'pieces': pieces,
                                    'next_move': metadata[i + 1]['last_move'] if i + 1 < len(metadata) else None})
    return {'metric_protocol': METRIC_PROTOCOL, 'probe': probe,
            'trace': {**report(timeline), 'timeline': timeline}}
