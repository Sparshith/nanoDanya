"""Read the trained identity-location head on the traced game."""
import io

import chess
import chess.pgn
import torch

from chess_token_utils import resolve_token_id
from data_prep.prepare_involved_pieces import INITIAL_SQUARES, label_move
from nanochat.gpt import GPT, GPTConfig
from training.common import forward_hidden, load_model_state
from training.involved_pieces import InvolvedPieceModel


@torch.inference_mode()
def inspect_head(checkpoint_path, pgn, device="cuda", all_positions=False):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = GPTConfig(**saved["meta"]["model_config"])
    stoi = saved["meta"]["tokenizer"]["stoi"]
    model = InvolvedPieceModel(GPT(config))
    load_model_state(model.policy, saved)
    model.piece_head.load_state_dict(saved["piece_head"])
    del saved
    model.to(device).eval()
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    identities = dict(zip(INITIAL_SQUARES, range(32)))
    prefix = [stoi["<bos>"]]
    positions = []
    previous = None
    for ply, move in enumerate([*game.mainline_moves(), None]):
        san = board.san(move) if move is not None else None
        if all_positions or ply in range(1, 72, 2):
            involved = label_move(board.copy(), identities.copy(), move) if move is not None else []
            positions.append({"ply": ply, "prefix": prefix.copy(), "next_move": san,
                              "last_move": previous, "fen": board.fen(),
                              "side_to_move": "White" if board.turn else "Black",
                              "living": [(identity, square, board.piece_at(square).symbol())
                                         for square, identity in sorted(identities.items())],
                              "involved": {identity for identity, _, _ in involved}})
        if move is None:
            break
        token = resolve_token_id(stoi, san)
        if token is None:
            raise ValueError(f"Trace move missing from tokenizer: {san}")
        prefix.append(token)
        label_move(board, identities, move)
        previous = san
    length = max(len(row["prefix"]) for row in positions)
    if length > config.sequence_len:
        raise ValueError("Trace exceeds model context")
    tokens = torch.full((len(positions), length), stoi["<eos>"], dtype=torch.long, device=device)
    for i, position in enumerate(positions):
        tokens[i, :len(position["prefix"])] = torch.tensor(position["prefix"], device=device)
    hidden = forward_hidden(model.policy, tokens)
    last = torch.tensor([len(row["prefix"]) - 1 for row in positions], device=device)
    probabilities = model.piece_head(hidden[torch.arange(len(positions), device=device), last]).view(-1, 32, 64).softmax(-1).cpu()
    predictions = probabilities.argmax(-1)
    timeline = []
    for i, position in enumerate(positions):
        pieces = []
        living = {identity: (square, symbol) for identity, square, symbol in position["living"]}
        for identity in range(32):
            square, symbol = living.get(identity, (None, chess.Board().piece_at(INITIAL_SQUARES[identity]).symbol()))
            predicted = int(predictions[i, identity])
            pieces.append({"identity": chess.square_name(INITIAL_SQUARES[identity]), "piece": symbol,
                           "alive": square is not None,
                           "actual": chess.square_name(square) if square is not None else None,
                           "predicted": chess.square_name(predicted),
                           "predicted_probability": float(probabilities[i, identity, predicted]),
                           "correct": square == predicted, "involved": identity in position["involved"],
                           "actual_probability": float(probabilities[i, identity, square]) if square is not None else None})
        timeline.append({key: position[key] for key in ("ply", "next_move", "last_move", "fen", "side_to_move")}
                        | {"pieces": pieces, "input_tokens": len(position["prefix"])})
    all_pieces = [piece for row in timeline for piece in row["pieces"] if piece["alive"]]
    involved = [piece for piece in all_pieces if piece["involved"]]
    return {"checkpoint": checkpoint_path, "timeline": timeline,
            "known_living_correct": sum(piece["correct"] for piece in all_pieces),
            "known_living_count": len(all_pieces),
            "involved_correct": sum(piece["correct"] for piece in involved),
            "involved_count": len(involved),
            "protocol": "Identity-specific current locations. Truth determines which pieces are alive for scoring; this does not measure captured-piece detection or full-board validity."}
