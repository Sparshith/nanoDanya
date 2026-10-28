"""Matched living-piece readouts of frozen models' final normalized hidden states."""
import hashlib
import io
import json
import pickle
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
from torch import nn

from chess_inference import token_for_id
from chess_token_utils import resolve_token_id
from data_prep.prepare_involved_pieces import INITIAL_SQUARES, file_hash
from inference.board_input import BoardInputPlayer, BoardState
from model_loading import load_model
from probes.build_positions import load_games
from training.board_input import BoardInputModel
from training.common import forward_hidden, get_device
from training.involved_pieces import living_piece_loss, occupancy_counts


class FrozenReadout(nn.Module):
    def __init__(self, player, seed=42, classes_per_piece=64, slots=32):
        super().__init__()
        self.backbone = player.model if isinstance(player, BoardInputPlayer) else player
        self.backbone.eval().requires_grad_(False)
        policy = self.backbone.policy if isinstance(self.backbone, BoardInputModel) else self.backbone
        self.config = policy.config
        self.classes_per_piece = classes_per_piece
        self.slots = slots
        torch.manual_seed(seed)
        self.head = nn.Linear(self.config.n_embd, slots * classes_per_piece).to(policy.transformer.wte.weight.device)

    def forward(self, tokens, boards):
        with torch.no_grad():
            if isinstance(self.backbone, BoardInputModel):
                if not self.backbone.enabled:
                    boards = torch.stack((torch.full_like(boards[..., 0], 64),
                                          torch.zeros_like(boards[..., 1])), -1)
                additions = self.backbone.encoder(boards)
                hidden = forward_hidden(self.backbone.policy, tokens, additions)
            else:
                hidden = forward_hidden(self.backbone, tokens)
        return self.head(hidden).unflatten(-1, (self.slots, self.classes_per_piece))


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def targets(tokens, boards, bos):
    truth = boards[..., 0].masked_fill(boards[..., 0] == 64, -1)
    alive = (truth >= 0) & ((tokens == bos).cumsum(1) > 0).unsqueeze(-1)
    return truth, alive


def counts(logits, truth, alive):
    result = occupancy_counts(logits, alive, truth)
    off_start = alive & (truth != torch.tensor(INITIAL_SQUARES, device=truth.device))
    result.update(off_start_count=int(off_start.sum()),
                  off_start_correct=int(((logits.argmax(-1) == truth) & off_start).sum()))
    return result


def finish_counts(total):
    return total | {"living_accuracy": total["living_correct"] / total["living_count"],
                    "exact_board_rate": total["exact_boards"] / total["scored_positions"],
                    "collision_position_rate": total["collision_positions"] / total["scored_positions"],
                    "off_start_accuracy": total["off_start_correct"] / max(total["off_start_count"], 1)}


def add_counts(total, row):
    for key, value in row.items():
        total[key] = total.get(key, 0) + value


def sequence_batch(sequences, itos, eos, device):
    length = max(map(len, sequences))
    x = np.full((len(sequences), length), eos, dtype=np.int64)
    boards = np.zeros((len(sequences), length, 32, 2), dtype=np.int64)
    boards[..., 0] = 64
    for row, sequence in enumerate(sequences):
        state = BoardState()
        x[row, :len(sequence)] = sequence
        for col, token in enumerate(sequence):
            boards[row, col] = state.append(token_for_id(itos, token))
    return torch.from_numpy(x).to(device), torch.from_numpy(boards).to(device)


def run_readout(checkpoint, data_dir, board_dir, pgn, output, steps=3000, batch_size=32, val_batches=10,
                full_board=False, square_board=False):
    from probes.full_board_readout import full_loss, evaluate_full
    from probes.square_readout import square_loss, square_loss_parts, evaluate_squares
    assert not (full_board and square_board)
    data_dir, board_dir, output = map(Path, (data_dir, board_dir, output))
    output.mkdir(parents=True, exist_ok=True)
    device = get_device()
    player, config, stoi, itos = load_model(checkpoint, str(device))
    model = FrozenReadout(player, classes_per_piece=13 if square_board else 71 if full_board else 64,
                          slots=64 if square_board else 32).to(device)
    del player
    frozen_hash = state_hash(model.backbone)
    initial_head_hash = state_hash(model.head)
    manifest = json.loads((board_dir / "manifest.json").read_text())
    meta = pickle.loads((data_dir / "meta.pkl").read_bytes())
    assert meta["stoi"] == stoi
    assert file_hash(data_dir / "meta.pkl") == manifest["source_labels"]["meta_sha256"]
    splits = {}
    for split in ("train", "val"):
        assert file_hash(data_dir / f"{split}.bin") == manifest["source_labels"]["splits"][split]["source_sha256"]
        assert file_hash(board_dir / f"{split}.npy") == manifest["splits"][split]["features_sha256"]
        splits[split] = (np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r"),
                         np.load(board_dir / f"{split}.npy", mmap_mode="r"))
    protocol = {"version": 1, "checkpoint": checkpoint, "checkpoint_sha256": file_hash(checkpoint),
                "steps": steps, "batch_size": batch_size, "sequence_len": config.sequence_len,
                "head_lr": 1e-3, "seed": 42, "val_batches": val_batches,
                "initial_head_hash": initial_head_hash, "board_data": manifest,
                "readout": "linear 32 identities x 64 squares, final normalized transformer hidden state",
                "loss": "mean living-piece location CE; complete prefixes only; no collision penalty",
                "selection": "final fixed step, no layer or checkpoint selection"}
    if full_board:
        protocol.update(version=2, readout="linear 32 identities x (65 locations including captured + 6 kinds)",
                        loss="mean all-identity location CE + mean living-identity kind CE; complete prefixes only",
                        metric_version=1)
    if square_board:
        protocol.update(version=3, readout="linear 64 squares x 13 classes, final normalized hidden state",
                        loss="per board: 0.5 mean occupied CE + 0.5 mean empty CE; then mean valid boards",
                        sampling="uniform token-window starts; no move-stage resampling",
                        material_weighting=False, metric_version=1,
                        class_order="empty, white P N B R Q K, black P N B R Q K")
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    rng = torch.Generator().manual_seed(42)
    start = 1
    latest = output / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        assert saved["protocol"] == protocol
        model.head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["train_rng"])
        start = saved["step"] + 1
    compiled = torch.compile(model) if device.type == "cuda" else model

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    def batch(split, generator):
        tokens, boards = splits[split]
        indices = torch.randint(len(tokens) - config.sequence_len, (batch_size,), generator=generator).tolist()
        x = np.stack([tokens[i:i + config.sequence_len] for i in indices]).astype(np.int64)
        b = np.stack([boards[i:i + config.sequence_len] for i in indices]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(b).to(device)

    @torch.no_grad()
    def validate():
        generator = torch.Generator().manual_seed(43)
        total, loss_sum = {}, 0.0
        square_totals = {"positions": 0, "occupied_loss_sum": 0.0, "empty_loss_sum": 0.0}
        for _ in range(val_batches):
            x, boards = batch("val", generator)
            truth, alive = targets(x, boards, stoi["<bos>"])
            with autocast():
                logits = compiled(x, boards)
                loss = (square_loss(logits, x, boards, stoi) if square_board else
                        full_loss(logits, x, boards, stoi) if full_board
                        else living_piece_loss(logits, truth, alive))
            if square_board:
                occupied, empty, valid = square_loss_parts(logits.float(), x, boards, stoi)
                square_totals["positions"] += int(valid.sum())
                square_totals["occupied_loss_sum"] += float(occupied[valid].sum())
                square_totals["empty_loss_sum"] += float(empty[valid].sum())
                continue
            if full_board:
                loss_sum += float(loss)
                continue
            row = counts(logits, truth, alive)
            loss_sum += float(loss) * row["living_count"]
            add_counts(total, row)
        if square_board:
            n = square_totals["positions"]
            occupied = square_totals["occupied_loss_sum"] / n
            empty = square_totals["empty_loss_sum"] / n
            return {"balanced_loss": (occupied + empty) / 2, "occupied_loss": occupied,
                    "empty_loss": empty, "positions": n}
        if full_board:
            return {"full_loss": loss_sum / val_batches}
        return finish_counts(total) | {"living_loss": loss_sum / total["living_count"]}

    print(json.dumps({"checkpoint": checkpoint, "readout_steps": steps}), flush=True)
    if square_board and start == 1:
        row = {"checkpoint": checkpoint, "step": 0, **validate()}
        with (output / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    for step in range(start, steps + 1):
        x, boards = batch("train", rng)
        truth, alive = targets(x, boards, stoi["<bos>"])
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = compiled(x, boards)
            loss = (square_loss(logits, x, boards, stoi) if square_board else
                    full_loss(logits, x, boards, stoi) if full_board
                    else living_piece_loss(logits, truth, alive))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 500 == 0 or step == steps:
            row = {"checkpoint": checkpoint, "step": step, "train_loss": float(loss.detach()), **validate()}
            with (output / "metrics.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
            temporary = latest.with_suffix(".tmp")
            torch.save({"protocol": protocol, "step": step, "head": model.head.state_dict(),
                        "optimizer": optimizer.state_dict(), "train_rng": rng.get_state()}, temporary)
            temporary.replace(latest)
            print(json.dumps(row), flush=True)
    assert all(p.grad is None and not p.requires_grad for p in model.backbone.parameters())
    assert state_hash(model.backbone) == frozen_hash
    torch.save({"protocol": protocol, "head": model.head.state_dict()}, output / "head.pt")
    if full_board or square_board:
        evaluate = evaluate_squares if square_board else evaluate_full
        result = {"protocol": protocol, "frozen_state_hash": frozen_hash, "backbone_bitwise_unchanged": True,
                  "validation": json.loads((output / "metrics.jsonl").read_text().splitlines()[-1]),
                  **evaluate(model, data_dir, pgn, stoi, itos, output, device)}
        if square_board:
            result["learning_curve"] = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    @torch.no_grad()
    def inspect_sequences(sequences):
        x, boards = sequence_batch(sequences, itos, stoi["<eos>"], device)
        truth, alive = targets(x, boards, stoi["<bos>"])
        # The saved heads are scored in float32, matching standard inference.
        logits = model(x, boards)
        return logits, truth, alive, boards

    games = load_games(data_dir / "probe.bin", stoi["<bos>"], stoi["<eos>"], 200)
    truncated = sum(len(game) > config.sequence_len for game in games)
    games = [game[:config.sequence_len] for game in games]
    per_game, total = [], {}
    for offset in range(0, len(games), 8):
        with torch.no_grad():
            logits, truth, alive, _ = inspect_sequences(games[offset:offset + 8])
        for i in range(len(logits)):
            row = counts(logits[i:i + 1], truth[i:i + 1], alive[i:i + 1])
            add_counts(total, row)
            per_game.append({"game_id": offset + i, **row})
    probe = {"aggregate": finish_counts(total), "per_game": per_game, "truncated_games": truncated,
             "source_sha256": file_hash(data_dir / "probe.bin")}

    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    sequence, metadata = [stoi["<bos>"]], [{"ply": 0, "fen": board.fen(), "last_move": None}]
    for ply, move in enumerate(game.mainline_moves(), 1):
        san = board.san(move)
        token = resolve_token_id(stoi, san)
        if token is None:
            raise ValueError(f"Trace move missing from tokenizer: {san}")
        sequence.append(token)
        board.push(move)
        metadata.append({"ply": ply, "fen": board.fen(), "last_move": san})
    assert len(sequence) <= config.sequence_len
    with torch.no_grad():
        logits, truth, alive, boards = inspect_sequences([sequence])
        trace_counts = finish_counts(counts(logits, truth, alive))
        probs = logits[0].softmax(-1).cpu()
        predictions = probs.argmax(-1)
        boards = boards[0].cpu()
    timeline = []
    for i, position in enumerate(metadata):
        pieces = []
        for identity, start_square in enumerate(INITIAL_SQUARES):
            actual, kind = map(int, boards[i, identity])
            predicted = int(predictions[i, identity])
            pieces.append({"identity": chess.square_name(start_square), "alive": actual != 64,
                           "piece": chess.Piece(kind, identity < 16).symbol() if kind else None,
                           "actual": chess.square_name(actual) if actual != 64 else None,
                           "predicted": chess.square_name(predicted), "correct": actual == predicted,
                           "actual_probability": float(probs[i, identity, actual]) if actual != 64 else None,
                           "predicted_probability": float(probs[i, identity, predicted])})
        timeline.append(position | {"next_move": metadata[i + 1]["last_move"] if i + 1 < len(metadata) else None,
                                    "pieces": pieces})
    result = {"protocol": protocol, "frozen_state_hash": frozen_hash, "backbone_bitwise_unchanged": True,
              "validation": json.loads((output / "metrics.jsonl").read_text().splitlines()[-1]),
              "probe": probe, "trace": {"aggregate": trace_counts, "timeline": timeline}}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
