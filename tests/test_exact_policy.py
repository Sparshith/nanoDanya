from copy import deepcopy
import json
from pathlib import Path
import shutil

import chess
import pytest
import torch

from data_prep import prepare_own_game_rewards as prep
from chess_inference import legal_token_ids
from training.grpo import exact_loss_terms, last_logprobs
from training.common import build_checkpoint
from training.train_exact_policy import (encode_positions, make_batch, policy_metrics, priority_indices,
                                         restore_progress, save_progress, training_indices, validation_positions)
from nanochat.gpt import GPT, GPTConfig


def test_whole_game_splits_are_order_independent_and_disjoint():
    games = [{"game_id": str(i)} for i in range(100)]
    splits = prep.game_splits(games, 9)
    assert splits == prep.game_splits(list(reversed(games)), 9)
    assert [list(splits.values()).count(name) for name in ("train", "val", "test")] == [80, 10, 10]
    with pytest.raises(ValueError, match="Duplicate"):
        prep.game_splits(games + games[:1], 9)


def test_candidate_histories_replay_to_own_turn_with_matching_tokens():
    source = Path(__file__).parents[1] / "benchmark/results/sf1500_plain_games15m_l16_400.jsonl"
    if not source.exists():
        pytest.skip("Archived benchmark is local-only")
    game = next(row for row in map(json.loads, source.read_text().splitlines()) if row["type"] == "game")
    board, stoi = chess.Board(), {"<bos>": 0}
    for san in game["moves"]:
        move = board.parse_san(san)
        canonical = board.san(move)
        if canonical not in stoi:
            stoi[canonical] = len(stoi)
        board.push(move)
    rows = prep.candidate_positions(game, stoi, 512, 9, per_game=12)
    assert rows == prep.candidate_positions(game, stoi, 512, 9, per_game=12)
    assert len(rows) == 12
    for row in rows:
        board, tokens = chess.Board(), [0]
        for san in row["history"].split():
            move = board.parse_san(san)
            tokens.append(stoi[board.san(move)])
            board.push(move)
        assert board.fen() == row["fen"] and tokens == row["prefix"]
        assert board.turn == (game["model_color"] == "white")


@pytest.mark.parametrize("history, mate", [("e4 e5 Qh5 Nc6 Bc4 Nf6", "Qxf7#"), ("f3 e5 g4", "Qh4#")])
def test_engine_grading_covers_every_move_and_mates_from_either_side(history, mate):
    engine = shutil.which("stockfish")
    if not engine:
        pytest.skip("Stockfish unavailable")
    board = chess.Board()
    for san in history.split():
        board.push_san(san)
    prep.init_engine(engine, 10000)
    try:
        row = prep.grade({"history": history, "fen": board.fen(), "played": mate, "game_id": "test", "ply": len(history.split())})
    finally:
        prep.ENGINE.quit()
    assert set(row["moves"]) == {board.san(move) for move in board.legal_moves}
    assert row["cp"][mate] == 10000 and row["played_cp_loss"] == 0
    assert row["moves"][mate] == pytest.approx(prep.winprob(1000))


def test_metrics_condition_on_legal_moves_and_sum_all_best_variants_and_ties():
    logp = torch.tensor([[0.4, 0.1, 0.2, 0.3]]).log()
    legal = torch.tensor([[False, True, True, True]])
    rewards = torch.tensor([[0., 1., 1., 0.]])
    blunders = torch.tensor([[False, False, False, True]])
    result = policy_metrics(logp, logp, rewards, legal, blunders)
    assert result["reward"].item() == pytest.approx(0.5)
    assert result["best_mass"].item() == pytest.approx(0.5)
    assert result["blunder_mass"].item() == pytest.approx(0.5)
    assert result["illegal_mass"].item() == pytest.approx(0.4)
    assert result["raw_top1_illegal"].item() == 1


def test_partial_vocabulary_matches_game_harness_action_support():
    board = chess.Board()
    stoi = {"<bos>": 0, "<eos>": 1, "e4": 2, "e4+": 3, "d4": 4, "e5": 5}
    moves = {board.san(move): .5 for move in board.legal_moves}
    record = {"prefix": [0], "moves": moves, "cp": {san: 0 for san in moves}}
    encoded = encode_positions([record], stoi, 8)
    assert set(encoded[0]["ids"]) == set(legal_token_ids(stoi, board, allow_eos=False))
    assert len(encoded[0]["moves"]) == 20
    _, _, rewards, legal, _ = make_batch(encoded, len(stoi), torch.device("cpu"))
    logits = torch.zeros((1, len(stoi)), requires_grad=True)
    logp = logits.log_softmax(-1)
    loss, _, _ = exact_loss_terms(logp, logp.detach(), rewards, legal)
    assert loss.item() == pytest.approx(-.5)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="Invalid history or ambiguous reward mapping"):
        encode_positions([record], {"<bos>": 0, "<eos>": 1}, 8)


def test_batched_exact_update_matches_per_position_gradients_without_sampling(monkeypatch):
    torch.manual_seed(11)
    model = GPT(GPTConfig(sequence_len=8, vocab_size=5, n_layer=1, n_head=2, n_kv_head=2, n_embd=16))
    other = deepcopy(model)
    stoi = {"<bos>": 0, "e4": 1, "e4+": 2, "d4": 3, "Nf3": 4}
    records = [{"prefix": prefix, "moves": {"e4": .9, "d4": .2}, "cp": {"e4": 600, "d4": -400}, "priority": i == 0}
               for i, prefix in enumerate(([0], [0, 1, 3]))]
    encoded = encode_positions(records, stoi, 8)
    assert encoded[0]["ids"] == [1, 2, 3]
    monkeypatch.setattr(torch, "multinomial", lambda *args, **kwargs: pytest.fail("Exact objective must not sample actions"))
    tokens, last, rewards, legal, _ = make_batch(encoded, 5, torch.device("cpu"))
    logp = last_logprobs(model, tokens, last, .8)
    loss, kl, _ = exact_loss_terms(logp, logp.detach(), rewards, legal)
    (loss + .05 * kl).backward()
    for record in encoded:
        tokens, last, rewards, legal, _ = make_batch([record], 5, torch.device("cpu"))
        logp = last_logprobs(other, tokens, last, .8)
        loss, kl, _ = exact_loss_terms(logp, logp.detach(), rewards, legal)
        ((loss + .05 * kl) / len(encoded)).backward()
    for left, right in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-4)
    indices = training_indices(encoded, 64, torch.Generator().manual_seed(3))
    assert len(indices) == 64 and indices[32:] == [0] * 32


def test_fixed_validation_sample_includes_every_game_without_using_labels():
    records = [{"game_id": game, "ply": ply, "reward": ply % 2} for game in ("a", "b", "c") for ply in range(12)]
    sample = validation_positions(records, 4, 17)
    assert len(sample) == 12
    assert {game: sum(row["game_id"] == game for row in sample) for game in ("a", "b", "c")} == {"a": 4, "b": 4, "c": 4}
    changed = [dict(row, reward=1 - row["reward"]) for row in records]
    assert [(row["game_id"], row["ply"]) for row in sample] == [
        (row["game_id"], row["ply"]) for row in validation_positions(changed, 4, 17)]
    assert validation_positions(records, 0, 17) is records


def test_precomputed_priority_keeps_the_same_sampling_sequence():
    records = [{"priority": i % 3 == 0} for i in range(20)]
    slow, fast = torch.Generator().manual_seed(21), torch.Generator().manual_seed(21)
    priority = priority_indices(records)
    for _ in range(3):
        assert training_indices(records, 64, slow) == training_indices(records, 64, fast, priority)


def test_resume_restores_optimizer_sampling_and_reference_independence(tmp_path):
    torch.manual_seed(21)
    config = GPTConfig(sequence_len=8, vocab_size=5, n_layer=1, n_head=2, n_kv_head=2, n_embd=16)
    model = GPT(config)
    frozen = deepcopy(model).requires_grad_(False)
    frozen_state = deepcopy(frozen.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    generator = torch.Generator().manual_seed(21)
    records = [{"prefix": [0, i + 1], "ids": [1, 2], "rewards": [.8, .2], "blunder_ids": [2], "priority": i == 0} for i in range(3)]

    def update(m, opt, gen):
        batch = [records[i] for i in training_indices(records, 4, gen)]
        tokens, last, rewards, legal, _ = make_batch(batch, 5, torch.device("cpu"))
        with torch.no_grad():
            reference = last_logprobs(frozen, tokens, last, .8)
        logp = last_logprobs(m, tokens, last, .8)
        pg, kl, _ = exact_loss_terms(logp, reference, rewards, legal)
        opt.zero_grad()
        (pg + .05 * kl).backward()
        opt.step()

    update(model, optimizer, generator)
    protocol, history = {"steps": 2}, [{"step": 0}, {"step": 1}]
    path = tmp_path / "resume.pt"
    torch.save(build_checkpoint(model, config, {}, 1), tmp_path / "best.pt")
    save_progress(path, model, config, {}, optimizer, generator, history, protocol, 1, .8)
    random_after_save = torch.rand(3)
    update(model, optimizer, generator)
    restored = GPT(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    restored_generator = torch.Generator()
    (tmp_path / "best.pt").write_bytes(b"interrupted checkpoint write")
    result = restore_progress(path, restored, restored_optimizer, restored_generator, protocol)
    assert result == (history, 1, .8)
    assert torch.load(tmp_path / "best.pt", weights_only=False)["step"] == 1
    torch.testing.assert_close(torch.rand(3), random_after_save)
    update(restored, restored_optimizer, restored_generator)
    for left, right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert all(torch.equal(value, frozen.state_dict()[key]) for key, value in frozen_state.items())
    with pytest.raises(ValueError, match="protocol differs"):
        restore_progress(path, restored, restored_optimizer, restored_generator, {"steps": 3})


def test_grading_resume_keeps_complete_rows_and_discards_only_unfinished_last_line(tmp_path):
    row = {"game_id": "a", "ply": 20, "split": "train", "priority": True, "min_depth": 12}
    complete = json.dumps(row) + "\n"
    path = tmp_path / "train.jsonl"
    path.write_text(complete + '{"game_id":')
    completed, counts, priority, depths = prep.completed_records(tmp_path)
    assert completed == {("a", 20)}
    assert dict(counts) == {"train": 1} and dict(priority) == {"train": 1} and depths == [12]
    assert path.read_text() == complete
    path.write_text(complete + complete)
    with pytest.raises(ValueError, match="duplicate position"):
        prep.completed_records(tmp_path)
