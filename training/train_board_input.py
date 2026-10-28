"""Matched next-move continuation with real or constant board inputs."""
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data_prep.prepare_involved_pieces import file_hash
from nanochat.gpt import GPT, GPTConfig
from training.board_input import BoardInputModel
from training.common import build_checkpoint, get_device, load_model_state


def main():
    data_dir = Path(os.environ["DATASET_DIR"])
    board_dir = Path(os.environ["BOARD_INPUT_DIR"])
    output = Path(os.environ["CKPT_DIR"])
    initial_path = os.environ["INIT_CHECKPOINT"]
    enabled = os.getenv("BOARD_INPUT_ENABLED", "1") == "1"
    steps = int(os.getenv("MAX_ITERS", "1000"))
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    seed = int(os.getenv("SEED", "42"))
    lr = float(os.getenv("LR", "3e-6"))
    encoder_lr = float(os.getenv("BOARD_ENCODER_LR", "1e-4"))
    val_batches = int(os.getenv("VAL_BATCHES", "10"))
    eval_interval = int(os.getenv("EVAL_INTERVAL", "100"))
    meta = pickle.loads((data_dir / "meta.pkl").read_bytes())
    manifest = json.loads((board_dir / "manifest.json").read_text())
    assert manifest["version"] == 1
    assert manifest["source_labels"]["meta_sha256"] == file_hash(data_dir / "meta.pkl")
    splits = {}
    for split in ("train", "val"):
        tokens_path, boards_path = data_dir / f"{split}.bin", board_dir / f"{split}.npy"
        assert file_hash(tokens_path) == manifest["source_labels"]["splits"][split]["source_sha256"]
        assert file_hash(boards_path) == manifest["splits"][split]["features_sha256"]
        tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
        boards = np.load(boards_path, mmap_mode="r")
        assert boards.shape == (len(tokens), 32, 2)
        splits[split] = tokens, boards
    torch.manual_seed(seed)
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    assert initial["meta"]["tokenizer"]["stoi"] == meta["stoi"]
    config = GPTConfig(**initial["meta"]["model_config"])
    policy = GPT(config)
    load_model_state(policy, initial)
    del initial
    device = get_device()
    model = BoardInputModel(policy, enabled).to(device)
    optimizer = torch.optim.AdamW([
        {"params": policy.parameters(), "lr": lr},
        {"params": model.encoder.parameters(), "lr": encoder_lr},
    ])
    rng = torch.Generator().manual_seed(seed)
    protocol = {"objective": "next_move_ce_with_board_input_v1", "board_enabled": enabled,
                "steps": steps, "seed": seed, "batch_size": batch_size, "sequence_len": config.sequence_len,
                "lr": lr, "encoder_lr": encoder_lr, "piece_dim": model.piece_dim,
                "val_batches": val_batches, "eval_interval": eval_interval,
                "initialization": initial_path, "initial_sha256": file_hash(initial_path),
                "board_data": manifest, "selection": "final_step_for_matched_comparison"}
    output.mkdir(parents=True, exist_ok=True)
    start = 1
    if (output / "latest.pt").exists():
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        assert saved["protocol"] == protocol
        load_model_state(policy, saved)
        model.encoder.load_state_dict(saved["board_input"]["encoder"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["train_rng"])
        start = saved["step"] + 1
        del saved
    compiled = torch.compile(model) if device.type == "cuda" else model

    def batch(split, generator):
        tokens, boards = splits[split]
        starts = torch.randint(len(tokens) - config.sequence_len, (batch_size,), generator=generator).tolist()
        x = np.stack([tokens[i:i + config.sequence_len] for i in starts]).astype(np.int64)
        y = np.stack([tokens[i + 1:i + config.sequence_len + 1] for i in starts]).astype(np.int64)
        b = np.stack([boards[i:i + config.sequence_len] for i in starts]).astype(np.int64)
        return [torch.from_numpy(a).to(device) for a in (x, y, b)]

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    @torch.no_grad()
    def evaluate():
        model.eval()
        generator = torch.Generator().manual_seed(seed + 1)
        totals = {"next_move_loss": 0.0, "blank_board_loss": 0.0,
                  "complete_prefix_move_loss": 0.0, "complete_prefix_blank_loss": 0.0}
        complete_count = 0
        for _ in range(val_batches):
            x, y, boards = batch("val", generator)
            complete = (x == meta["stoi"]["<bos>"]).cumsum(1) > 0
            complete &= (y != meta["stoi"]["<bos>"]) & (y != meta["stoi"]["<eos>"])
            with autocast():
                actual = compiled(x, boards, targets=y, loss_reduction="none").view_as(y)
                blank = compiled(x, boards, targets=y, loss_reduction="none", blank=True).view_as(y)
            totals["next_move_loss"] += float(actual.mean()) / val_batches
            totals["blank_board_loss"] += float(blank.mean()) / val_batches
            totals["complete_prefix_move_loss"] += float(actual[complete].sum())
            totals["complete_prefix_blank_loss"] += float(blank[complete].sum())
            complete_count += int(complete.sum())
        for key in ("complete_prefix_move_loss", "complete_prefix_blank_loss"):
            totals[key] /= complete_count
        with autocast():
            encoded = model.encoder(boards if enabled else torch.stack(
                (torch.full_like(boards[..., 0], 64), torch.zeros_like(boards[..., 1])), -1))
        totals.update(complete_prefix_move_count=complete_count,
                      board_vector_rms=float(encoded.float().square().mean().sqrt()),
                      move_embedding_rms=float(policy.transformer.wte(x).float().square().mean().sqrt()))
        return totals

    def save(path, step, include_optimizer):
        extra = {"optimizer": optimizer.state_dict(), "train_rng": rng.get_state()} if include_optimizer else {}
        saved = build_checkpoint(policy, config, meta, step, board_input=model.checkpoint_metadata(),
                                 protocol=protocol, **extra)
        temporary = path.with_suffix(".tmp")
        torch.save(saved, temporary)
        temporary.replace(path)

    print(json.dumps(protocol), flush=True)
    if start == 1:
        baseline = {"step": 0, **evaluate()}
        (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
        print(json.dumps(baseline), flush=True)
    for step in range(start, steps + 1):
        model.train()
        x, y, boards = batch("train", rng)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            loss = compiled(x, boards, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % eval_interval == 0 or step == steps:
            row = {"step": step, "train_loss": float(loss.detach()), **evaluate()}
            with (output / "metrics.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            save(output / "latest.pt", step, True)
    save(output / "final.pt", steps, False)


if __name__ == "__main__":
    main()
