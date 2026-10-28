"""Next-move pretraining with optional move-involved piece-location supervision."""
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

from nanochat.gpt import GPT, GPTConfig
from data_prep.prepare_involved_pieces import INITIAL_SQUARES, file_hash
from training.common import build_checkpoint, config_from_env, get_device, load_model_state
from training.involved_pieces import InvolvedPieceModel, collision_loss, living_piece_loss, loss_terms, make_batch, occupancy_batch, occupancy_counts


def main():
    data_dir = ROOT / "data" / os.getenv("DATASET_DIR", "processed")
    labels_dir = Path(os.environ["PIECE_LABELS_DIR"])
    output = Path(os.environ["CKPT_DIR"])
    weight = float(os.getenv("PIECE_LOSS_WEIGHT", "0.1"))
    seed = int(os.getenv("SEED", "42"))
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    steps = int(os.getenv("MAX_ITERS", "1000"))
    accumulation = int(os.getenv("GRAD_ACCUM_STEPS", "1"))
    eval_interval = int(os.getenv("EVAL_INTERVAL", "100"))
    val_batches = int(os.getenv("VAL_BATCHES", "10"))
    lr = float(os.getenv("LR", "1e-5"))
    head_lr = float(os.getenv("PIECE_HEAD_LR", str(lr)))
    freeze_policy = os.getenv("FREEZE_POLICY", "0") == "1"
    init_head = os.getenv("INIT_PIECE_HEAD", "0") == "1"
    collision_weight = float(os.getenv("COLLISION_LOSS_WEIGHT", "0"))
    occupancy_dir = os.getenv("PIECE_OCCUPANCY_DIR")
    location_scope = os.getenv("PIECE_LOCATION_SCOPE", "involved")
    if location_scope not in ("involved", "all_living") or (location_scope == "all_living" and not occupancy_dir):
        raise ValueError("PIECE_LOCATION_SCOPE must be involved or all_living; all_living requires occupancy labels")
    if collision_weight < 0 or (collision_weight > 0 and not occupancy_dir):
        raise ValueError("Collision loss needs a nonnegative weight and PIECE_OCCUPANCY_DIR")
    if weight < 0 or min(batch_size, steps, accumulation, eval_interval, val_batches) < 1:
        raise ValueError("Use a nonnegative piece loss weight and positive training counts")
    if min(lr, head_lr) <= 0 or (freeze_policy and weight == 0):
        raise ValueError("Learning rates must be positive; a frozen policy needs nonzero piece loss")
    meta = pickle.loads((data_dir / "meta.pkl").read_bytes())
    manifest = json.loads((labels_dir / "manifest.json").read_text())
    if (manifest["version"] != 1 or manifest["identity_start_squares"] != list(INITIAL_SQUARES)
            or manifest["meta_sha256"] != file_hash(data_dir / "meta.pkl")):
        raise ValueError("Piece-label metadata does not match the dataset")
    splits = {}
    for split in ("train", "val"):
        source, labels = data_dir / f"{split}.bin", labels_dir / f"{split}.npy"
        record = manifest["splits"][split]
        if file_hash(source) != record["source_sha256"] or file_hash(labels) != record["labels_sha256"]:
            raise ValueError(f"{split} tokens or piece labels changed since preparation")
        splits[split] = (np.memmap(source, dtype=np.uint16, mode="r"), np.load(labels, mmap_mode="r"))

    occupancy = {}
    if occupancy_dir:
        folder = Path(occupancy_dir)
        occupancy_manifest = json.loads((folder / "manifest.json").read_text())
        if occupancy_manifest["version"] != 1 or occupancy_manifest["piece_labels"] != manifest:
            raise ValueError("Occupancy labels differ from piece-label data")
        for split in splits:
            path = folder / f"{split}_alive.npy"
            if file_hash(path) != occupancy_manifest["splits"][split]["alive_sha256"]:
                raise ValueError(f"{split} occupancy labels changed")
            occupancy[split] = np.load(path, mmap_mode="r")
        path = folder / "val_locations.npy"
        if file_hash(path) != occupancy_manifest["splits"]["val"]["locations_sha256"]:
            raise ValueError("Validation locations changed")
        location_labels = {"val": np.load(path, mmap_mode="r")}
        if location_scope == "all_living":
            location_manifest = json.loads((folder / "train_locations.json").read_text())
            path = folder / "train_locations.npy"
            if location_manifest["piece_labels"] != manifest or file_hash(path) != location_manifest["locations_sha256"]:
                raise ValueError("All-piece training locations differ from the dataset")
            location_labels["train"] = np.load(path, mmap_mode="r")

    torch.manual_seed(seed)
    device = get_device()
    initial_path = os.getenv("INIT_CHECKPOINT")
    initial = torch.load(initial_path, map_location="cpu", weights_only=False) if initial_path else None
    if initial:
        if initial["meta"]["tokenizer"]["stoi"] != meta["stoi"]:
            raise ValueError("Initialization tokenizer differs from training data")
        config = GPTConfig(**initial["meta"]["model_config"])
    else:
        config = config_from_env(meta["context_length"], meta["vocab_size"])
    policy = GPT(config)
    if initial:
        load_model_state(policy, initial)
    model = InvolvedPieceModel(policy).to(device)
    if init_head:
        if initial is None or "piece_head" not in initial:
            raise ValueError("INIT_PIECE_HEAD requires an initialization checkpoint with a piece head")
        model.piece_head.load_state_dict(initial["piece_head"])
    del initial
    policy.requires_grad_(not freeze_policy)
    model.piece_head.requires_grad_(weight > 0 or collision_weight > 0)
    separate_head = head_lr != lr or freeze_policy or init_head
    if separate_head:
        optimizer = torch.optim.AdamW([
            {"params": policy.parameters(), "lr": lr},
            {"params": model.piece_head.parameters(), "lr": head_lr},
        ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_rng = torch.Generator().manual_seed(seed)
    protocol = {"objective": "move_involved_piece_locations_v1", "piece_loss_weight": weight,
                "labels": manifest, "seed": seed, "batch_size": batch_size,
                "grad_accum_steps": accumulation, "lr": lr,
                "val_batches": val_batches, "eval_interval": eval_interval,
                "initialization": initial_path, "selection": "piece_loss" if freeze_policy else "next_move_loss"}
    if separate_head:
        protocol.update(head_lr=head_lr, freeze_policy=freeze_policy, init_piece_head=init_head)
    if occupancy_dir:
        protocol.update(collision_weight=collision_weight, occupancy=occupancy_manifest,
                        collision_objective="expected_colliding_pairs_per_living_piece_v1")
        if freeze_policy:
            protocol["selection"] = "head_loss"
    if location_scope == "all_living":
        protocol.update(objective="all_living_piece_locations_v1", location_scope=location_scope,
                        train_locations=location_manifest)
    output.mkdir(parents=True, exist_ok=True)
    resume = output / "latest.pt"
    best, start = float("inf"), 1
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["protocol"] != protocol or saved["meta"]["model_config"] != config.__dict__:
            raise ValueError("Resume configuration differs from this run")
        load_model_state(policy, saved)
        model.piece_head.load_state_dict(saved["piece_head"])
        optimizer.load_state_dict(saved["optimizer"])
        train_rng.set_state(saved["train_rng"])
        best, start = saved["best_metric"], saved["step"] + 1
        del saved

    for split, (tokens, labels) in splits.items():
        if len(tokens) <= config.sequence_len or labels.shape != (len(tokens), 2, 3):
            raise ValueError(f"{split} must contain more than one context of tokens with aligned labels")

    compiled = torch.compile(model) if device.type == "cuda" else model

    def batch(split, generator):
        tokens, labels = splits[split]
        starts = torch.randint(len(tokens) - config.sequence_len, (batch_size,), generator=generator).tolist()
        tensors = make_batch(tokens, labels, starts, config.sequence_len, meta["stoi"]["<bos>"])
        alive, truth = None, None
        if occupancy_dir:
            alive = occupancy_batch(occupancy[split], starts, config.sequence_len, tensors[0], meta["stoi"]["<bos>"]).to(device)
            if split == "val" or location_scope == "all_living":
                truth = torch.from_numpy(np.stack([location_labels[split][i + 1:i + config.sequence_len + 1]
                                                  for i in starts]).astype(np.int64)).to(device)
        return *[tensor.to(device) for tensor in tensors], alive, truth

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    @torch.no_grad()
    def evaluate():
        model.eval()
        generator = torch.Generator().manual_seed(seed + 1)
        move_sum, piece_sum, count = 0.0, 0.0, 0
        role_counts, role_correct = [0] * 3, [0] * 3
        collision_sum, living_loss_sum, board_counts = 0.0, 0.0, {}
        for _ in range(val_batches):
            inputs, targets, involved, alive, truth = batch("val", generator)
            with autocast():
                moves, locations = compiled(inputs)
                move_loss, piece_loss, correct = loss_terms(moves, locations, targets, involved)
            if alive is not None:
                counts = occupancy_counts(locations, alive, truth)
                living_loss_sum += float(living_piece_loss(locations.float(), truth, alive)) * counts["living_count"]
                collision_sum += float(collision_loss(locations, alive)) * counts["scored_positions"]
                for key, value in counts.items():
                    board_counts[key] = board_counts.get(key, 0) + value
            n = int((involved[..., 1] >= 0).sum())
            move_sum += float(move_loss)
            piece_sum += float(piece_loss) * n
            count += n
            for role in range(3):
                mask = involved[..., 2] == role
                role_counts[role] += int(mask.sum())
                role_correct[role] += int((correct & mask).sum())
        metrics = {"next_move_loss": move_sum / val_batches,
                   "piece_loss": piece_sum / count if count else None,
                   "piece_accuracy": sum(role_correct) / count if count else None,
                   "supervised_pieces": count}
        for role, name in enumerate(("mover", "captured", "castling_rook")):
            metrics[f"{name}_accuracy"] = role_correct[role] / role_counts[role] if role_counts[role] else None
            metrics[f"{name}_count"] = role_counts[role]
        if occupancy_dir:
            metrics.update(board_counts)
            metrics["collision_loss"] = collision_sum / max(board_counts["scored_positions"], 1)
            metrics["living_accuracy"] = board_counts["living_correct"] / max(board_counts["living_count"], 1)
            metrics["collision_position_rate"] = board_counts["collision_positions"] / max(board_counts["scored_positions"], 1)
            metrics["living_loss"] = living_loss_sum / max(board_counts["living_count"], 1)
            objective_loss = metrics["living_loss"] if location_scope == "all_living" else metrics["piece_loss"]
            metrics["head_loss"] = weight * objective_loss + collision_weight * metrics["collision_loss"]
        return metrics

    def save(path, step):
        saved = build_checkpoint(policy, config, meta, step, piece_head=model.piece_head.state_dict(),
                                 optimizer=optimizer.state_dict(), train_rng=train_rng.get_state(),
                                 best_metric=best, protocol=protocol)
        temporary = path.with_suffix(".tmp")
        torch.save(saved, temporary)
        temporary.replace(path)

    print(json.dumps(protocol), flush=True)
    if occupancy_dir and start == 1:
        baseline = {"step": 0, **evaluate()}
        (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
        print(json.dumps(baseline), flush=True)
    for step in range(start, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accumulation):
            inputs, targets, involved, alive, truth = batch("train", train_rng)
            with autocast():
                moves, locations = compiled(inputs)
                move_loss, piece_loss, _ = loss_terms(moves, locations, targets, involved)
                if location_scope == "all_living":
                    piece_loss = living_piece_loss(locations, truth, alive)
                loss = move_loss + weight * piece_loss
                if collision_weight:
                    loss = loss + collision_weight * collision_loss(locations, alive)
            (loss / accumulation).backward()
            total += float(loss.detach()) / accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % eval_interval == 0 or step == steps:
            metrics = evaluate()
            if not metrics["supervised_pieces"]:
                raise ValueError("Validation windows contain no complete game prefixes; increase context or validation coverage")
            if metrics[protocol["selection"]] < best:
                best = metrics[protocol["selection"]]
                save(output / "best.pt", step)
            save(resume, step)
            row = {"step": step, "train_loss": total, **metrics}
            with (output / "metrics.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
