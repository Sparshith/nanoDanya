"""Bounded exact legal-action policy improvement on game-disjoint positions."""

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random

import torch

from nanochat.gpt import GPT, GPTConfig
from training.common import build_checkpoint, get_device, load_model_state
from training.grpo import encode_rewards, exact_loss_terms, last_logprobs, reward_vocabulary


def encode_positions(records, stoi, context):
    vocabulary = reward_vocabulary(stoi)
    encoded = []
    for record in records:
        ids, rewards = encode_rewards(record["moves"], vocabulary)
        if not 0 < len(record["prefix"]) <= context or len(ids) != len(set(ids)) or not ids:
            raise ValueError("Invalid history or ambiguous reward mapping")
        best_cp = max(record["cp"].values())
        blunder_ids, _ = encode_rewards({san: 1 for san, cp in record["cp"].items() if best_cp - cp >= 300}, vocabulary)
        encoded.append(dict(record, ids=ids, rewards=rewards, blunder_ids=blunder_ids))
    return encoded


def make_batch(records, vocab_size, device):
    lengths = torch.tensor([len(row["prefix"]) for row in records], device=device)
    tokens = torch.zeros((len(records), int(lengths.max())), dtype=torch.long, device=device)
    rewards = torch.zeros((len(records), vocab_size), device=device)
    legal = torch.zeros_like(rewards, dtype=torch.bool)
    blunders = torch.zeros_like(legal)
    for i, row in enumerate(records):
        tokens[i, :len(row["prefix"])] = torch.tensor(row["prefix"], device=device)
        rewards[i, row["ids"]] = torch.tensor(row["rewards"], device=device)
        legal[i, row["ids"]] = True
        blunders[i, row["blunder_ids"]] = True
    return tokens, lengths - 1, rewards, legal, blunders


def policy_metrics(logp, reference, rewards, legal, blunders):
    conditional = logp.masked_fill(~legal, -torch.inf).softmax(-1)
    best = rewards.masked_fill(~legal, -torch.inf).max(-1, keepdim=True).values
    best_mask = legal & (rewards == best)
    return {"reward": (conditional * rewards).sum(-1),
            "best_mass": (conditional * best_mask).sum(-1),
            "blunder_mass": (conditional * blunders).sum(-1),
            "illegal_mass": (logp.exp() * ~legal).sum(-1),
            "raw_top1_illegal": (~legal.gather(1, logp.argmax(-1, keepdim=True))).float().squeeze(-1),
            "kl": (logp.exp() * (logp - reference)).sum(-1)}


def priority_indices(records):
    indices = torch.tensor([i for i, row in enumerate(records) if row["priority"]], dtype=torch.long)
    if not len(indices):
        raise ValueError("No winning/playable-blunder positions")
    return indices


def training_indices(records, size, generator, priority=None):
    uniform = torch.randint(len(records), (size - size // 2,), generator=generator)
    if priority is None:
        priority = priority_indices(records)
    targeted = priority[torch.randint(len(priority), (size // 2,), generator=generator)]
    return torch.cat((uniform, targeted)).tolist()


def validation_positions(records, per_game, seed):
    if not per_game:
        return records
    games = {}
    for record in records:
        games.setdefault(record["game_id"], []).append(record)
    selected = []
    for game_id, positions in sorted(games.items()):
        rng = random.Random(f"validation:{seed}:{game_id}")
        selected.extend(sorted(rng.sample(positions, min(per_game, len(positions))), key=lambda row: row["ply"]))
    return selected


def save_progress(path, model, config, tokenizer, optimizer, generator, history, protocol, best_step, best_reward):
    best_checkpoint = None
    if best_step:
        best_checkpoint = torch.load(path.parent / "best.pt", map_location="cpu", weights_only=False)
        if best_checkpoint["step"] != best_step:
            raise ValueError("Best checkpoint does not match selection history")
    checkpoint = build_checkpoint(model, config, tokenizer, history[-1]["step"],
                                  optimizer=optimizer.state_dict(), generator_state=generator.get_state(),
                                  torch_rng=torch.get_rng_state(),
                                  cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                                  history=history, protocol=protocol, best_step=best_step, best_reward=best_reward,
                                  best_checkpoint=best_checkpoint)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def restore_progress(path, model, optimizer, generator, protocol):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["protocol"] != protocol:
        raise ValueError("Resume protocol differs from the saved run")
    load_model_state(model, checkpoint)
    optimizer.load_state_dict(checkpoint["optimizer"])
    generator.set_state(checkpoint["generator_state"])
    torch.set_rng_state(checkpoint["torch_rng"])
    if checkpoint["cuda_rng"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if checkpoint["best_step"]:
        if checkpoint["best_checkpoint"]["step"] != checkpoint["best_step"]:
            raise ValueError("Saved best checkpoint does not match selection history")
        temporary = path.parent / "best.tmp"
        torch.save(checkpoint["best_checkpoint"], temporary)
        temporary.replace(path.parent / "best.pt")
    return checkpoint["history"], checkpoint["best_step"], checkpoint["best_reward"]


def main():
    dataset = Path(os.environ["EXACT_DATASET"])
    output = Path(os.environ["EXACT_OUTPUT"])
    resume = os.getenv("EXACT_RESUME", "0") == "1"
    output.mkdir(parents=True, exist_ok=resume)
    if resume and not (output / "resume.pt").exists():
        raise ValueError("Resume requires an existing progress checkpoint")
    seed = int(os.getenv("SEED", "20260918"))
    steps = int(os.getenv("MAX_ITERS", "300"))
    lr = float(os.getenv("LR", "3e-6"))
    eval_interval = int(os.getenv("EVAL_INTERVAL", "25"))
    val_per_game = int(os.getenv("VAL_PER_GAME", "0"))
    if steps < 1 or eval_interval < 1 or val_per_game < 0:
        raise ValueError("Invalid training or validation budget")
    beta, temperature, batch_size = 0.05, 0.8, 64
    micro_batch = int(os.getenv("MICRO_BATCH", "8"))
    if micro_batch < 1 or batch_size % micro_batch:
        raise ValueError("Micro batch must divide the effective batch of 64")
    checkpoint_path = os.getenv("CKPT_INIT", "/data/checkpoints/plain/games-15m/l16_best.pt")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = GPTConfig(**checkpoint["meta"]["model_config"])
    tokenizer = checkpoint["meta"]["tokenizer"]
    model, reference = GPT(config).to(device), GPT(config).to(device)
    load_model_state(model, checkpoint)
    load_model_state(reference, checkpoint)
    reference.eval().requires_grad_(False)
    del checkpoint
    manifest = json.loads((dataset / "manifest.json").read_text())

    def load_split(split):
        path = dataset / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][split]:
            raise ValueError("Dataset hash changed")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert records and all(row["split"] == split for row in records)
        return encode_positions(records, tokenizer["stoi"], config.sequence_len)

    train, val = load_split("train"), load_split("val")
    assert {row["game_id"] for row in train}.isdisjoint(row["game_id"] for row in val)
    val = validation_positions(val, val_per_game, seed)
    priority = priority_indices(train)
    protocol = {"checkpoint": checkpoint_path, "dataset_manifest": manifest, "seed": seed,
                "steps": steps, "lr": lr, "beta": beta, "temperature": temperature,
                "eval_interval": eval_interval, "val_per_game": val_per_game,
                "validation_positions": len(val),
                "batch_size": batch_size, "micro_batch": micro_batch, "trainable": "all parameters",
                "action_support": "legal tokens in the fixed vocabulary; unrepresentable moves have zero probability",
                "objective": "exact legal-conditional expected engine reward plus full-vocabulary forward KL",
                "selection": "fixed validation reward; KL <= 0.2 and illegal mass <= baseline + 0.005",
                "evaluation": "test split once for selected checkpoint, followed by fresh full games",
                "training_sampling": "half uniform and half priority, with replacement"}
    if not resume:
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    @torch.no_grad()
    def evaluate(records):
        model.eval()
        rows = []
        for offset in range(0, len(records), micro_batch):
            batch = records[offset:offset + micro_batch]
            tokens, last, rewards, legal, blunders = make_batch(batch, config.vocab_size, device)
            with autocast():
                logp = last_logprobs(model, tokens, last, temperature)
                ref = last_logprobs(reference, tokens, last, temperature)
                metrics = policy_metrics(logp, ref, rewards, legal, blunders)
            for i, record in enumerate(batch):
                rows.append({"game_id": record["game_id"], "ply": record["ply"],
                             **{key: float(value[i]) for key, value in metrics.items()}})
        return {key: sum(row[key] for row in rows) / len(rows) for key in metrics}, rows

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if resume:
        history, best_step, best_reward = restore_progress(output / "resume.pt", model, optimizer, generator, protocol)
        baseline = history[0]["validation"]
    else:
        baseline, _ = evaluate(val)
        best_reward, best_step = baseline["reward"], 0
        history = [{"step": 0, "validation": baseline, "eligible": True}]
    print(json.dumps(history[-1]), flush=True)
    for step in range(history[-1]["step"] + 1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        indices = training_indices(train, batch_size, generator, priority)
        for offset in range(0, batch_size, micro_batch):
            batch = [train[i] for i in indices[offset:offset + micro_batch]]
            tokens, last, rewards, legal, _ = make_batch(batch, config.vocab_size, device)
            with autocast():
                logp = last_logprobs(model, tokens, last, temperature)
                with torch.no_grad():
                    ref = last_logprobs(reference, tokens, last, temperature)
                policy, kl, _ = exact_loss_terms(logp, ref, rewards, legal)
                loss = (policy + beta * kl) * len(batch) / batch_size
            loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % eval_interval == 0 or step == steps:
            metrics, _ = evaluate(val)
            eligible = metrics["kl"] <= 0.2 and metrics["illegal_mass"] <= baseline["illegal_mass"] + 0.005
            if eligible and metrics["reward"] > best_reward:
                best_step, best_reward = step, metrics["reward"]
                torch.save(build_checkpoint(model, config, tokenizer, step, protocol=protocol), output / "best.pt")
            row = {"step": step, "validation": metrics, "eligible": eligible,
                   "gradient_norm": float(gradient), "best_step": best_step}
            history.append(row)
            (output / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
            save_progress(output / "resume.pt", model, config, tokenizer, optimizer, generator,
                          history, protocol, best_step, best_reward)
            print(json.dumps(row), flush=True)
    torch.save(build_checkpoint(model, config, tokenizer, steps, optimizer=optimizer.state_dict(), protocol=protocol), output / "final.pt")
    test = load_split("test")
    assert {row["game_id"] for row in test}.isdisjoint(row["game_id"] for row in train + val)
    load_model_state(model, {"model": reference.state_dict()})
    test_baseline, baseline_rows = evaluate(test)
    if best_step:
        saved = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        load_model_state(model, saved)
        del saved
    selected, selected_rows = evaluate(test)
    result = {"best_step": best_step, "baseline": test_baseline, "selected": selected,
              "positions": len(test), "games": len({row["game_id"] for row in test}),
              "baseline_rows": baseline_rows, "selected_rows": selected_rows}
    (output / "test.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("_rows")}), flush=True)


if __name__ == "__main__":
    main()
