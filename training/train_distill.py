import os
import pickle
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig

from common import get_device
from student import StudentGPT


def env_int(name, default):
    return int(os.getenv(name, default))


def learning_rate(step, schedule):
    peak = schedule["peak_lr"]
    if schedule["name"] == "constant":
        return peak
    warmup = schedule["warmup_steps"]
    decay = schedule["decay_steps"]
    if warmup and step <= warmup:
        return peak * step / warmup
    progress = min(1.0, max(0.0, (step - warmup) / (decay - warmup)))
    return schedule["min_lr"] + 0.5 * (peak - schedule["min_lr"]) * (1 + math.cos(math.pi * progress))


def sample_batch(tokens, block_size, batch_size, generator, end=None):
    end = len(tokens) if end is None else end
    last_start = end - block_size - 1
    if last_start < 0:
        raise ValueError("dataset split is smaller than BLOCK_SIZE + 1")
    starts = torch.randint(last_start + 1, (batch_size,), generator=generator).numpy()
    offsets = np.arange(block_size)
    x = np.asarray(tokens[starts[:, None] + offsets], dtype=np.int64)
    y = np.asarray(tokens[starts[:, None] + offsets + 1], dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def loss_terms(student_logits, teacher_logits, targets, temperature):
    student = student_logits.float().reshape(-1, student_logits.size(-1))
    ce = F.cross_entropy(student, targets.reshape(-1))
    if teacher_logits is None:
        return ce, ce.new_zeros(())
    teacher = teacher_logits.float().reshape(-1, teacher_logits.size(-1))
    teacher_probs = F.softmax(teacher / temperature, dim=-1)
    student_log_probs = F.log_softmax(student / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return ce, kl * temperature * temperature


def main():
    seed = env_int("SEED", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    dataset_dir = Path("data") / os.getenv("DATASET_DIR", "datasets/games/15m")
    meta = pickle.loads((dataset_dir / "meta.pkl").read_bytes())
    train_tokens = np.memmap(dataset_dir / "train.bin", dtype=np.uint16, mode="r")
    val_tokens = np.memmap(dataset_dir / "val.bin", dtype=np.uint16, mode="r")
    alpha = float(os.getenv("DISTILL_ALPHA", "0.5"))
    temperature = float(os.getenv("TEMPERATURE", "2.0"))
    if not 0 <= alpha <= 1 or temperature <= 0:
        raise ValueError("DISTILL_ALPHA must be in [0,1] and TEMPERATURE must be positive")
    teacher, teacher_path = None, None
    sequence_len = meta["context_length"]
    if alpha:
        teacher_path = Path(os.getenv("TEACHER_CKPT", "/data/checkpoints/plain/games-15m/l12_best.pt"))
        teacher_state = torch.load(teacher_path, map_location="cpu", weights_only=False)
        teacher_config = GPTConfig(**teacher_state["meta"]["model_config"])
        if teacher_config.vocab_size != meta["vocab_size"]:
            raise ValueError("teacher and dataset vocabularies differ")
        if teacher_state["meta"]["tokenizer"]["stoi"] != meta["stoi"]:
            raise ValueError("teacher and dataset token IDs differ")
        teacher = GPT(teacher_config).to(device).eval()
        teacher.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in teacher_state["model"].items()})
        del teacher_state
        for p in teacher.parameters():
            p.requires_grad_(False)
        sequence_len = teacher_config.sequence_len

    config = GPTConfig(
        sequence_len=sequence_len,
        vocab_size=meta["vocab_size"],
        n_layer=env_int("STUDENT_LAYERS", 6),
        n_head=env_int("STUDENT_HEADS", 6),
        n_kv_head=env_int("STUDENT_KV_HEADS", 6),
        n_embd=env_int("STUDENT_WIDTH", 384),
    )
    qat = bool(env_int("QAT", 0))
    student = StudentGPT(config, qat=qat).to(device)
    parameters = sum(p.numel() for p in student.parameters())
    print(f"teacher={teacher_path} student={config.n_layer}x{config.n_embd} heads={config.n_head} "
          f"params={parameters:,} qat={qat} train_tokens={len(train_tokens):,} val_tokens={len(val_tokens):,}")

    ckpt_dir = Path(os.getenv("CKPT_DIR", "/data/checkpoints/distill/student-l6e384"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "student_latest.pt"
    best_path = ckpt_dir / "student_best.pt"
    init_from = os.getenv("INIT_FROM")
    if init_from and latest_path.exists():
        raise ValueError("INIT_FROM conflicts with existing latest checkpoint in CKPT_DIR")
    if init_from:
        state = torch.load(init_from, map_location="cpu", weights_only=False)
        if state["meta"]["model_config"] != config.__dict__:
            raise ValueError("INIT_FROM student configuration differs")
        student.load_state_dict(state["model"])
        print(f"initialized from {init_from} at prior step {state['step']}")
        del state

    lr = float(os.getenv("LR", "3e-4"))
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr,
                                  weight_decay=float(os.getenv("WEIGHT_DECAY", "0.1")),
                                  fused=device.type == "cuda")
    train_rng = torch.Generator().manual_seed(seed)
    val_rng = torch.Generator().manual_seed(env_int("VAL_SEED", 314159))
    block_size = env_int("BLOCK_SIZE", min(128, config.sequence_len))
    if block_size > config.sequence_len:
        raise ValueError("BLOCK_SIZE exceeds model sequence length")
    batch_size = env_int("BATCH_SIZE", 4)
    grad_accum = env_int("GRAD_ACCUM_STEPS", 8)
    val_batches = env_int("VAL_BATCHES", 4)
    val_end = int(len(val_tokens) * float(os.getenv("VAL_FRACTION", "0.5")))
    fixed_val = [sample_batch(val_tokens, block_size, batch_size, val_rng, val_end)
                 for _ in range(val_batches)]
    max_steps = env_int("MAX_STEPS", 10000)
    schedule = {
        "name": os.getenv("LR_SCHEDULE", "constant"),
        "peak_lr": lr,
        "min_lr": float(os.getenv("MIN_LR", str(lr * 0.1))),
        "warmup_steps": env_int("WARMUP_STEPS", 0),
        "decay_steps": env_int("DECAY_STEPS", max_steps),
    }
    if schedule["name"] not in ("constant", "cosine"):
        raise ValueError("LR_SCHEDULE must be constant or cosine")
    if schedule["name"] == "cosine" and not (0 <= schedule["warmup_steps"] < schedule["decay_steps"]):
        raise ValueError("cosine schedule requires 0 <= WARMUP_STEPS < DECAY_STEPS")
    if not 0 <= schedule["min_lr"] <= lr:
        raise ValueError("MIN_LR must be between zero and LR")
    max_seconds = float(os.getenv("MAX_SECONDS", "0"))
    eval_interval = env_int("EVAL_INTERVAL", 100)
    ckpt_interval = env_int("CKPT_INTERVAL", 500)
    start_step, best_metric, best_step = 1, float("inf"), 0
    if latest_path.exists():
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        if state["meta"]["model_config"] != config.__dict__ or state["meta"]["qat"] != qat:
            raise ValueError("latest checkpoint architecture or QAT mode differs")
        if state["meta"].get("distill_alpha", 0.5) != alpha:
            raise ValueError("latest checkpoint training objective differs")
        student.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if device.type == "cuda":
            for group in optimizer.param_groups:
                group["fused"] = True
        schedule_keys = ("LR_SCHEDULE", "LR", "MIN_LR", "WARMUP_STEPS", "DECAY_STEPS")
        if "schedule" in state and not any(key in os.environ for key in schedule_keys):
            schedule = state["schedule"]
        train_rng.set_state(state["train_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and "cuda_rng" in state:
            torch.cuda.set_rng_state(state["cuda_rng"])
        start_step = state["step"] + 1
        best_metric = state["best_metric"]
        best_step = state["best_step"]
        print(f"resumed {latest_path} at step {start_step}")
        del state

    print(f"run: block={block_size} batch={batch_size} accum={grad_accum} lr_schedule={schedule} "
          f"alpha={alpha:g} temperature={temperature:g} max_steps={max_steps} "
          f"max_seconds={max_seconds:g} eval_interval={eval_interval} val_batches={val_batches} "
          f"fixed_val_first_tokens={val_end:,}")
    amp = device.type == "cuda"

    def checkpoint(step, include_optimizer):
        result = {
            "model": student.state_dict(),
            "meta": {"model_type": "student", "model_family": "student_tied", "model_config": config.__dict__,
                     "tokenizer": meta, "qat": qat, "quantization": "symmetric_int4_group64",
                     "distill_alpha": alpha, "temperature": temperature},
            "step": step,
            "best_metric": best_metric,
            "best_step": best_step,
            "schedule": schedule,
        }
        if include_optimizer:
            result.update(optimizer=optimizer.state_dict(), train_rng=train_rng.get_state(),
                          torch_rng=torch.get_rng_state())
            if device.type == "cuda":
                result["cuda_rng"] = torch.cuda.get_rng_state()
        return result

    def save(path, step, include_optimizer):
        temp_path = path.with_suffix(".tmp")
        torch.save(checkpoint(step, include_optimizer), temp_path)
        os.replace(temp_path, path)

    def validate(step):
        nonlocal best_metric, best_step
        student.eval()
        ce_sum, kl_sum = 0.0, 0.0
        with torch.inference_mode():
            for x_cpu, y_cpu in fixed_val:
                x = x_cpu.to(device, non_blocking=True)
                y = y_cpu.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
                    teacher_logits = teacher(x) if teacher is not None else None
                    student_logits = student(x)
                ce, kl = loss_terms(student_logits, teacher_logits, y, temperature)
                ce_sum += ce.item()
                kl_sum += kl.item()
        ce_mean, kl_mean = ce_sum / val_batches, kl_sum / val_batches
        metric = (1 - alpha) * ce_mean + alpha * kl_mean
        if metric < best_metric:
            best_metric, best_step = metric, step
            save(best_path, step, False)
            print(f"new best student_best.pt at step={step} val_loss={metric:.4f}")
        print(f"validation step={step} loss={metric:.4f} ce={ce_mean:.4f} "
              f"kl={kl_mean:.4f} best={best_metric:.4f}@{best_step}")
        student.train()

    student.train()
    run_start = time.monotonic()
    last_step = start_step - 1
    for step in range(start_step, max_steps + 1):
        step_lr = learning_rate(step, schedule)
        for group in optimizer.param_groups:
            group["lr"] = step_lr
        optimizer.zero_grad(set_to_none=True)
        train_loss = torch.zeros((), device=device)
        for _ in range(grad_accum):
            x_cpu, y_cpu = sample_batch(train_tokens, block_size, batch_size, train_rng)
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
                teacher_logits = teacher(x) if teacher is not None else None
            with torch.amp.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
                student_logits = student(x)
                ce, kl = loss_terms(student_logits, teacher_logits, y, temperature)
                loss = (1 - alpha) * ce + alpha * kl
            (loss / grad_accum).backward()
            train_loss += loss.detach() / grad_accum
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        last_step = step
        train_loss_value = train_loss.item()
        elapsed = time.monotonic() - run_start
        if step == start_step or step % 10 == 0:
            done = step - start_step + 1
            tokens_per_sec = done * batch_size * block_size * grad_accum / max(elapsed, 1e-6)
            print(f"step={step} train_loss={train_loss_value:.4f} lr={step_lr:.3g} elapsed={elapsed:.1f}s "
                  f"tokens_per_sec={tokens_per_sec:.0f}")
        timed_out = max_seconds > 0 and elapsed >= max_seconds
        if step == start_step or step % eval_interval == 0 or step == max_steps or timed_out:
            validate(step)
        if step % ckpt_interval == 0 or step == max_steps or timed_out:
            save(latest_path, step, True)
            print(f"saved student_latest.pt at step={step}")
        if timed_out:
            print(f"stopped at MAX_SECONDS after step={step}")
            break
    if last_step == start_step - 1:
        print(f"no steps taken; latest={latest_path if latest_path.exists() else 'absent'}")


if __name__ == "__main__":
    main()
