from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_registry import resolve_model_ref
from nanochat.gpt import GPT, GPTConfig


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_data_dir(env_name: str, default: str) -> Path:
    raw = os.getenv(env_name, default).strip() or default
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / "data" / path


def load_meta(data_dir: Path) -> dict[str, Any]:
    return pickle.loads((data_dir / "meta.pkl").read_bytes())


def tokenizer_fingerprint(meta: dict[str, Any]) -> tuple[int, int, str, str]:
    stoi = meta["stoi"]
    return (
        int(meta["vocab_size"]),
        int(meta["context_length"]),
        str(stoi["<bos>"]),
        str(stoi["<eos>"]),
    )


def load_hard_split(data_dir: Path, split: str, context_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefixes_np = np.fromfile(data_dir / f"{split}_hard_prefixes.bin", dtype=np.uint16)
    lengths_np = np.fromfile(data_dir / f"{split}_hard_lengths.bin", dtype=np.uint16)
    targets_np = np.fromfile(data_dir / f"{split}_hard_targets.bin", dtype=np.uint16)

    if len(lengths_np) != len(targets_np):
        raise ValueError(f"{split}: lengths/targets mismatch: {len(lengths_np)} vs {len(targets_np)}")
    if len(targets_np) == 0:
        raise ValueError(f"{split}: no hard examples found in {data_dir}")
    if prefixes_np.size != len(targets_np) * context_length:
        raise ValueError(
            f"{split}: prefix size mismatch: {prefixes_np.size} values for "
            f"{len(targets_np)} examples x context_length {context_length}"
        )

    prefixes_np = prefixes_np.reshape(len(targets_np), context_length)
    # PyTorch does not implement CPU advanced indexing for uint16 on the Modal
    # image. Keep the resident CPU copy compact enough with int32, then cast
    # sampled batches to long on the target device.
    prefixes = torch.from_numpy(prefixes_np.astype(np.int32))
    lengths = torch.from_numpy(lengths_np.astype(np.int32))
    targets = torch.from_numpy(targets_np.astype(np.int32))
    return prefixes, lengths, targets


def sample_lm_batch(tokens: torch.Tensor, block_size: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = tokens.size(0) - block_size - 1
    if max_start <= 0:
        raise ValueError(f"not enough tokens for block_size={block_size}: {tokens.size(0)}")
    idx = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in idx])
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in idx])
    return x, y


def sample_hard_batch(
    prefixes: torch.Tensor,
    lengths: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, prefixes.size(0), (batch_size,))
    return prefixes[idx], lengths[idx], targets[idx]


def hard_loss_from_batch(
    model: GPT,
    x: torch.Tensor,
    lengths: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    logits = model(x)
    row = torch.arange(x.size(0), device=x.device)
    last_pos = torch.clamp(lengths.to(x.device) - 1, min=0)
    next_logits = logits[row, last_pos, :]
    return F.cross_entropy(next_logits.float(), targets.to(x.device))


def build_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    config: GPTConfig,
    tokenizer_meta: dict[str, Any],
    *,
    step: int,
    best_metric: float,
    best_step: int,
    no_improve_evals: int,
    extra_meta: dict[str, Any],
) -> dict[str, Any]:
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    return {
        "model": clean_sd,
        "meta": {
            "model_config": config.__dict__,
            "tokenizer": tokenizer_meta,
            "king_safety_sft": extra_meta,
        },
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_metric": best_metric,
        "best_step": best_step,
        "no_improve_evals": no_improve_evals,
    }


def resolve_init_checkpoint() -> Path:
    explicit = os.getenv("INIT_CKPT_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser()

    init_model_ref = os.getenv("INIT_MODEL_REF", "plain/games-5m").strip()
    path, spec = resolve_model_ref(init_model_ref)
    candidates = [Path(path)]
    if spec is not None:
        candidates.extend(
            [
                Path("/data") / "actual_5m" / "chess_actual_5m_uniform_L12_H6_E768_best.pt",
                Path("/data") / spec.filename,
                Path("/data") / spec.dataset / spec.filename,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> None:
    normal_data_dir = resolve_data_dir("NORMAL_DATASET_DIR", "actual_5m")
    hard_data_dir = resolve_data_dir("HARD_DATASET_DIR", "king_safety_sft")

    normal_meta = load_meta(normal_data_dir)
    hard_meta = load_meta(hard_data_dir)
    if tokenizer_fingerprint(normal_meta) != tokenizer_fingerprint(hard_meta):
        raise ValueError("normal and hard datasets do not share the same tokenizer/context contract")

    vocab_size = int(normal_meta["vocab_size"])
    context_length = int(normal_meta["context_length"])

    train_tokens = torch.from_numpy(np.fromfile(normal_data_dir / "train.bin", dtype=np.uint16))
    val_tokens = torch.from_numpy(np.fromfile(normal_data_dir / "val.bin", dtype=np.uint16))
    hard_train_x, hard_train_lengths, hard_train_targets = load_hard_split(hard_data_dir, "train", context_length)
    hard_val_x, hard_val_lengths, hard_val_targets = load_hard_split(hard_data_dir, "val", context_length)

    print(f"normal data: {normal_data_dir}")
    print(f"hard data:   {hard_data_dir}")
    print(f"normal train={train_tokens.numel():,} val={val_tokens.numel():,} tokens")
    print(f"hard train={hard_train_targets.numel():,} val={hard_val_targets.numel():,} examples")

    ckpt_name = os.getenv(
        "CKPT_NAME",
        "chess_actual_5m_king_safety_sft_70_30_L12_H6_E768.pt",
    )
    default_ckpt_path = Path("/data") / "king_safety_sft" / ckpt_name
    if not Path("/data").exists():
        default_ckpt_path = PROJECT_ROOT / "models" / ckpt_name
    ckpt_path = Path(os.getenv("CKPT_PATH", str(default_ckpt_path))).expanduser()
    best_ckpt_path = ckpt_path.with_name(ckpt_path.stem + "_best" + ckpt_path.suffix)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    max_iters = int(os.getenv("MAX_ITERS", "3000"))
    eval_interval = int(os.getenv("EVAL_INTERVAL", "100"))
    ckpt_interval = int(os.getenv("CKPT_INTERVAL", "500"))
    batch_size = int(os.getenv("BATCH_SIZE", "64"))
    val_batches = int(os.getenv("VAL_BATCHES", "8"))
    hard_fraction = float(os.getenv("HARD_FRACTION", "0.30"))
    lr = float(os.getenv("LR", "1e-5"))
    weight_decay = float(os.getenv("WEIGHT_DECAY", "0.0"))
    early_stop_patience = int(os.getenv("EARLY_STOP_PATIENCE", "0"))
    early_stop_min_steps = int(os.getenv("EARLY_STOP_MIN_STEPS", "0"))
    early_stop_min_delta = float(os.getenv("EARLY_STOP_MIN_DELTA", "0.0"))

    if not (0.0 <= hard_fraction <= 1.0):
        raise ValueError("HARD_FRACTION must be between 0 and 1")
    hard_batch = int(round(batch_size * hard_fraction))
    normal_batch = batch_size - hard_batch
    if hard_batch == 0 or normal_batch == 0:
        raise ValueError("BATCH_SIZE and HARD_FRACTION must leave at least one normal and one hard example")

    print(
        f"run config: max_iters={max_iters} eval_interval={eval_interval} ckpt_interval={ckpt_interval} "
        f"batch_size={batch_size} normal_batch={normal_batch} hard_batch={hard_batch} "
        f"lr={lr} val_batches={val_batches}"
    )

    if env_flag("DRY_RUN", False):
        xb_norm_cpu, yb_norm_cpu = sample_lm_batch(train_tokens, context_length, normal_batch)
        xb_hard_cpu, len_hard_cpu, y_hard_cpu = sample_hard_batch(
            hard_train_x,
            hard_train_lengths,
            hard_train_targets,
            hard_batch,
        )
        print(
            "dry run: "
            f"normal_x={tuple(xb_norm_cpu.shape)} normal_y={tuple(yb_norm_cpu.shape)} "
            f"hard_x={tuple(xb_hard_cpu.shape)} hard_lengths={tuple(len_hard_cpu.shape)} "
            f"hard_targets={tuple(y_hard_cpu.shape)}"
        )
        return

    if not env_flag("NANODANYA_MODAL_TRAIN", False) and not env_flag("ALLOW_LOCAL_TRAINING", False):
        raise RuntimeError(
            "Refusing to train outside the Modal training entrypoint. "
            "Use `uv run modal run modal_train.py --datasets actual_5m,king_safety_sft "
            "--script sft/train_king_safety_sft.py ...`, or set ALLOW_LOCAL_TRAINING=1 for an explicit override."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = GPTConfig(
        sequence_len=context_length,
        vocab_size=vocab_size,
        n_layer=int(os.getenv("N_LAYER", "12")),
        n_head=int(os.getenv("N_HEAD", "6")),
        n_kv_head=int(os.getenv("N_KV_HEAD", os.getenv("N_HEAD", "6"))),
        n_embd=int(os.getenv("N_EMBD", "768")),
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    start_step = 1
    best_metric = float("inf")
    best_step = 0
    no_improve_evals = 0
    init_path = resolve_init_checkpoint()

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_config = state.get("meta", {}).get("model_config", {})
        if saved_config.get("n_layer") != config.n_layer or saved_config.get("n_embd") != config.n_embd:
            raise ValueError(f"checkpoint config mismatch in {ckpt_path}")
        sd = {k.replace("_orig_mod.", ""): v for k, v in state["model"].items()}
        model.load_state_dict(sd)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1
        best_metric = float(state.get("best_metric", float("inf")))
        best_step = int(state.get("best_step", 0))
        no_improve_evals = int(state.get("no_improve_evals", 0))
        print(f"resumed from {ckpt_path} at step {start_step}")
    else:
        if not init_path.exists():
            raise FileNotFoundError(f"initial checkpoint not found: {init_path}")
        state = torch.load(init_path, map_location=device, weights_only=False)
        saved_config = state.get("meta", {}).get("model_config", {})
        if saved_config.get("n_layer") != config.n_layer or saved_config.get("n_embd") != config.n_embd:
            raise ValueError(f"initial checkpoint config mismatch in {init_path}")
        sd = {k.replace("_orig_mod.", ""): v for k, v in state["model"].items()}
        model.load_state_dict(sd)
        print(f"initialized from {init_path}")

    if device.type == "cuda" and env_flag("COMPILE", True):
        model = torch.compile(model)

    train_tokens = train_tokens.to("cpu")
    val_tokens = val_tokens.to("cpu")
    hard_train_x = hard_train_x.to("cpu")
    hard_train_lengths = hard_train_lengths.to("cpu")
    hard_train_targets = hard_train_targets.to("cpu")
    hard_val_x = hard_val_x.to("cpu")
    hard_val_lengths = hard_val_lengths.to("cpu")
    hard_val_targets = hard_val_targets.to("cpu")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    extra_meta = {
        "normal_data_dir": str(normal_data_dir),
        "hard_data_dir": str(hard_data_dir),
        "init_checkpoint": str(init_path),
        "hard_fraction": hard_fraction,
        "normal_batch": normal_batch,
        "hard_batch": hard_batch,
    }

    for step in range(start_step, max_iters + 1):
        xb_norm_cpu, yb_norm_cpu = sample_lm_batch(train_tokens, context_length, normal_batch)
        xb_hard_cpu, len_hard_cpu, y_hard_cpu = sample_hard_batch(
            hard_train_x,
            hard_train_lengths,
            hard_train_targets,
            hard_batch,
        )

        xb_norm = xb_norm_cpu.to(device, dtype=torch.long, non_blocking=True)
        yb_norm = yb_norm_cpu.to(device, dtype=torch.long, non_blocking=True)
        xb_hard = xb_hard_cpu.to(device, dtype=torch.long, non_blocking=True)
        len_hard = len_hard_cpu.to(device, dtype=torch.long, non_blocking=True)
        y_hard = y_hard_cpu.to(device, dtype=torch.long, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            logits_norm = model(xb_norm)
            normal_loss = F.cross_entropy(logits_norm.view(-1, logits_norm.size(-1)).float(), yb_norm.view(-1))
            hard_loss = hard_loss_from_batch(model, xb_hard, len_hard, y_hard)
            loss = (1.0 - hard_fraction) * normal_loss + hard_fraction * hard_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        should_stop = False
        if step % eval_interval == 0 or step == 1:
            val_normal_sum = 0.0
            val_hard_sum = 0.0
            for _ in range(val_batches):
                xb_val_cpu, yb_val_cpu = sample_lm_batch(val_tokens, context_length, normal_batch)
                xb_hval_cpu, len_hval_cpu, y_hval_cpu = sample_hard_batch(
                    hard_val_x,
                    hard_val_lengths,
                    hard_val_targets,
                    hard_batch,
                )
                xb_val = xb_val_cpu.to(device, dtype=torch.long, non_blocking=True)
                yb_val = yb_val_cpu.to(device, dtype=torch.long, non_blocking=True)
                xb_hval = xb_hval_cpu.to(device, dtype=torch.long, non_blocking=True)
                len_hval = len_hval_cpu.to(device, dtype=torch.long, non_blocking=True)
                y_hval = y_hval_cpu.to(device, dtype=torch.long, non_blocking=True)

                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                    logits_val = model(xb_val)
                    val_normal = F.cross_entropy(
                        logits_val.view(-1, logits_val.size(-1)).float(),
                        yb_val.view(-1),
                    )
                    val_hard = hard_loss_from_batch(model, xb_hval, len_hval, y_hval)
                val_normal_sum += float(val_normal.item())
                val_hard_sum += float(val_hard.item())

            val_normal_loss = val_normal_sum / val_batches
            val_hard_loss = val_hard_sum / val_batches
            val_mixed_loss = (1.0 - hard_fraction) * val_normal_loss + hard_fraction * val_hard_loss
            improved = val_mixed_loss < (best_metric - early_stop_min_delta)
            if improved:
                best_metric = val_mixed_loss
                best_step = step
                no_improve_evals = 0
                checkpoint = build_checkpoint(
                    model,
                    optimizer,
                    config,
                    normal_meta,
                    step=step,
                    best_metric=best_metric,
                    best_step=best_step,
                    no_improve_evals=no_improve_evals,
                    extra_meta=extra_meta,
                )
                torch.save(checkpoint, best_ckpt_path)
                print(f"new best mixed val {best_metric:.4f} at step {step}, saved {best_ckpt_path.name}")
            elif early_stop_patience > 0 and step >= early_stop_min_steps:
                no_improve_evals += 1

            print(
                f"step {step:04d}/{max_iters} train {loss.item():.4f} "
                f"normal {normal_loss.item():.4f} hard {hard_loss.item():.4f} "
                f"val_mix {val_mixed_loss:.4f} val_normal {val_normal_loss:.4f} "
                f"val_hard {val_hard_loss:.4f} best {best_metric:.4f}@{best_step} "
                f"no_improve {no_improve_evals}"
            )

            if early_stop_patience > 0 and step >= early_stop_min_steps and no_improve_evals >= early_stop_patience:
                should_stop = True

        if step % ckpt_interval == 0 or should_stop:
            checkpoint = build_checkpoint(
                model,
                optimizer,
                config,
                normal_meta,
                step=step,
                best_metric=best_metric,
                best_step=best_step,
                no_improve_evals=no_improve_evals,
                extra_meta=extra_meta,
            )
            torch.save(checkpoint, ckpt_path)
            print(f"saved checkpoint to {ckpt_path} at step {step}")

        if should_stop:
            print(f"early stopping at step {step}: best={best_metric:.4f} from step {best_step}")
            break


if __name__ == "__main__":
    main()
