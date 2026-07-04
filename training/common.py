import os

import torch

from nanochat.gpt import GPTConfig, norm

CONFIG_KEYS = ("sequence_len", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd")
SOFTCAP = 15


def env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def config_from_env(context_length, vocab_size):
    n_head = int(os.getenv("N_HEAD", "6"))
    return GPTConfig(
        sequence_len=context_length,
        vocab_size=vocab_size,
        n_layer=int(os.getenv("N_LAYER", "12")),
        n_head=n_head,
        n_kv_head=int(os.getenv("N_KV_HEAD", str(n_head))),
        n_embd=int(os.getenv("N_EMBD", "768")),
    )


def arch_name(config):
    if config.n_kv_head == config.n_head:
        return f"L{config.n_layer}_H{config.n_head}_E{config.n_embd}"
    return f"L{config.n_layer}_H{config.n_head}_KV{config.n_kv_head}_E{config.n_embd}"


def config_mismatches(saved_config, config):
    current_config = config.__dict__
    return {
        key: (saved_config.get(key), current_config[key])
        for key in CONFIG_KEYS
        if saved_config.get(key) != current_config[key]
    }


def load_checkpoint(ckpt_path, config, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    mismatches = config_mismatches(state.get("meta", {}).get("model_config", {}), config)
    if mismatches:
        raise ValueError(f"checkpoint config mismatch in {ckpt_path}: {mismatches}")
    return state


def load_model_state(model, state):
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state["model"].items()})


def clean_state_dict(module):
    return {k.replace("_orig_mod.", ""): v for k, v in module.state_dict().items()}


def build_checkpoint(model, config, meta, step, **extra):
    ckpt = {
        "model": clean_state_dict(model),
        "meta": {"model_config": config.__dict__, "tokenizer": meta},
        "step": step,
    }
    ckpt.update(extra)
    return ckpt


def sample_batch(tokens, block_size, batch_size, shifted=(), aligned=()):
    max_start = tokens.size(0) - block_size - 1
    idx = torch.randint(0, max_start, (batch_size,))
    out = [
        torch.stack([tokens[i : i + block_size] for i in idx]),
        torch.stack([tokens[i + 1 : i + block_size + 1] for i in idx]),
    ]
    out += [torch.stack([a[i + 1 : i + block_size + 1] for i in idx]) for a in shifted]
    out += [torch.stack([a[i : i + block_size] for i in idx]) for a in aligned]
    return out


def train_loop(model, optimizer, config, meta, device, ckpt_path, best_ckpt_path,
               sample_train, sample_val, train_step, val_step, metric_key, max_iters,
               grad_accum_steps=1):
    eval_interval = int(os.getenv("EVAL_INTERVAL", "100"))
    ckpt_interval = int(os.getenv("CKPT_INTERVAL", "1000"))
    val_batches = int(os.getenv("VAL_BATCHES", "1"))
    patience = int(os.getenv("EARLY_STOP_PATIENCE", "0"))
    min_steps = int(os.getenv("EARLY_STOP_MIN_STEPS", "0"))
    min_delta = float(os.getenv("EARLY_STOP_MIN_DELTA", "0.0"))
    print(
        f"run config: ckpt_name={ckpt_path.name} max_iters={max_iters} eval_interval={eval_interval} "
        f"ckpt_interval={ckpt_interval} grad_accum_steps={grad_accum_steps} val_batches={val_batches} "
        f"metric={metric_key} early_stop_patience={patience} early_stop_min_steps={min_steps} "
        f"early_stop_min_delta={min_delta}"
    )

    start_step = 1
    best_metric = float("inf")
    best_step = 0
    no_improve = 0
    if ckpt_path.exists():
        state = load_checkpoint(ckpt_path, config, device)
        load_model_state(model, state)
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"] + 1
        best_metric = state.get("best_metric", float("inf"))
        best_step = state.get("best_step", 0)
        no_improve = state.get("no_improve_evals", 0)
        print(
            f"Loaded checkpoint from {ckpt_path} at step {start_step} "
            f"(best_metric={best_metric:.4f}, best_step={best_step}, no_improve_evals={no_improve})"
        )
    else:
        print("No checkpoint found, starting fresh")

    compiled = torch.compile(model) if device.type == "cuda" else model
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def to_device(batch):
        return [t.to(device, non_blocking=True) for t in batch]

    def save(path, step):
        checkpoint = build_checkpoint(
            compiled, config, meta, step,
            optimizer=optimizer.state_dict(),
            best_metric=best_metric,
            best_step=best_step,
            no_improve_evals=no_improve,
        )
        torch.save(checkpoint, path)

    for step in range(start_step, max_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        for _ in range(grad_accum_steps):
            batch = to_device(sample_train())
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                loss = train_step(compiled, *batch)
            scaler.scale(loss / grad_accum_steps).backward()
            train_loss_sum += loss.item()
        train_loss = train_loss_sum / grad_accum_steps
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        should_stop = False
        if step % eval_interval == 0 or step == 1:
            sums = {}
            for _ in range(val_batches):
                batch = to_device(sample_val())
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                    for key, value in val_step(compiled, *batch).items():
                        sums[key] = sums.get(key, 0.0) + value
            metrics = {key: value / val_batches for key, value in sums.items()}
            metric_value = metrics[metric_key]
            if metric_value < (best_metric - min_delta):
                best_metric = metric_value
                best_step = step
                no_improve = 0
                save(best_ckpt_path, step)
                print(f"New best {metric_key} {best_metric:.4f} at step {step}, saved {best_ckpt_path.name}")
            elif patience > 0 and step >= min_steps:
                no_improve += 1

            metric_str = " ".join(f"{key} {value:.4f}" for key, value in metrics.items())
            print(
                f"step {step:04d}/{max_iters} train {train_loss:.4f} {metric_str} "
                f"best {best_metric:.4f}@{best_step} no_improve {no_improve}"
            )
            if patience > 0 and step >= min_steps and no_improve >= patience:
                should_stop = True

        if step % ckpt_interval == 0 or should_stop:
            save(ckpt_path, step)
            print(f"Saved checkpoint to {ckpt_path} at step {step}")

        if should_stop:
            print(f"Early stopping at step {step}: best={best_metric:.4f} from step {best_step}")
            break


def forward_hidden(model, xb):
    T = xb.size(1)
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    x = norm(model.transformer.wte(xb))
    for block in model.transformer.h:
        x = block(x, cos_sin, None)
    return norm(x)


def move_logits(model, x):
    logits = model.lm_head(x)
    return (SOFTCAP * torch.tanh(logits / SOFTCAP)).float()
