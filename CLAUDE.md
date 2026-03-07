# nanoDanya

## Rules

- Do not modify Jupyter notebooks (.ipynb files). Use plain Python scripts instead.
- Do not modify the nanochat submodule.
- Follow coding style guidelines from the user-level CLAUDE.md (flat scripts, no unnecessary abstractions, match existing patterns).

## nanochat

nanochat is installed as an editable local dependency via `pyproject.toml`. Import directly:

```python
from nanochat.gpt import GPT, GPTConfig
```

Do not use `sys.path` hacks to import nanochat.

## Modal Training

Run training on Modal with:

```bash
uv run modal run modal_train.py --dataset <dataset_name> --script <training_script>
```

Examples:

```bash
# default (processed dataset, training/train.py)
uv run modal run modal_train.py

# weighted puzzle training
uv run modal run modal_train.py --dataset puzzle_weighted --script training/train_weighted.py
```

`--dataset` is the subdirectory name under `data/` (and the matching volume path). `--script` is the training script path relative to project root.

Do not create separate modal_train files for different training runs. Always use the same `modal_train.py` with different arguments.
