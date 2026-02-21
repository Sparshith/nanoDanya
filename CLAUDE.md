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
