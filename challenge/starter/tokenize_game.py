# Turns the moves of one game into token ids with vocabulary.json.
# Usage: python3 tokenize_game.py "1. e4 e5 2. Nf3 Nc6 3. Bb5"
import json
import sys
from pathlib import Path

vocabulary = json.loads((Path(__file__).parent / "vocabulary.json").read_text())
ids = {token: i for i, token in enumerate(vocabulary)}
moves = [m for m in sys.argv[1].split() if not m.endswith(".")]
tokens = [ids["<bos>"]] + [ids[m] for m in moves]
print(tokens)
