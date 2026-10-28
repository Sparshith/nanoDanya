from pathlib import Path

import modal

from chess_inference import choose_move_from_logits
from chess_token_utils import resolve_token_id

app = modal.App("nanodanya-chess")

SERVE_MODEL_REF = "plain/whole-game-20261016/l16_final"
SERVE_MODEL_PATH = "/data/checkpoints/plain/whole-game-20261016/l16_final.pt"
SERVE_MODEL_SHA256 = "6b8a0c4dbe5ce5fc5f9d7f1f8b08445fb1b5046488a0ed23e04bd8e3621379ff"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1+cu124", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("chess==1.11.2", "fastapi[standard]==0.115.12")
    .add_local_python_source("chess_inference", "chess_token_utils")
)

for name in ["__init__.py", "gpt.py", "common.py", "muon.py", "adamw.py"]:
    image = image.add_local_file(Path(__file__).resolve().parents[1] / "nanochat/nanochat" / name,
                                 "/root/nanochat/" + name)

volume = modal.Volume.from_name("nanodanya-data")


@app.cls(
    image=image,
    gpu="T4",
    max_containers=1,
    scaledown_window=300,
    volumes={"/data": volume},
)
@modal.concurrent(max_inputs=10)
class ChessModel:
    @modal.enter()
    def load_model(self):
        import hashlib
        import torch
        from nanochat.gpt import GPT, GPTConfig

        torch.set_num_threads(1)
        digest = hashlib.sha256()
        with open(SERVE_MODEL_PATH, "rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == SERVE_MODEL_SHA256
        ckpt = torch.load(SERVE_MODEL_PATH, map_location="cpu", weights_only=False)
        assert ckpt["step"] == 250000
        config = GPTConfig(**ckpt["meta"]["model_config"])
        self.model = GPT(config).eval().cuda()
        self.model.load_state_dict(ckpt["model"])
        self.stoi = ckpt["meta"]["tokenizer"]["stoi"]
        self.itos = ckpt["meta"]["tokenizer"]["itos"]
        self.config = config
        print(f"Loaded {SERVE_MODEL_REF}, checkpoint step {ckpt.get('step')}")

    @modal.method()
    def get_move(self, moves: list[str], temperature: float = 1.0) -> str:
        import torch
        import chess

        board = chess.Board()
        tokens = ["<bos>"]

        for move in moves:
            parsed = board.parse_san(move)
            tokens.append(board.san(parsed))
            board.push(parsed)

        token_ids = []
        for token in tokens:
            token_id = resolve_token_id(self.stoi, token)
            if token_id is None:
                raise ValueError(f"Move is missing from the tokenizer: {token}")
            token_ids.append(token_id)
        if not token_ids:
            token_ids = [self.stoi["<bos>"]]

        x = torch.tensor(token_ids, device="cuda")[None, :]
        with torch.inference_mode():
            logits = self.model(x[:, -self.config.sequence_len:])

        _, next_token = choose_move_from_logits(
            logits[0, -1, :],
            board,
            self.stoi,
            self.itos,
            temperature=temperature,
            allow_eos=False,
            legal_mask=True,
        )
        if next_token is None:
            raise ValueError("no legal move available")
        return next_token


@app.function(image=image, max_containers=1)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def serve():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import chess

    web_app = FastAPI(title="nanoDanya Chess API")

    class MoveRequest(BaseModel):
        moves: list[str] = []
        fen: str | None = None
        temperature: float = 1.0

    class MoveResponse(BaseModel):
        move: str
        moves_played: list[str]

    @web_app.post("/move", response_model=MoveResponse)
    async def get_move(request: MoveRequest):
        model = ChessModel()

        moves = list(request.moves)
        if request.fen:
            board = chess.Board(request.fen)
            moves = []
            temp_board = chess.Board()
            for move in board.move_stack:
                moves.append(temp_board.san(move))
                temp_board.push(move)

        try:
            next_move = await model.get_move.remote.aio(moves, request.temperature)
            return MoveResponse(move=next_move, moves_played=moves + [next_move])
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @web_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_ref": SERVE_MODEL_REF,
            "model_path": SERVE_MODEL_PATH,
            "checkpoint_sha256": SERVE_MODEL_SHA256,
            "checkpoint_step": 250000,
        }

    return web_app


@app.local_entrypoint()
def check():
    import asyncio
    import json
    import time

    records = Path(__file__).resolve().parents[1] / "output/whole-game-20261016/benchmark/whole_game-h2h-0.jsonl"
    games = [json.loads(line) for line in records.read_text().splitlines()][1:-1]
    positions = [[], ["e4", "d6", "Bb5"]]
    positions += [game["moves"][:min(120, len(game["moves"]) - 1)] for game in games[:8]]
    model = ChessModel()

    async def verify():
        started = time.monotonic()
        move = await model.get_move.remote.aio([], .2)
        cold_seconds = time.monotonic() - started
        started = time.monotonic()
        moves = await asyncio.gather(*(model.get_move.remote.aio(history, .2) for history in positions))
        return dict(model_ref=SERVE_MODEL_REF, checkpoint_sha256=SERVE_MODEL_SHA256,
                    cold_seconds=cold_seconds, concurrent_requests=len(positions),
                    concurrent_seconds=time.monotonic() - started,
                    warmup_move=move, positions=positions, moves=moves)

    result = asyncio.run(verify())
    output = Path(__file__).resolve().parents[1] / "output/lichess-whole-game-20261019/serving-preflight.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
