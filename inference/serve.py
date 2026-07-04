import modal

from chess_inference import choose_move_from_logits
from chess_token_utils import resolve_token_id

app = modal.App("nanodanya-chess")

SERVE_MODEL_REF = "plain/games-15m"
SERVE_MODEL_PATH = "/data/checkpoints/plain/games-15m/l16_best.pt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "chess", "fastapi[standard]")
    .add_local_python_source("chess_inference", "chess_token_utils")
)

volume = modal.Volume.from_name("nanodanya-data")


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/data": volume},
)
class ChessModel:
    @modal.enter()
    def load_model(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from dataclasses import dataclass

        @dataclass
        class GPTConfig:
            sequence_len: int = 1024
            vocab_size: int = 50304
            n_layer: int = 12
            n_head: int = 6
            n_kv_head: int = 6
            n_embd: int = 768

        def norm(x):
            return F.rms_norm(x, (x.size(-1),))

        def apply_rotary_emb(x, cos, sin):
            d = x.shape[3] // 2
            x1, x2 = x[..., :d], x[..., d:]
            y1 = x1 * cos + x2 * sin
            y2 = x1 * (-sin) + x2 * cos
            return torch.cat([y1, y2], 3).to(x.dtype)

        class CausalSelfAttention(nn.Module):
            def __init__(self, config, layer_idx):
                super().__init__()
                self.layer_idx = layer_idx
                self.n_head = config.n_head
                self.n_kv_head = config.n_kv_head
                self.n_embd = config.n_embd
                self.head_dim = self.n_embd // self.n_head
                self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
                self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
                self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
                self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

            def forward(self, x, cos_sin, kv_cache=None):
                B, T, C = x.size()
                q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
                k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
                v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
                cos, sin = cos_sin
                q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
                q, k = norm(q), norm(k)
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                enable_gqa = self.n_head != self.n_kv_head
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
                y = y.transpose(1, 2).contiguous().view(B, T, -1)
                return self.c_proj(y)

        class MLP(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
                self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

            def forward(self, x):
                x = self.c_fc(x)
                x = F.relu(x).square()
                return self.c_proj(x)

        class Block(nn.Module):
            def __init__(self, config, layer_idx):
                super().__init__()
                self.attn = CausalSelfAttention(config, layer_idx)
                self.mlp = MLP(config)

            def forward(self, x, cos_sin, kv_cache=None):
                x = x + self.attn(norm(x), cos_sin, kv_cache)
                x = x + self.mlp(norm(x))
                return x

        class GPT(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.transformer = nn.ModuleDict({
                    "wte": nn.Embedding(config.vocab_size, config.n_embd),
                    "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
                })
                self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
                self.rotary_seq_len = config.sequence_len * 10
                head_dim = config.n_embd // config.n_head
                cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
                self.register_buffer("cos", cos, persistent=False)
                self.register_buffer("sin", sin, persistent=False)

            def _precompute_rotary(self, seq_len, head_dim, base=10000):
                device = self.transformer.wte.weight.device
                channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                inv_freq = 1.0 / (base ** (channel_range / head_dim))
                t = torch.arange(seq_len, dtype=torch.float32, device=device)
                freqs = torch.outer(t, inv_freq)
                cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
                return cos[None, :, None, :], sin[None, :, None, :]

            def forward(self, idx):
                B, T = idx.size()
                cos_sin = self.cos[:, :T], self.sin[:, :T]
                x = norm(self.transformer.wte(idx))
                for block in self.transformer.h:
                    x = block(x, cos_sin)
                x = norm(x)
                logits = self.lm_head(x)
                return 15 * torch.tanh(logits / 15)

        ckpt = torch.load(SERVE_MODEL_PATH, map_location="cpu", weights_only=False)
        config = GPTConfig(**ckpt["meta"]["model_config"])
        self.model = GPT(config).eval().cuda()
        self.model.load_state_dict(ckpt["model"])
        self.stoi = ckpt["meta"]["tokenizer"]["stoi"]
        self.itos = ckpt["meta"]["tokenizer"]["itos"]
        self.config = config

    @modal.method()
    def get_move(self, moves: list[str], temperature: float = 1.0) -> str:
        import torch
        import chess

        board = chess.Board()
        tokens = ["<bos>"]

        for move in moves:
            board.push_san(move)
            tokens.append(move)

        token_ids = []
        for token in tokens:
            token_id = resolve_token_id(self.stoi, token)
            if token_id is not None:
                token_ids.append(token_id)
        if not token_ids:
            token_ids = [self.stoi["<bos>"]]

        x = torch.tensor(token_ids, device="cuda")[None, :]
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


@app.function(image=image)
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
            next_move = model.get_move.remote(moves, request.temperature)
            return MoveResponse(move=next_move, moves_played=moves + [next_move])
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @web_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_ref": SERVE_MODEL_REF,
            "model_path": SERVE_MODEL_PATH,
        }

    return web_app
