"""nanoDanya Lichess bot on Modal.

The listener is intentionally cheap: it only holds the Lichess event stream open
and calls the separate nanoDanya model endpoint when it is our turn.
"""

import modal

app = modal.App("nanodanya-lichess-bot")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "berserk", "chess"
)

NANODANYA_API_URL = "https://sparshithsampath--nanodanya-chess-serve.modal.run/move"
BOT_TIMEOUT_SECONDS = 24 * 60 * 60
RUN_SECONDS = int(23.5 * 60 * 60)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("lichess-bot-token")],
    schedule=modal.Period(hours=23),
    timeout=BOT_TIMEOUT_SECONDS,
    cpu=0.125,
    memory=256,
    max_containers=2,
)
def run_bot():
    import os
    import json
    import urllib.request
    import threading
    import time
    import chess
    import berserk

    def get_nanodanya_move(moves: list[str], temperature: float = 0.8) -> str:
        data = json.dumps({"moves": moves, "temperature": temperature}).encode()
        req = urllib.request.Request(
            NANODANYA_API_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read().decode())
        return result["move"]

    def san_to_uci(board: chess.Board, san: str) -> str:
        move = board.parse_san(san)
        return move.uci()

    def uci_to_san(board: chess.Board, uci: str) -> str:
        move = chess.Move.from_uci(uci)
        return board.san(move)

    class NanoDanyaBot:
        def __init__(self, token: str):
            session = berserk.TokenSession(token)
            self.client = berserk.Client(session)
            self.username = self.client.account.get()["username"]
            self.active_games = set()
            print(f"Logged in as: {self.username}")

        def handle_game(self, game_id: str):
            print(f"Starting game: {game_id}")
            board = chess.Board()
            san_moves = []

            for event in self.client.bots.stream_game_state(game_id):
                if event["type"] == "gameFull":
                    white = event["white"].get("name") or event["white"].get("id", "?")
                    black = event["black"].get("name") or event["black"].get("id", "?")
                    print(f"Game: {white} vs {black}")

                    we_are_white = white.lower() == self.username.lower()
                    state = event["state"]
                    moves_uci = state["moves"].split() if state["moves"] else []

                    for uci in moves_uci:
                        san = uci_to_san(board, uci)
                        san_moves.append(san)
                        board.push_san(san)

                    if self._is_our_turn(board, we_are_white):
                        self._make_move(game_id, board, san_moves)

                elif event["type"] == "gameState":
                    moves_uci = event["moves"].split() if event["moves"] else []

                    board = chess.Board()
                    san_moves = []
                    for uci in moves_uci:
                        san = uci_to_san(board, uci)
                        san_moves.append(san)
                        board.push_san(san)

                    if board.is_game_over():
                        print(f"Game over: {board.result()}")
                        return

                    we_are_white = len(moves_uci) % 2 == 0
                    if event.get("status") == "started":
                        we_are_white = self._check_color(game_id)

                    if self._is_our_turn(board, we_are_white):
                        self._make_move(game_id, board, san_moves)

        def _check_color(self, game_id: str) -> bool:
            game = self.client.games.export(game_id)
            white = game.get("players", {}).get("white", {}).get("user", {}).get("name", "")
            return white.lower() == self.username.lower()

        def _is_our_turn(self, board: chess.Board, we_are_white: bool) -> bool:
            return board.turn == chess.WHITE if we_are_white else board.turn == chess.BLACK

        def _make_move(self, game_id: str, board: chess.Board, san_moves: list[str]):
            if board.is_game_over():
                return

            try:
                print(f"Getting move for position after: {' '.join(san_moves[-6:]) if san_moves else 'start'}")
                san_move = get_nanodanya_move(san_moves)
                uci_move = san_to_uci(board, san_move)
            except Exception as e:
                print(f"Error choosing move: {e}")
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    return
                uci_move = legal_moves[0].uci()
                san_move = board.san(legal_moves[0])
                print(f"Fallback move: {san_move} ({uci_move})")

            try:
                print(f"Playing: {san_move} ({uci_move})")
                self.client.bots.make_move(game_id, uci_move)
            except Exception as e:
                print(f"Lichess rejected move: {e}")

        def handle_challenge(self, challenge: dict):
            challenge_id = challenge["id"]
            challenger = challenge["challenger"]["name"]
            variant = challenge["variant"]["key"]

            if variant != "standard":
                print(f"Declining {challenger}'s challenge: unsupported variant {variant}")
                self.client.bots.decline_challenge(challenge_id, reason="standard")
                return

            print(f"Accepting challenge from {challenger}")
            self.client.bots.accept_challenge(challenge_id)

        def run(self):
            print("nanoDanya Lichess Bot started!")
            print("Waiting for challenges...")

            for event in self.client.bots.stream_incoming_events():
                if event["type"] == "challenge":
                    self.handle_challenge(event["challenge"])
                elif event["type"] == "gameStart":
                    game_id = event["game"]["gameId"]
                    if game_id in self.active_games:
                        continue
                    self.active_games.add(game_id)
                    thread = threading.Thread(target=self._safe_handle_game, args=(game_id,))
                    thread.daemon = True
                    thread.start()

        def _safe_handle_game(self, game_id: str):
            try:
                self.handle_game(game_id)
            except Exception as e:
                print(f"Game {game_id} error: {e}")
            finally:
                self.active_games.discard(game_id)

    token = os.environ["LICHESS_BOT_TOKEN"]
    bot = NanoDanyaBot(token)

    def listen():
        while True:
            try:
                bot.run()
            except Exception as e:
                print(f"Error: {e}")
                print("Reconnecting in 5 seconds...")
                time.sleep(5)

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()

    deadline = time.time() + RUN_SECONDS
    while time.time() < deadline:
        time.sleep(30)
    print("Reached soft deadline, exiting cleanly for the next scheduled run")
