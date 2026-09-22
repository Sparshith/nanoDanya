from contextlib import closing
import time

import chess


def play_game(client, game_id, username, choose_move):
    pending_history = None
    pending_move = None
    submitted_history = None
    while True:
        try:
            with closing(client.bots.stream_game_state(game_id)) as events:
                for event in events:
                    if event["type"] == "gameFull":
                        white = event["white"].get("name") or event["white"]["id"]
                        we_are_white = white.lower() == username.lower()
                        state = event["state"]
                    elif event["type"] == "gameState":
                        state = event
                    else:
                        continue

                    if state["status"] not in ("created", "started"):
                        print(f"Game {game_id} finished: {state['status']}")
                        return state
                    history = tuple(state["moves"].split())
                    board = chess.Board()
                    san_moves = []
                    for uci in history:
                        move = chess.Move.from_uci(uci)
                        san_moves.append(board.san(move))
                        board.push(move)
                    if board.is_game_over():
                        winner = board.outcome().winner
                        return {"status": "mate" if board.is_checkmate() else "draw",
                                **({"winner": "white" if winner else "black"} if winner is not None else {})}
                    if board.turn != we_are_white or history == submitted_history:
                        continue

                    if history != pending_history:
                        try:
                            san_move = choose_move(san_moves)
                            move = board.parse_san(san_move)
                        except Exception as e:
                            print(f"Game {game_id} error choosing move: {e}")
                            move = next(iter(board.legal_moves))
                            print(f"Game {game_id} fallback move: {board.san(move)}")
                        pending_history = history
                        pending_move = move.uci()

                    print(f"Game {game_id} playing: {board.san(chess.Move.from_uci(pending_move))} ({pending_move})")
                    # A lost response may hide an accepted move. Reconnect before retrying.
                    client.bots.make_move(game_id, pending_move)
                    submitted_history = history
        except Exception as e:
            print(f"Game {game_id} error, refreshing state: {e}")
            if getattr(e, "status_code", None) == 429:
                time.sleep(65)
                continue
        time.sleep(5)
