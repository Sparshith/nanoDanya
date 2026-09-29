"""nanoDanya Lichess bot on Modal.

The listener is intentionally cheap: it only holds the Lichess event stream open
and calls the separate nanoDanya model endpoint when it is our turn.
"""

import modal

from lichess_bot.schedule import can_start, challenge_cooldown, finish_trial_game, games_today, limits, record_game, trial_counts, trial_key

app = modal.App("nanodanya-lichess-bot")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "berserk==0.14.0", "chess==1.11.2", "requests==2.34.2"
).add_local_python_source("lichess_bot")

state_store = modal.Dict.from_name("nanodanya-lichess-state", create_if_missing=True)

NANODANYA_API_URL = "https://sparshithsampath--nanodanya-chess-serve.modal.run/move"
BOT_TIMEOUT_SECONDS = 24 * 60 * 60
RUN_SECONDS = int(23.5 * 60 * 60)

OPPONENT_RATING_RANGE = (1000, 2200)
ALLOWED_SPEEDS = {"bullet", "rapid", "classical", "correspondence"}
TRIAL_CLOCKS = {"rapid": (600, 0), "bullet": (60, 1)}
TEMPERATURE = 0.2


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("lichess-bot-token")],
    schedule=modal.Period(hours=23),
    timeout=BOT_TIMEOUT_SECONDS,
    cpu=0.125,
    memory=256,
    max_containers=1,
)
def run_bot(dry_run: bool = False):
    import os
    import json
    import random
    import urllib.request
    import threading
    import time
    from lichess_bot.game import play_game
    import berserk

    def get_nanodanya_move(moves: list[str], temperature: float = TEMPERATURE) -> str:
        data = json.dumps({"moves": moves, "temperature": temperature}).encode()
        req = urllib.request.Request(
            NANODANYA_API_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read().decode())
        return result["move"]

    class NanoDanyaBot:
        def __init__(self, token: str):
            session = berserk.TokenSession(token)
            self.client = berserk.Client(session)
            self.username = self.client.account.get()["username"]
            self.active_games = set()
            self.lock = threading.RLock()
            self.state = state_store.get("budget", {"games": {}, "last_game_finish": 0})
            bullet_config = state_store.get("bullet_config")
            if bullet_config:
                self.state.setdefault("bullet_trial", bullet_config)
            self.state.setdefault("challenge_cooldowns", {})
            cutoff = time.time() - 2 * 24 * 60 * 60
            self.state["games"] = {gid: start for gid, start in self.state["games"].items() if start >= cutoff}
            history_since = min(cutoff, *(self.state.get(key, {}).get("started_at", cutoff) for key in ("trial", "bullet_trial")))
            for game in self.client.games.export_by_player(
                self.username, since=int(history_since * 1000), ongoing=True,
                moves=False, clocks=False, evals=False, opening=False,
            ):
                record_game(self.state, game["id"], game["createdAt"].timestamp(), game.get("speed", "rapid"))
                finish_trial_game(self.state, game["id"], game)
            if not dry_run:
                state_store["budget"] = self.state
            self.pending_challenge_id = None
            self.pending_since = 0.0
            self.draining = False
            print(f"Logged in as: {self.username}")
            print(f"Move temperature: {TEMPERATURE}")

        def _save_progress(self):
            for speed in TRIAL_CLOCKS:
                trial = self.state.get(trial_key(speed))
                if not trial:
                    continue
                completed, pending = trial_counts(self.state, speed)
                print(f"{speed} trial: {completed}/{trial['target']} completed, {pending} ongoing")
                if completed >= trial["target"] and "finished_at" not in trial:
                    trial["finished_at"] = time.time()
                    print(f"{speed} trial complete; no further {speed} games will be started")
            state_store["budget"] = self.state
            for speed in TRIAL_CLOCKS:
                trial = self.state.get(trial_key(speed))
                field = "final_" + speed
                if trial and "finished_at" in trial and field not in trial:
                    try:
                        trial[field] = self.client.account.get()["perfs"][speed]
                        state_store["budget"] = self.state
                        print(f"Final {speed} rating: {trial[field]}")
                    except Exception as error:
                        print(f"Could not fetch final rating: {error}")
                if trial and field in trial:
                    self.state.setdefault("past_trials", []).append(self.state.pop(trial_key(speed)))
                    state_store["budget"] = self.state
                    print(f"{speed} trial archived; back to normal limits")

        def _at_capacity(self):
            return len(self.active_games) >= self.state.get("trial", {}).get("max_concurrent", 1)

        def matchmake(self):
            with self.lock:
                self._matchmake()

        def _matchmake(self):
            if state_store.get("drain", False):
                self.draining = True
            if self.draining:
                self._cancel_pending()
                return
            if self._at_capacity():
                return
            speed = next((speed for speed in ("bullet", "rapid") if can_start(self.state, time.time(), speed)), None)
            if speed is None:
                self._cancel_pending()
                return
            cooldowns = self.state["challenge_cooldowns"]
            if time.time() < cooldowns.get("*", 0):
                return
            if self.pending_challenge_id and time.time() - self.pending_since < 60:
                return

            for game in self.client.games.get_ongoing():
                self.start_game(game["gameId"], game.get("speed", "rapid"))
            if self._at_capacity() or not can_start(self.state, time.time(), speed):
                return

            bots = [bot for bot in self.client.bots.get_online_bots(limit=200)
                    if time.time() >= cooldowns.get(bot["username"].lower(), 0)]
            lo, hi = OPPONENT_RATING_RANGE
            candidates = [
                b for b in bots
                if b["username"].lower() != self.username.lower()
                and lo <= b.get("perfs", {}).get(speed, {}).get("rating", 0) <= hi
            ]
            if not candidates:
                candidates = [b for b in bots if b["username"].lower() != self.username.lower()]
            if not candidates:
                print("Matchmaking: no online bots found")
                return

            if self.pending_challenge_id:
                self._cancel_pending()
                for game in self.client.games.get_ongoing():
                    self.start_game(game["gameId"], game.get("speed", "rapid"))
                if self._at_capacity() or not can_start(self.state, time.time(), speed):
                    return

            opponent = random.choice(candidates)["username"]
            print(f"Matchmaking: challenging {opponent} for {speed}")
            try:
                if speed == "bullet":
                    get_nanodanya_move([])
                clock_limit, clock_increment = TRIAL_CLOCKS[speed]
                challenge = self.client.challenges.create(
                    opponent,
                    rated=True,
                    clock_limit=clock_limit,
                    clock_increment=clock_increment,
                )
                self.pending_challenge_id = (challenge.get("challenge") or challenge).get("id")
                self.pending_since = time.time()
            except berserk.exceptions.ResponseError as e:
                cause = e.cause if isinstance(e.cause, dict) else {}
                if cause.get("ratelimit") or e.status_code == 429:
                    owner, until = challenge_cooldown(cause, opponent, time.time())
                    cooldowns[owner] = until
                else:
                    cooldowns[opponent.lower()] = time.time() + 30 * 60
                state_store["budget"] = self.state
                print(f"Matchmaking: challenge to {opponent} failed: {e}")

        def _cancel_pending(self):
            if self.pending_challenge_id:
                try:
                    self.client.challenges.cancel(self.pending_challenge_id)
                except berserk.exceptions.ResponseError as e:
                    if e.status_code not in (400, 404):
                        raise
                self.pending_challenge_id = None

        def handle_challenge(self, challenge: dict):
            with self.lock:
                self._handle_challenge(challenge)

        def _handle_challenge(self, challenge: dict):
            challenge_id = challenge["id"]
            challenger = challenge["challenger"]["name"]
            variant = challenge["variant"]["key"]
            speed = challenge.get("speed")

            if challenger.lower() == self.username.lower():
                return

            if variant != "standard":
                print(f"Declining {challenger}'s challenge: unsupported variant {variant}")
                self.client.bots.decline_challenge(challenge_id, reason="standard")
                return

            if speed not in ALLOWED_SPEEDS:
                print(f"Declining {challenger}'s challenge: too fast ({speed})")
                self.client.bots.decline_challenge(challenge_id, reason="tooFast")
                return

            if self.state.get("trial") or self.state.get("bullet_trial"):
                clock = challenge.get("timeControl", {})
                if (not self.state.get(trial_key(speed)) or not challenge.get("rated")
                        or (clock.get("limit"), clock.get("increment")) != TRIAL_CLOCKS.get(speed)):
                    self.client.bots.decline_challenge(challenge_id, reason="later")
                    return
            elif speed == "bullet":
                self.client.bots.decline_challenge(challenge_id, reason="tooFast")
                return

            if self.draining or self._at_capacity() or self.pending_challenge_id or not can_start(self.state, time.time(), speed):
                self.client.bots.decline_challenge(challenge_id, reason="later")
                return
            if speed == "bullet":
                get_nanodanya_move([])

            print(f"Accepting challenge from {challenger}")
            self.client.bots.accept_challenge(challenge_id)
            self.pending_challenge_id = challenge_id
            self.pending_since = time.time()

        def start_game(self, game_id, speed="rapid"):
            with self.lock:
                if game_id in self.active_games:
                    return
                self.active_games.add(game_id)
                record_game(self.state, game_id, time.time(), speed)
                self._save_progress()
                self.pending_challenge_id = None
                print(f"Today's games: {games_today(self.state, time.time())}")
                thread = threading.Thread(target=self._safe_handle_game, args=(game_id,), daemon=True)
                thread.start()

        def run(self):
            print("nanoDanya Lichess Bot started!")
            print("Waiting for challenges...")

            for event in self.client.bots.stream_incoming_events():
                if event["type"] == "challenge":
                    self.handle_challenge(event["challenge"])
                elif event["type"] == "gameStart":
                    self.start_game(event["game"]["gameId"], event["game"].get("speed", "rapid"))
                elif event["type"] in ("challengeCanceled", "challengeDeclined"):
                    with self.lock:
                        if event["challenge"]["id"] == self.pending_challenge_id:
                            self.pending_challenge_id = None

        def _safe_handle_game(self, game_id: str):
            result = None
            try:
                result = play_game(self.client, game_id, self.username, get_nanodanya_move)
            finally:
                with self.lock:
                    if result:
                        finish_trial_game(self.state, game_id, result)
                    self.active_games.discard(game_id)
                    self.state["last_game_finish"] = time.time()
                    self._save_progress()

    token = os.environ["LICHESS_BOT_TOKEN"]
    bot = NanoDanyaBot(token)
    ongoing = bot.client.games.get_ongoing()
    cap, gap, tick = limits(time.time(), bot.state)
    print(f"Matchmaking limits: {cap} games/{'trial' if bot.state.get('trial') else 'day'}, {gap}s gap, {tick}s tick; temperature {TEMPERATURE}")
    if dry_run:
        return {"username": bot.username, "games_today": games_today(bot.state, time.time()),
                "cap": cap, "gap_seconds": gap, "tick_seconds": tick,
                "temperature": TEMPERATURE, "trial": bot.state.get("trial"),
                "bullet_trial": bot.state.get("bullet_trial"),
                "ongoing_games": [game["gameId"] for game in ongoing]}
    bot._save_progress()
    for game in ongoing:
        bot.start_game(game["gameId"], game.get("speed", "rapid"))

    def listen():
        while True:
            try:
                bot.run()
                print("Event stream closed, reconnecting in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                print(f"Error: {e}")
                if "429" in str(e):
                    print("Rate limited, backing off 65 seconds...")
                    time.sleep(65)
                else:
                    print("Reconnecting in 5 seconds...")
                    time.sleep(5)

    def matchmaking():
        while True:
            time.sleep(limits(time.time(), bot.state)[2])
            try:
                bot.matchmake()
            except Exception as e:
                print(f"Matchmaking error: {e}")

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()

    matchmaker = threading.Thread(target=matchmaking, daemon=True)
    matchmaker.start()

    deadline = time.time() + RUN_SECONDS
    while time.time() < deadline:
        time.sleep(5)
        if bot.draining and not bot.active_games:
            break
    with bot.lock:
        bot.draining = True
        if bot.pending_challenge_id:
            try:
                bot._cancel_pending()
            except berserk.exceptions.ResponseError:
                pass
    while bot.active_games:
        time.sleep(5)
    print("Reached soft deadline, exiting cleanly for the next scheduled run")


@app.local_entrypoint()
def check():
    print(run_bot.remote(dry_run=True))
