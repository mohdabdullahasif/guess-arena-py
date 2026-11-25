"""FastAPI JSON and WebSocket API for Guess Arena (Stretch goal)."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any, Literal

from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from arena.engine import Game, GameConfig, GameState, GuessResult
from arena.exceptions import ArenaError, ModeNotFoundError
from arena.modes import list_modes
from arena.players import BinarySearchBot, LearningBot, RandomBot, simulate_games
from arena.storage import ArenaStorage, resolve_db_path


class CreateGameRequest(BaseModel):
    """Payload to start a new hosted Guess Arena game session."""

    player_name: str = Field(default="Player", min_length=1)
    mode: str = Field(default="classic")
    difficulty: Literal["easy", "medium", "hard", "custom"] = "medium"
    low: int | None = None
    high: int | None = None
    max_tries: int | None = None
    seed: int | None = None


class GuessRequest(BaseModel):
    """Payload to submit a single integer guess."""

    guess: int


class SimulateRequest(BaseModel):
    """Payload to run a headless bot simulation."""

    bot: Literal["random", "binary", "learning"] = "binary"
    games: int = Field(default=1000, ge=1, le=20000)
    low: int = 1
    high: int = 100
    seed: int | None = 42


class ActiveSession(BaseModel):
    """Metadata for an in-memory HTTP game session."""

    model_config = {"arbitrary_types_allowed": True}

    player_name: str
    game: Game
    persisted: bool = False


def _serialize_state(state: GameState) -> dict[str, Any]:
    return {
        "low": state.low,
        "high": state.high,
        "current_low": state.current_low,
        "current_high": state.current_high,
        "attempts": state.attempts,
        "max_tries": state.max_tries,
        "remaining_tries": state.remaining_tries,
        "mode_name": state.mode_name,
        "difficulty": state.difficulty,
        "game_over": state.game_over,
        "won": state.won,
        "elapsed_seconds": round(state.elapsed_seconds, 3),
        "history": [_serialize_result(item) for item in state.history],
    }


def _serialize_result(result: GuessResult) -> dict[str, Any]:
    return {
        "attempt": result.attempt,
        "guess": result.guess,
        "outcome": result.outcome.value,
        "hint": result.hint,
        "remaining_tries": result.remaining_tries,
        "current_low": result.current_low,
        "current_high": result.current_high,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "distance": result.distance,
        "temperature": result.temperature,
        "won": result.won,
        "game_over": result.game_over,
        "score": result.score,
    }


def _build_config(payload: CreateGameRequest) -> GameConfig:
    if payload.difficulty != "custom" and payload.low is None and payload.high is None:
        return GameConfig.from_difficulty(
            payload.difficulty,
            max_tries=payload.max_tries,
            seed=payload.seed,
        )
    return GameConfig(
        low=payload.low if payload.low is not None else 1,
        high=payload.high if payload.high is not None else 100,
        max_tries=payload.max_tries,
        seed=payload.seed,
        difficulty=payload.difficulty,
    )


def create_app(storage: ArenaStorage | None = None) -> FastAPI:
    """Create the Guess Arena FastAPI JSON & WebSocket application."""
    api = FastAPI(
        title="Guess Arena API",
        version="0.1.0",
        description="JSON REST and WebSocket API over the Guess Arena engine and SQLite stats.",
    )
    sessions: dict[str, ActiveSession] = {}

    def get_storage() -> ArenaStorage:
        if storage is not None:
            return storage
        return ArenaStorage(resolve_db_path())

    @api.exception_handler(ModeNotFoundError)
    async def _mode_not_found(_: Request, exc: ModeNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @api.exception_handler(ArenaError)
    async def _arena_error(_: Request, exc: ArenaError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @api.get("/")
    def root() -> dict[str, Any]:
        return {
            "name": "Guess Arena API",
            "docs": "/docs",
            "redoc": "/redoc",
            "endpoints": [
                "GET /health",
                "GET /modes",
                "POST /games",
                "GET /games/{game_id}",
                "POST /games/{game_id}/guess",
                "POST /simulate",
                "GET /stats/{player_name}",
                "GET /leaderboard",
                "WS /ws/play",
            ],
        }

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/modes")
    def modes() -> list[dict[str, Any]]:
        return [
            {
                "name": mode.name,
                "description": mode.description,
                "is_reverse": mode.is_reverse,
            }
            for mode in list_modes()
        ]

    @api.post("/games", status_code=201)
    def create_game(payload: CreateGameRequest) -> dict[str, Any]:
        cfg = _build_config(payload)
        game = Game(cfg, mode=payload.mode)
        game_id = uuid.uuid4().hex[:12]
        sessions[game_id] = ActiveSession(player_name=payload.player_name.strip(), game=game)
        return {"game_id": game_id, "state": _serialize_state(game.state)}

    @api.get("/games/{game_id}")
    def get_game(game_id: str) -> dict[str, Any]:
        session = sessions.get(game_id)
        if session is None:
            return JSONResponse(status_code=404, content={"detail": "Game session not found."})  # type: ignore[return-value]
        return {"game_id": game_id, "state": _serialize_state(session.game.state)}

    @api.post("/games/{game_id}/guess")
    def submit_guess(
        game_id: str,
        payload: GuessRequest,
        repo: ArenaStorage = Depends(get_storage),
    ) -> dict[str, Any]:
        session = sessions.get(game_id)
        if session is None:
            return JSONResponse(status_code=404, content={"detail": "Game session not found."})  # type: ignore[return-value]
        result = session.game.submit_guess(payload.guess)
        response: dict[str, Any] = {
            "game_id": game_id,
            "result": _serialize_result(result),
            "state": _serialize_state(session.game.state),
        }
        if result.game_over and not session.persisted:
            summary = session.game.summary()
            repo.record_game(session.player_name, summary, is_bot=False)
            session.persisted = True
            response["secret"] = summary.secret
        return response

    @api.post("/simulate")
    def run_simulation(payload: SimulateRequest) -> dict[str, Any]:
        cfg = GameConfig(low=payload.low, high=payload.high, seed=payload.seed)
        factories = {
            "random": lambda: RandomBot(seed=payload.seed),
            "binary": BinarySearchBot,
            "learning": lambda: LearningBot(low=payload.low, high=payload.high),
        }
        report = simulate_games(factories[payload.bot], num_games=payload.games, config=cfg)
        return asdict(report)

    @api.get("/stats/{player_name}")
    def player_stats(
        player_name: str,
        difficulty: str | None = None,
        repo: ArenaStorage = Depends(get_storage),
    ) -> dict[str, Any]:
        return asdict(repo.get_player_stats(player_name, difficulty=difficulty))

    @api.get("/leaderboard")
    def leaderboard(
        difficulty: str | None = None,
        mode: str | None = None,
        limit: int = 10,
        repo: ArenaStorage = Depends(get_storage),
    ) -> list[dict[str, Any]]:
        entries = repo.get_leaderboard(difficulty=difficulty, mode=mode, limit=limit)
        return [asdict(entry) for entry in entries]

    @api.websocket("/ws/play")
    async def websocket_play(websocket: WebSocket) -> None:
        await websocket.accept()
        repo = get_storage()
        game: Game | None = None
        player_name = "WebSocketPlayer"
        persisted = False

        try:
            while True:
                data = await websocket.receive_json()
                action = data.get("action")
                if action == "start":
                    player_name = str(data.get("player_name", "WebSocketPlayer")).strip() or "WebSocketPlayer"
                    mode = str(data.get("mode", "classic"))
                    difficulty = str(data.get("difficulty", "medium"))
                    cfg = (
                        GameConfig.from_difficulty(difficulty, seed=data.get("seed"))
                        if difficulty in {"easy", "medium", "hard"}
                        and "low" not in data
                        and "high" not in data
                        else GameConfig(
                            low=int(data.get("low", 1)),
                            high=int(data.get("high", 100)),
                            max_tries=data.get("max_tries"),
                            seed=data.get("seed"),
                            difficulty=difficulty,
                        )
                    )
                    game = Game(cfg, mode=mode)
                    persisted = False
                    await websocket.send_json({"type": "started", "state": _serialize_state(game.state)})
                elif action == "guess":
                    if game is None:
                        await websocket.send_json({"type": "error", "message": "Start a game first."})
                        continue
                    try:
                        value = int(data["value"])
                        result = game.submit_guess(value)
                    except (KeyError, TypeError, ValueError, ArenaError) as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
                        continue
                    payload: dict[str, Any] = {
                        "type": "result",
                        "result": _serialize_result(result),
                        "state": _serialize_state(game.state),
                    }
                    if result.game_over and not persisted:
                        summary = game.summary()
                        repo.record_game(player_name, summary, is_bot=False)
                        persisted = True
                        payload["secret"] = summary.secret
                    await websocket.send_json(payload)
                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"Unknown action: {action!r}"}
                    )
        except WebSocketDisconnect:
            return

    return api


app = create_app()
