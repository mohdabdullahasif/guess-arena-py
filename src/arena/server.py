"""Asyncio TCP multiplayer server for Guess Arena (MVP 6)."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from arena.engine import GameConfig
from arena.modes import GameMode, GuessOutcome, get_mode


@dataclass(slots=True)
class ConnectedPlayer:
    """State tracked per connected multiplayer client."""

    name: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    wins: int = 0
    round_attempts: int = 0
    total_guesses: int = 0
    last_distance: int | None = None
    finished_round: bool = False


@dataclass(slots=True)
class ArenaServer:
    """Asyncio TCP server hosting multi-client Guess Arena rounds."""

    host: str = "127.0.0.1"
    port: int = 8765
    config: GameConfig = field(default_factory=GameConfig)
    mode_name: str = "classic"
    preset_secrets: Sequence[int] | None = None

    _server: asyncio.Server | None = field(default=None, init=False)
    _clients: dict[str, ConnectedPlayer] = field(default_factory=dict, init=False)
    _mode: GameMode = field(init=False)
    _rng: random.Random = field(init=False)
    _round_number: int = field(default=0, init=False)
    _secret: int = field(default=0, init=False)
    _round_active: bool = field(default=False, init=False)
    _round_winner: str | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self._mode = get_mode(self.mode_name)
        self.config = self._mode.configure(self.config)
        self._rng = random.Random(self.config.seed)
        self._start_new_round_unlocked()

    @property
    def bound_port(self) -> int:
        """Actual listening TCP port (useful when initialized with port=0)."""
        if self._server is not None and self._server.sockets:
            return int(self._server.sockets[0].getsockname()[1])
        return self.port

    @property
    def round_number(self) -> int:
        return self._round_number

    @property
    def connected_players(self) -> list[str]:
        return sorted(self._clients)

    def _next_secret(self) -> int:
        if self.preset_secrets is not None and self._round_number < len(self.preset_secrets):
            return int(self.preset_secrets[self._round_number])
        if self.config.secret is not None and self._round_number == 0:
            return self.config.secret
        return self._rng.randint(self.config.low, self.config.high)

    def _start_new_round_unlocked(self) -> None:
        self._secret = self._next_secret()
        self._round_number += 1
        self._round_active = True
        self._round_winner = None
        for player in self._clients.values():
            player.round_attempts = 0
            player.last_distance = None
            player.finished_round = False

    def _scoreboard_payload(self) -> list[dict[str, Any]]:
        ordered = sorted(
            self._clients.values(),
            key=lambda p: (-p.wins, p.round_attempts, p.name.lower()),
        )
        return [
            {
                "name": p.name,
                "wins": p.wins,
                "round_attempts": p.round_attempts,
                "total_guesses": p.total_guesses,
            }
            for p in ordered
        ]

    async def _send(self, writer: asyncio.StreamWriter, payload: dict[str, Any]) -> bool:
        if writer.is_closing():
            return False
        data = (json.dumps(payload) + "\n").encode("utf-8")
        try:
            writer.write(data)
            await writer.drain()
            return True
        except (ConnectionResetError, BrokenPipeError, OSError):
            return False

    async def _broadcast(
        self,
        payload: dict[str, Any],
        *,
        exclude: str | None = None,
    ) -> None:
        targets = [
            (name, client)
            for name, client in list(self._clients.items())
            if name != exclude
        ]
        for _, client in targets:
            await self._send(client.writer, payload)

    async def _read_line(self, reader: asyncio.StreamReader) -> dict[str, Any] | None:
        try:
            raw = await reader.readline()
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError, OSError):
            return None
        if not raw:
            return None
        try:
            parsed = json.loads(raw.decode("utf-8").strip())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"type": "__invalid_json__"}
        if not isinstance(parsed, dict):
            return {"type": "__invalid_json__"}
        return parsed

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        player_name: str | None = None
        try:
            join_msg = await self._read_line(reader)
            if join_msg is None:
                return
            if join_msg.get("type") != "join":
                await self._send(writer, {"type": "error", "message": "First message must be join."})
                return

            candidate_name = str(join_msg.get("name", "")).strip()
            if not candidate_name:
                await self._send(writer, {"type": "error", "message": "Player name cannot be empty."})
                return

            async with self._lock:
                if candidate_name in self._clients:
                    await self._send(
                        writer,
                        {"type": "error", "message": f"Name '{candidate_name}' is already taken."},
                    )
                    return
                player_name = candidate_name
                session = ConnectedPlayer(name=player_name, reader=reader, writer=writer)
                self._clients[player_name] = session
                welcome = {
                    "type": "welcome",
                    "player": player_name,
                    "round": self._round_number,
                    "low": self.config.low,
                    "high": self.config.high,
                    "mode": self._mode.name,
                    "round_active": self._round_active,
                    "scoreboard": self._scoreboard_payload(),
                }
                await self._send(writer, welcome)
                await self._broadcast(
                    {
                        "type": "player_joined",
                        "player": player_name,
                        "scoreboard": self._scoreboard_payload(),
                    },
                    exclude=player_name,
                )

            while True:
                msg = await self._read_line(reader)
                if msg is None:
                    break
                msg_type = msg.get("type")
                if msg_type == "__invalid_json__":
                    await self._send(writer, {"type": "error", "message": "Malformed JSON message."})
                    continue
                if msg_type == "quit":
                    break
                if msg_type == "new_round":
                    async with self._lock:
                        if not self._round_active:
                            self._start_new_round_unlocked()
                            await self._broadcast(
                                {
                                    "type": "new_round",
                                    "round": self._round_number,
                                    "low": self.config.low,
                                    "high": self.config.high,
                                    "mode": self._mode.name,
                                    "scoreboard": self._scoreboard_payload(),
                                }
                            )
                        else:
                            await self._send(
                                writer,
                                {"type": "error", "message": "Current round is still active."},
                            )
                    continue
                if msg_type == "guess":
                    await self._process_guess(player_name, msg.get("value"))
                    continue
                await self._send(
                    writer,
                    {"type": "error", "message": f"Unknown message type: {msg_type!r}."},
                )
        finally:
            if player_name is not None:
                async with self._lock:
                    removed = self._clients.pop(player_name, None)
                    if removed is not None:
                        await self._broadcast(
                            {
                                "type": "player_left",
                                "player": player_name,
                                "scoreboard": self._scoreboard_payload(),
                            }
                        )
            if not writer.is_closing():
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass

    async def _process_guess(self, player_name: str, raw_value: Any) -> None:
        async with self._lock:
            player = self._clients.get(player_name)
            if player is None:
                return
            if not self._round_active:
                await self._send(
                    player.writer,
                    {
                        "type": "error",
                        "message": f"Round {self._round_number} already won by {self._round_winner}. Request new_round.",
                    },
                )
                return
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                await self._send(
                    player.writer,
                    {"type": "error", "message": f"Guess must be an integer, got {raw_value!r}."},
                )
                return
            if not (self.config.low <= raw_value <= self.config.high):
                await self._send(
                    player.writer,
                    {
                        "type": "error",
                        "message": f"Guess {raw_value} must be within [{self.config.low}, {self.config.high}].",
                    },
                )
                return

            player.round_attempts += 1
            player.total_guesses += 1
            feedback = self._mode.evaluate(
                guess=raw_value,
                secret=self._secret,
                attempt=player.round_attempts,
                config=self.config,
                elapsed_seconds=0.0,
                previous_distance=player.last_distance,
            )
            player.last_distance = feedback.distance

            await self._send(
                player.writer,
                {
                    "type": "hint",
                    "round": self._round_number,
                    "guess": raw_value,
                    "outcome": feedback.outcome.value,
                    "hint": feedback.hint,
                    "attempts": player.round_attempts,
                },
            )

            if feedback.outcome is GuessOutcome.CORRECT:
                self._round_active = False
                self._round_winner = player_name
                player.wins += 1
                await self._broadcast(
                    {
                        "type": "round_over",
                        "round": self._round_number,
                        "winner": player_name,
                        "attempts": player.round_attempts,
                        "secret": self._secret,
                        "scoreboard": self._scoreboard_payload(),
                    }
                )
            else:
                await self._broadcast(
                    {
                        "type": "progress",
                        "player": player_name,
                        "attempts": player.round_attempts,
                        "scoreboard": self._scoreboard_payload(),
                    }
                )

    async def start(self) -> None:
        """Start accepting TCP multiplayer connections."""
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)

    async def stop(self) -> None:
        """Close all connected clients and stop the TCP server."""
        async with self._lock:
            for client in list(self._clients.values()):
                if not client.writer.is_closing():
                    client.writer.close()
            self._clients.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def serve_forever(self) -> None:
        """Run the server until cancelled."""
        await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()
