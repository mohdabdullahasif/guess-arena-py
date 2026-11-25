"""SQLite persistence for players, games, guesses, stats, and leaderboards (MVP 3)."""

from __future__ import annotations

import os
import sqlite3
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from arena.engine import GameSummary
from arena.exceptions import StorageError

DEFAULT_DB_PATH = Path.home() / ".guess_arena" / "arena.db"


def resolve_db_path(explicit: str | Path | None = None) -> Path:
    """Resolve SQLite database path from explicit argument, env var, or default."""
    if explicit is not None:
        return Path(explicit)
    env_path = os.environ.get("GUESS_ARENA_DB")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


@dataclass(frozen=True, slots=True)
class PlayerStats:
    """Aggregated player performance metrics across recorded games."""

    player_name: str
    is_bot: bool
    total_games: int
    wins: int
    win_rate: float
    avg_tries: float
    best_score: int
    avg_score: float
    best_streak: int
    streaks_by_difficulty: dict[str, int] = field(default_factory=dict)
    tries_distribution: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    """Single ranked row in the Guess Arena leaderboard."""

    rank: int
    player_name: str
    is_bot: bool
    total_games: int
    wins: int
    win_rate: float
    avg_tries: float
    best_streak: int
    best_score: int
    avg_score: float


def _max_win_streak(outcomes: list[bool]) -> int:
    best = 0
    current = 0
    for won in outcomes:
        if won:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


class ArenaStorage:
    """Thread-safe SQLite repository for Guess Arena players, games, and guesses."""

    _SCHEMA = """
    PRAGMA foreign_keys = ON;
    CREATE TABLE IF NOT EXISTS players (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        is_bot INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS games (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
        mode TEXT NOT NULL,
        difficulty TEXT NOT NULL,
        low INTEGER NOT NULL,
        high INTEGER NOT NULL,
        secret INTEGER NOT NULL,
        max_tries INTEGER,
        won INTEGER NOT NULL,
        attempts INTEGER NOT NULL,
        score INTEGER NOT NULL,
        duration_seconds REAL NOT NULL,
        played_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS guesses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
        attempt INTEGER NOT NULL,
        guess_value INTEGER NOT NULL,
        outcome TEXT NOT NULL,
        elapsed_seconds REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_games_player_diff ON games(player_id, difficulty);
    CREATE INDEX IF NOT EXISTS idx_guesses_game ON guesses(game_id);
    """

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self._is_memory = str(path) == ":memory:"
        self.path = Path(path) if not self._is_memory else Path(":memory:")
        self._lock = threading.RLock()
        self._memory_conn: sqlite3.Connection | None = None
        if self._is_memory:
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._is_memory:
            assert self._memory_conn is not None
            try:
                yield self._memory_conn
                self._memory_conn.commit()
            except sqlite3.DatabaseError as exc:
                self._memory_conn.rollback()
                raise StorageError(f"SQLite error: {exc}") from exc
            except Exception:
                self._memory_conn.rollback()
                raise
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(self.path, timeout=10.0, check_same_thread=False)
        except sqlite3.DatabaseError as exc:
            raise StorageError(f"Failed to open SQLite DB '{self.path}': {exc}") from exc
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
            conn.commit()
        except sqlite3.DatabaseError as exc:
            conn.rollback()
            raise StorageError(f"SQLite error in '{self.path}': {exc}") from exc
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            if not self._is_memory:
                conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(self._SCHEMA)

    def ensure_player(self, name: str, *, is_bot: bool = False) -> int:
        """Fetch or create a player row and return its ID."""
        cleaned = name.strip()
        if not cleaned:
            raise StorageError("Player name cannot be empty.")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, is_bot FROM players WHERE name = ?", (cleaned,)
            ).fetchone()
            if row is not None:
                return int(row["id"])
            cursor = conn.execute(
                "INSERT INTO players (name, is_bot, created_at) VALUES (?, ?, ?)",
                (cleaned, int(is_bot), now),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_game(
        self,
        player_name: str,
        summary: GameSummary,
        *,
        is_bot: bool = False,
    ) -> int:
        """Persist a completed game and all individual guess rows."""
        cleaned = player_name.strip()
        if not cleaned:
            raise StorageError("Player name cannot be empty.")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT id FROM players WHERE name = ?", (cleaned,)).fetchone()
            if row is None:
                cursor = conn.execute(
                    "INSERT INTO players (name, is_bot, created_at) VALUES (?, ?, ?)",
                    (cleaned, int(is_bot), now),
                )
                assert cursor.lastrowid is not None
                player_id = int(cursor.lastrowid)
            else:
                player_id = int(row["id"])

            game_cursor = conn.execute(
                """
                INSERT INTO games (
                    player_id, mode, difficulty, low, high, secret, max_tries,
                    won, attempts, score, duration_seconds, played_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    player_id,
                    summary.mode,
                    summary.difficulty,
                    summary.low,
                    summary.high,
                    summary.secret,
                    summary.max_tries,
                    int(summary.won),
                    summary.attempts,
                    summary.score,
                    summary.duration_seconds,
                    now,
                ),
            )
            assert game_cursor.lastrowid is not None
            game_id = int(game_cursor.lastrowid)

            if summary.history:
                conn.executemany(
                    """
                    INSERT INTO guesses (game_id, attempt, guess_value, outcome, elapsed_seconds)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            game_id,
                            item.attempt,
                            item.guess,
                            item.outcome.value,
                            item.elapsed_seconds,
                        )
                        for item in summary.history
                    ],
                )
            return game_id

    def get_player_stats(
        self,
        player_name: str,
        *,
        difficulty: str | None = None,
    ) -> PlayerStats:
        """Return aggregated statistics and per-difficulty win streaks for a player."""
        cleaned = player_name.strip()
        with self._lock, self._connect() as conn:
            player_row = conn.execute(
                "SELECT id, name, is_bot FROM players WHERE name = ?", (cleaned,)
            ).fetchone()
            if player_row is None:
                return PlayerStats(
                    player_name=cleaned,
                    is_bot=False,
                    total_games=0,
                    wins=0,
                    win_rate=0.0,
                    avg_tries=0.0,
                    best_score=0,
                    avg_score=0.0,
                    best_streak=0,
                )

            params: list[object] = [int(player_row["id"])]
            query = "SELECT difficulty, won, attempts, score FROM games WHERE player_id = ?"
            if difficulty is not None:
                query += " AND difficulty = ?"
                params.append(difficulty.strip().lower())
            query += " ORDER BY id ASC"
            rows = conn.execute(query, params).fetchall()

            if not rows:
                return PlayerStats(
                    player_name=str(player_row["name"]),
                    is_bot=bool(player_row["is_bot"]),
                    total_games=0,
                    wins=0,
                    win_rate=0.0,
                    avg_tries=0.0,
                    best_score=0,
                    avg_score=0.0,
                    best_streak=0,
                )

            outcomes = [bool(r["won"]) for r in rows]
            wins = sum(outcomes)
            win_attempts = [int(r["attempts"]) for r in rows if r["won"]]
            all_attempts = [int(r["attempts"]) for r in rows]
            scores = [int(r["score"]) for r in rows]

            by_diff: dict[str, list[bool]] = {}
            tries_dist: Counter[int] = Counter()
            for r in rows:
                diff_key = str(r["difficulty"])
                by_diff.setdefault(diff_key, []).append(bool(r["won"]))
                tries_dist[int(r["attempts"])] += 1

            streaks_by_diff = {
                diff: _max_win_streak(diff_outcomes)
                for diff, diff_outcomes in sorted(by_diff.items())
            }

            return PlayerStats(
                player_name=str(player_row["name"]),
                is_bot=bool(player_row["is_bot"]),
                total_games=len(rows),
                wins=wins,
                win_rate=wins / len(rows),
                avg_tries=(
                    sum(win_attempts) / len(win_attempts)
                    if win_attempts
                    else sum(all_attempts) / len(all_attempts)
                ),
                best_score=max(scores),
                avg_score=sum(scores) / len(scores),
                best_streak=_max_win_streak(outcomes),
                streaks_by_difficulty=streaks_by_diff,
                tries_distribution=dict(sorted(tries_dist.items())),
            )

    def get_leaderboard(
        self,
        *,
        difficulty: str | None = None,
        mode: str | None = None,
        limit: int = 10,
    ) -> list[LeaderboardEntry]:
        """Rank all players (humans and bots) by win rate, avg score, and avg tries."""
        with self._lock, self._connect() as conn:
            where_clauses: list[str] = []
            params: list[object] = []
            if difficulty is not None:
                where_clauses.append("g.difficulty = ?")
                params.append(difficulty.strip().lower())
            if mode is not None:
                where_clauses.append("g.mode = ?")
                params.append(mode.strip().lower())
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            rows = conn.execute(
                f"""
                SELECT p.id AS player_id, p.name AS player_name, p.is_bot AS is_bot,
                       g.won AS won, g.attempts AS attempts, g.score AS score
                FROM games g
                JOIN players p ON p.id = g.player_id
                {where_sql}
                ORDER BY p.id ASC, g.id ASC
                """,
                params,
            ).fetchall()

        grouped: dict[int, dict[str, object]] = {}
        for row in rows:
            pid = int(row["player_id"])
            bucket = grouped.setdefault(
                pid,
                {
                    "name": str(row["player_name"]),
                    "is_bot": bool(row["is_bot"]),
                    "outcomes": [],
                    "win_attempts": [],
                    "all_attempts": [],
                    "scores": [],
                },
            )
            won = bool(row["won"])
            attempts = int(row["attempts"])
            score = int(row["score"])
            bucket["outcomes"].append(won)  # type: ignore[attr-defined]
            bucket["all_attempts"].append(attempts)  # type: ignore[attr-defined]
            if won:
                bucket["win_attempts"].append(attempts)  # type: ignore[attr-defined]
            bucket["scores"].append(score)  # type: ignore[attr-defined]

        raw_entries: list[LeaderboardEntry] = []
        for bucket in grouped.values():
            outcomes: list[bool] = bucket["outcomes"]  # type: ignore[assignment]
            win_attempts: list[int] = bucket["win_attempts"]  # type: ignore[assignment]
            all_attempts: list[int] = bucket["all_attempts"]  # type: ignore[assignment]
            scores: list[int] = bucket["scores"]  # type: ignore[assignment]
            total = len(outcomes)
            wins = sum(outcomes)
            raw_entries.append(
                LeaderboardEntry(
                    rank=0,
                    player_name=str(bucket["name"]),
                    is_bot=bool(bucket["is_bot"]),
                    total_games=total,
                    wins=wins,
                    win_rate=wins / total,
                    avg_tries=(
                        sum(win_attempts) / len(win_attempts)
                        if win_attempts
                        else sum(all_attempts) / len(all_attempts)
                    ),
                    best_streak=_max_win_streak(outcomes),
                    best_score=max(scores),
                    avg_score=sum(scores) / len(scores),
                )
            )

        raw_entries.sort(
            key=lambda e: (-e.win_rate, -e.avg_score, e.avg_tries, -e.best_streak, -e.total_games)
        )
        limited = raw_entries[: max(1, limit)]
        return [
            LeaderboardEntry(
                rank=idx,
                player_name=entry.player_name,
                is_bot=entry.is_bot,
                total_games=entry.total_games,
                wins=entry.wins,
                win_rate=entry.win_rate,
                avg_tries=entry.avg_tries,
                best_streak=entry.best_streak,
                best_score=entry.best_score,
                avg_score=entry.avg_score,
            )
            for idx, entry in enumerate(limited, start=1)
        ]

    def get_human_guess_frequencies(self, *, low: int = 1, high: int = 100) -> dict[int, int]:
        """Count numbers guessed by human players within [low, high]."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT gs.guess_value AS val, COUNT(*) AS cnt
                FROM guesses gs
                JOIN games g ON g.id = gs.game_id
                JOIN players p ON p.id = g.player_id
                WHERE p.is_bot = 0 AND gs.guess_value BETWEEN ? AND ?
                GROUP BY gs.guess_value
                ORDER BY gs.guess_value ASC
                """,
                (low, high),
            ).fetchall()
        return {int(r["val"]): int(r["cnt"]) for r in rows}

    def get_human_secret_frequencies(self, *, low: int = 1, high: int = 100) -> dict[int, int]:
        """Count secrets picked by humans (e.g. reverse mode) within [low, high]."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT g.secret AS val, COUNT(*) AS cnt
                FROM games g
                JOIN players p ON p.id = g.player_id
                WHERE p.is_bot = 0 AND g.mode = 'reverse' AND g.secret BETWEEN ? AND ?
                GROUP BY g.secret
                ORDER BY g.secret ASC
                """,
                (low, high),
            ).fetchall()
        return {int(r["val"]): int(r["cnt"]) for r in rows}
