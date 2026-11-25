"""Core Guess Arena game engine with zero terminal I/O (MVP 1 & MVP 4)."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from arena.exceptions import (
    ArenaError,
    GameConfigError,
    GameOverError,
    InconsistentFeedbackError,
    InvalidGuessError,
)
from arena.modes import GameMode, GuessOutcome, get_mode

DIFFICULTY_PRESETS: dict[str, tuple[int, int, int]] = {
    "easy": (1, 50, 8),
    "medium": (1, 100, 10),
    "hard": (1, 1000, 12),
}


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Immutable configuration for a Guess Arena round."""

    low: int = 1
    high: int = 100
    max_tries: int | None = None
    seed: int | None = None
    difficulty: str = "medium"
    secret: int | None = None
    time_limit_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise GameConfigError(f"low ({self.low}) must be strictly less than high ({self.high}).")
        if self.max_tries is not None and self.max_tries <= 0:
            raise GameConfigError(f"max_tries must be positive, got {self.max_tries}.")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise GameConfigError("time_limit_seconds must be positive.")
        if self.secret is not None and not (self.low <= self.secret <= self.high):
            raise GameConfigError(
                f"secret ({self.secret}) must lie within [{self.low}, {self.high}]."
            )

    @classmethod
    def from_difficulty(
        cls,
        difficulty: str,
        *,
        max_tries: int | None = None,
        seed: int | None = None,
        secret: int | None = None,
        time_limit_seconds: float | None = None,
    ) -> GameConfig:
        """Create a GameConfig from a named difficulty preset ('easy', 'medium', 'hard')."""
        key = difficulty.strip().lower()
        if key not in DIFFICULTY_PRESETS:
            valid = ", ".join(DIFFICULTY_PRESETS)
            raise GameConfigError(f"Unknown difficulty {difficulty!r}. Expected: {valid}.")
        low, high, default_tries = DIFFICULTY_PRESETS[key]
        return cls(
            low=low,
            high=high,
            max_tries=max_tries if max_tries is not None else default_tries,
            seed=seed,
            difficulty=key,
            secret=secret,
            time_limit_seconds=time_limit_seconds,
        )

    @property
    def span(self) -> int:
        """Count of integer values in [low, high]."""
        return self.high - self.low + 1


@dataclass(frozen=True, slots=True)
class GuessResult:
    """Full structured result of a single guess."""

    attempt: int
    guess: int
    outcome: GuessOutcome
    hint: str
    remaining_tries: int | None
    current_low: int
    current_high: int
    elapsed_seconds: float
    distance: int | None = None
    temperature: str | None = None
    won: bool = False
    game_over: bool = False
    score: int = 0


@dataclass(frozen=True, slots=True)
class GameState:
    """Public, secret-free snapshot of the current game state."""

    low: int
    high: int
    current_low: int
    current_high: int
    attempts: int
    max_tries: int | None
    remaining_tries: int | None
    history: tuple[GuessResult, ...]
    mode_name: str
    difficulty: str
    game_over: bool
    won: bool
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class GameSummary:
    """Completed game summary suitable for persistence and display."""

    mode: str
    difficulty: str
    low: int
    high: int
    secret: int
    max_tries: int | None
    won: bool
    attempts: int
    score: int
    duration_seconds: float
    history: tuple[GuessResult, ...]


class Game:
    """Reusable Guess Arena game engine independent of UI or I/O."""

    def __init__(
        self,
        config: GameConfig | None = None,
        *,
        mode: GameMode | str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.mode: GameMode = get_mode(mode)
        base_config = config or GameConfig()
        self.config: GameConfig = self.mode.configure(base_config)
        self._clock = clock
        self._start_time = self._clock()
        self._rng = random.Random(self.config.seed)
        self._secret: int = (
            self.config.secret
            if self.config.secret is not None
            else self._rng.randint(self.config.low, self.config.high)
        )
        self._current_low = self.config.low
        self._current_high = self.config.high
        self._history: list[GuessResult] = []
        self._game_over = False
        self._won = False
        self._score = 0
        self._end_elapsed = 0.0

    def _elapsed(self) -> float:
        if self._game_over:
            return self._end_elapsed
        return max(0.0, self._clock() - self._start_time)

    def submit_guess(self, value: int) -> GuessResult:
        """Process a guess and return rich GuessResult metadata."""
        if self._game_over:
            raise GameOverError("Cannot guess after the game has ended.")
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidGuessError(f"Guess must be an integer, got {value!r}.")
        if not (self.config.low <= value <= self.config.high):
            raise InvalidGuessError(
                f"Guess {value} is outside [{self.config.low}, {self.config.high}]."
            )

        elapsed = max(0.0, self._clock() - self._start_time)
        attempt = len(self._history) + 1
        previous_distance = self._history[-1].distance if self._history else None

        # Enforce time limit before rewarding a late winning guess.
        timed_out = (
            self.config.time_limit_seconds is not None
            and elapsed > self.config.time_limit_seconds
        )

        feedback = self.mode.evaluate(
            guess=value,
            secret=self._secret,
            attempt=attempt,
            config=self.config,
            elapsed_seconds=elapsed,
            previous_distance=previous_distance,
        )

        if feedback.narrows_bounds and not timed_out:
            if feedback.outcome is GuessOutcome.HIGHER:
                self._current_low = max(self._current_low, min(self.config.high, value + 1))
            elif feedback.outcome is GuessOutcome.LOWER:
                self._current_high = min(self._current_high, max(self.config.low, value - 1))

        won = feedback.outcome is GuessOutcome.CORRECT and not timed_out
        out_of_tries = (
            self.config.max_tries is not None and attempt >= self.config.max_tries
        )
        game_over = won or out_of_tries or timed_out

        hint = feedback.hint
        if timed_out:
            hint = f"Time limit ({self.config.time_limit_seconds:.1f}s) exceeded!"

        remaining_tries = (
            max(0, self.config.max_tries - attempt)
            if self.config.max_tries is not None
            else None
        )

        score = 0
        if game_over:
            self._game_over = True
            self._won = won
            self._end_elapsed = elapsed
            self._score = self.mode.compute_score(
                won=won,
                attempts=attempt,
                max_tries=self.config.max_tries,
                span=self.config.span,
                elapsed_seconds=elapsed,
            )
            score = self._score

        result = GuessResult(
            attempt=attempt,
            guess=value,
            outcome=feedback.outcome,
            hint=hint,
            remaining_tries=remaining_tries,
            current_low=self._current_low,
            current_high=self._current_high,
            elapsed_seconds=elapsed,
            distance=feedback.distance,
            temperature=feedback.temperature,
            won=won,
            game_over=game_over,
            score=score,
        )
        self._history.append(result)
        return result

    def guess(self, value: int) -> GuessOutcome:
        """Process a guess and return its GuessOutcome (HIGHER, LOWER, CORRECT, or HINT)."""
        return self.submit_guess(value).outcome

    @property
    def state(self) -> GameState:
        """Return the current secret-free GameState."""
        attempts = len(self._history)
        remaining = (
            max(0, self.config.max_tries - attempts)
            if self.config.max_tries is not None
            else None
        )
        return GameState(
            low=self.config.low,
            high=self.config.high,
            current_low=self._current_low,
            current_high=self._current_high,
            attempts=attempts,
            max_tries=self.config.max_tries,
            remaining_tries=remaining,
            history=tuple(self._history),
            mode_name=self.mode.name,
            difficulty=self.config.difficulty,
            game_over=self._game_over,
            won=self._won,
            elapsed_seconds=self._elapsed(),
        )

    def reveal_secret(self) -> int:
        """Return the secret number once the round is over."""
        if not self._game_over:
            raise ArenaError("Secret cannot be revealed while the game is in progress.")
        return self._secret

    def summary(self) -> GameSummary:
        """Return a completed GameSummary for persistence and stats."""
        if not self._game_over:
            raise ArenaError("Game summary is only available after the game finishes.")
        return GameSummary(
            mode=self.mode.name,
            difficulty=self.config.difficulty,
            low=self.config.low,
            high=self.config.high,
            secret=self._secret,
            max_tries=self.config.max_tries,
            won=self._won,
            attempts=len(self._history),
            score=self._score,
            duration_seconds=self._end_elapsed,
            history=tuple(self._history),
        )


@dataclass(slots=True)
class ReverseSession:
    """State machine for Reverse Mode where a bot guesses and the user verifies answers."""

    config: GameConfig = field(default_factory=GameConfig)
    _current_low: int = field(init=False)
    _current_high: int = field(init=False)
    _history: list[GuessResult] = field(default_factory=list, init=False)
    _won: bool = field(default=False, init=False)
    _game_over: bool = field(default=False, init=False)
    _start_time: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self) -> None:
        self._current_low = self.config.low
        self._current_high = self.config.high

    @property
    def state(self) -> GameState:
        """Public GameState passed to the guessing bot."""
        attempts = len(self._history)
        remaining = (
            max(0, self.config.max_tries - attempts)
            if self.config.max_tries is not None
            else None
        )
        return GameState(
            low=self.config.low,
            high=self.config.high,
            current_low=self._current_low,
            current_high=self._current_high,
            attempts=attempts,
            max_tries=self.config.max_tries,
            remaining_tries=remaining,
            history=tuple(self._history),
            mode_name="reverse",
            difficulty=self.config.difficulty,
            game_over=self._game_over,
            won=self._won,
            elapsed_seconds=max(0.0, time.monotonic() - self._start_time),
        )

    def apply_feedback(self, bot_guess: int, outcome: GuessOutcome) -> GuessResult:
        """Apply human verification ('HIGHER', 'LOWER', 'CORRECT') to a bot guess."""
        if self._game_over:
            raise GameOverError("Reverse session has already finished.")
        if outcome not in {GuessOutcome.HIGHER, GuessOutcome.LOWER, GuessOutcome.CORRECT}:
            raise ArenaError(f"Reverse mode requires HIGHER, LOWER, or CORRECT; got {outcome}.")

        new_low = self._current_low
        new_high = self._current_high
        if outcome is GuessOutcome.HIGHER:
            new_low = max(new_low, bot_guess + 1)
        elif outcome is GuessOutcome.LOWER:
            new_high = min(new_high, bot_guess - 1)

        if outcome is not GuessOutcome.CORRECT and new_low > new_high:
            raise InconsistentFeedbackError(
                f"Contradictory feedback {outcome.value} for guess {bot_guess}: "
                f"remaining bounds [{new_low}, {new_high}] are empty!"
            )

        self._current_low = new_low
        self._current_high = new_high
        attempt = len(self._history) + 1
        won = outcome is GuessOutcome.CORRECT
        out_of_tries = (
            self.config.max_tries is not None and attempt >= self.config.max_tries
        )
        self._won = won
        self._game_over = won or out_of_tries

        result = GuessResult(
            attempt=attempt,
            guess=bot_guess,
            outcome=outcome,
            hint="Bot guessed correctly!" if won else f"User replied {outcome.value}",
            remaining_tries=(
                max(0, self.config.max_tries - attempt)
                if self.config.max_tries is not None
                else None
            ),
            current_low=self._current_low,
            current_high=self._current_high,
            elapsed_seconds=max(0.0, time.monotonic() - self._start_time),
            won=won,
            game_over=self._game_over,
        )
        self._history.append(result)
        return result

    def verify_against_secret(self, bot_guess: int, secret: int) -> GuessOutcome:
        """Helper when the user pre-entered their secret for automatic verification."""
        if bot_guess == secret:
            return GuessOutcome.CORRECT
        return GuessOutcome.HIGHER if bot_guess < secret else GuessOutcome.LOWER
