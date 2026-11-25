"""Pluggable game modes for Guess Arena (MVP 4)."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Any

from arena.exceptions import ModeNotFoundError

if TYPE_CHECKING:
    from arena.engine import GameConfig


class GuessOutcome(str, Enum):
    """Result returned for each guess."""

    HIGHER = "HIGHER"
    LOWER = "LOWER"
    CORRECT = "CORRECT"
    HINT = "HINT"


@dataclass(frozen=True, slots=True)
class ModeFeedback:
    """Feedback computed by a GameMode for a single guess."""

    outcome: GuessOutcome
    hint: str
    narrows_bounds: bool = True
    distance: int | None = None
    temperature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GameMode(ABC):
    """Abstract base class for pluggable Guess Arena game modes."""

    name: str
    description: str
    is_reverse: bool = False

    def configure(self, config: GameConfig) -> GameConfig:
        """Adjust GameConfig before a round begins (override in subclasses)."""
        return config

    @abstractmethod
    def evaluate(
        self,
        *,
        guess: int,
        secret: int,
        attempt: int,
        config: GameConfig,
        elapsed_seconds: float,
        previous_distance: int | None = None,
    ) -> ModeFeedback:
        """Evaluate a player's guess against the secret."""

    def compute_score(
        self,
        *,
        won: bool,
        attempts: int,
        max_tries: int | None,
        span: int,
        elapsed_seconds: float,
    ) -> int:
        """Calculate final round score."""
        if not won:
            return 0
        base = max(500, 1000 + int(math.log2(max(2, span)) * 100))
        penalty = max(0, attempts - 1) * 75
        return max(50, base - penalty)


class ClassicMode(GameMode):
    """Standard number guessing with HIGHER / LOWER directional hints."""

    name = "classic"
    description = "Standard HIGHER / LOWER hints."

    def evaluate(
        self,
        *,
        guess: int,
        secret: int,
        attempt: int,
        config: GameConfig,
        elapsed_seconds: float,
        previous_distance: int | None = None,
    ) -> ModeFeedback:
        if guess == secret:
            return ModeFeedback(
                outcome=GuessOutcome.CORRECT,
                hint="CORRECT!",
                narrows_bounds=True,
                distance=0,
            )
        if guess < secret:
            return ModeFeedback(
                outcome=GuessOutcome.HIGHER,
                hint="Higher!",
                narrows_bounds=True,
            )
        return ModeFeedback(
            outcome=GuessOutcome.LOWER,
            hint="Lower!",
            narrows_bounds=True,
        )


class TimedMode(ClassicMode):
    """Timed mode where the final score decays for every elapsed second."""

    name = "timed"
    description = "Score drops each second; optional time limit."

    def __init__(self, penalty_per_second: float = 15.0) -> None:
        self.penalty_per_second = penalty_per_second

    def compute_score(
        self,
        *,
        won: bool,
        attempts: int,
        max_tries: int | None,
        span: int,
        elapsed_seconds: float,
    ) -> int:
        base_score = super().compute_score(
            won=won,
            attempts=attempts,
            max_tries=max_tries,
            span=span,
            elapsed_seconds=elapsed_seconds,
        )
        if not won:
            return 0
        time_penalty = int(round(max(0.0, elapsed_seconds) * self.penalty_per_second))
        return max(10, base_score - time_penalty)


class HotColdMode(GameMode):
    """Distance-based feedback (boiling/hot/warm/cool/cold + warmer/colder) without direction."""

    name = "hot_cold"
    description = "Distance-based temperature hints instead of HIGHER/LOWER."

    @staticmethod
    def _classify_temperature(distance: int, span: int) -> str:
        ratio = distance / max(1, span)
        if distance <= 2 or ratio <= 0.02:
            return "BOILING"
        if distance <= 5 or ratio <= 0.06:
            return "HOT"
        if distance <= 15 or ratio <= 0.16:
            return "WARM"
        if distance <= 30 or ratio <= 0.35:
            return "COOL"
        return "COLD"

    def evaluate(
        self,
        *,
        guess: int,
        secret: int,
        attempt: int,
        config: GameConfig,
        elapsed_seconds: float,
        previous_distance: int | None = None,
    ) -> ModeFeedback:
        if guess == secret:
            return ModeFeedback(
                outcome=GuessOutcome.CORRECT,
                hint="CORRECT!",
                narrows_bounds=False,
                distance=0,
                temperature="EXACT",
            )
        distance = abs(guess - secret)
        span = config.high - config.low + 1
        temp = self._classify_temperature(distance, span)
        trend = ""
        if previous_distance is not None:
            if distance < previous_distance:
                trend = " (getting warmer)"
            elif distance > previous_distance:
                trend = " (getting colder)"
            else:
                trend = " (same distance)"
        return ModeFeedback(
            outcome=GuessOutcome.HINT,
            hint=f"{temp}! Distance: {distance}{trend}",
            narrows_bounds=False,
            distance=distance,
            temperature=temp,
        )


class DailyChallengeMode(ClassicMode):
    """Shared daily seeded challenge so every player gets the same secret on a given date."""

    name = "daily"
    description = "Same deterministic daily secret for everyone each day."

    def __init__(self, challenge_date: date | None = None) -> None:
        self.challenge_date = challenge_date

    def configure(self, config: GameConfig) -> GameConfig:
        ref_date = self.challenge_date or date.today()
        digest = hashlib.sha256(
            f"guess-arena:{ref_date.isoformat()}:{config.low}:{config.high}".encode("utf-8")
        ).hexdigest()
        daily_seed = int(digest[:16], 16)
        return replace(config, seed=daily_seed)


class HardcoreMode(GameMode):
    """Limited tries with zero directional or distance hints."""

    name = "hardcore"
    description = "Strictly limited tries and no hints."

    def configure(self, config: GameConfig) -> GameConfig:
        if config.max_tries is not None:
            return config
        span = config.high - config.low + 1
        default_limit = max(3, span.bit_length() - 1)
        return replace(config, max_tries=default_limit)

    def evaluate(
        self,
        *,
        guess: int,
        secret: int,
        attempt: int,
        config: GameConfig,
        elapsed_seconds: float,
        previous_distance: int | None = None,
    ) -> ModeFeedback:
        if guess == secret:
            return ModeFeedback(
                outcome=GuessOutcome.CORRECT,
                hint="CORRECT!",
                narrows_bounds=False,
                distance=0,
            )
        return ModeFeedback(
            outcome=GuessOutcome.HINT,
            hint="Incorrect! (No hints in Hardcore mode)",
            narrows_bounds=False,
        )

    def compute_score(
        self,
        *,
        won: bool,
        attempts: int,
        max_tries: int | None,
        span: int,
        elapsed_seconds: float,
    ) -> int:
        if not won:
            return 0
        return super().compute_score(
            won=won,
            attempts=attempts,
            max_tries=max_tries,
            span=span,
            elapsed_seconds=elapsed_seconds,
        ) * 2


class ReverseMode(ClassicMode):
    """Reverse mode where the human holds the number and verifies bot guesses."""

    name = "reverse"
    description = "You pick the secret number and verify the bot's guesses."
    is_reverse = True


ModeFactory = Callable[[], GameMode]

_MODE_REGISTRY: dict[str, ModeFactory] = {
    "classic": ClassicMode,
    "timed": TimedMode,
    "hot_cold": HotColdMode,
    "daily": DailyChallengeMode,
    "hardcore": HardcoreMode,
    "reverse": ReverseMode,
}


def register_mode(name: str, factory: ModeFactory) -> None:
    """Register a pluggable GameMode factory without modifying engine.py."""
    _MODE_REGISTRY[name.strip().lower()] = factory


def get_mode(mode: GameMode | str | None) -> GameMode:
    """Resolve a mode instance by object or registered name."""
    if mode is None:
        return ClassicMode()
    if isinstance(mode, GameMode):
        return mode
    key = mode.strip().lower()
    factory = _MODE_REGISTRY.get(key)
    if factory is None:
        available = ", ".join(sorted(_MODE_REGISTRY))
        raise ModeNotFoundError(f"Unknown mode {mode!r}. Available modes: {available}.")
    return factory()


def list_modes() -> list[GameMode]:
    """Instantiate and return all registered game modes."""
    return [factory() for factory in _MODE_REGISTRY.values()]
