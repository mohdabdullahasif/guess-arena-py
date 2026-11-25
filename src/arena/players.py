"""Player abstraction, AI bots, simulation proof, and ML-flavored LearningBot (MVP 2 & 7)."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from arena.engine import Game, GameConfig, GameState
from arena.exceptions import ArenaError
from arena.modes import GameMode

# Empirically common human bias profile for 1..100 (lucky numbers, primes ending in 7/3, midpoint 50).
HUMAN_FAVORED_NUMBERS: dict[int, float] = {
    7: 18.0,
    13: 10.0,
    17: 11.0,
    21: 6.0,
    23: 9.0,
    27: 8.0,
    33: 6.0,
    37: 16.0,
    42: 14.0,
    47: 8.0,
    50: 14.0,
    57: 6.0,
    63: 5.0,
    67: 8.0,
    69: 11.0,
    73: 15.0,
    77: 12.0,
    88: 7.0,
    99: 6.0,
}


def default_human_prior(low: int, high: int) -> dict[int, float]:
    """Return prior weights modeling common human number-selection biases."""
    weights: dict[int, float] = {}
    span = high - low + 1
    for num in range(low, high + 1):
        weight = 1.0
        # Slight edge-avoidance when span >= 10
        if span >= 10 and (num <= low + 1 or num >= high - 1):
            weight *= 0.4
        if num in HUMAN_FAVORED_NUMBERS:
            weight = HUMAN_FAVORED_NUMBERS[num]
        elif num % 10 == 7:
            weight = max(weight, 4.0)
        elif num % 10 == 3:
            weight = max(weight, 2.5)
        elif num % 10 == 0 and num != low and num != high:
            weight = max(weight, 2.0)
        weights[num] = weight
    return weights


def sample_human_biased_secrets(
    count: int,
    *,
    low: int = 1,
    high: int = 100,
    seed: int = 42,
    weights: Mapping[int, float] | None = None,
) -> list[int]:
    """Sample realistic human-biased secret numbers for benchmarking LearningBot."""
    rng = random.Random(seed)
    prior = dict(weights) if weights is not None else default_human_prior(low, high)
    population = list(range(low, high + 1))
    pop_weights = [max(0.01, float(prior.get(n, 1.0))) for n in population]
    return rng.choices(population, weights=pop_weights, k=count)


class Player(ABC):
    """Common interface for humans and AI bots."""

    def __init__(self, name: str, *, is_bot: bool = True) -> None:
        self.name = name
        self.is_bot = is_bot

    @abstractmethod
    def next_guess(self, state: GameState) -> int:
        """Choose the next guess given the current secret-free GameState."""


class HumanPlayer(Player):
    """Interactive human player backed by an injectable prompt callback."""

    def __init__(
        self,
        name: str = "Player",
        *,
        prompt_fn: Callable[[GameState], int] | None = None,
    ) -> None:
        super().__init__(name=name, is_bot=False)
        self._prompt_fn = prompt_fn

    def next_guess(self, state: GameState) -> int:
        if self._prompt_fn is not None:
            return self._prompt_fn(state)
        raw = input(f"Guess a number [{state.current_low}-{state.current_high}]: ").strip()
        try:
            return int(raw)
        except ValueError as exc:
            raise ArenaError(f"Invalid integer input: {raw!r}") from exc


class RandomBot(Player):
    """Bot that guesses uniformly at random among remaining untried candidates."""

    def __init__(self, name: str = "RandomBot", *, seed: int | None = None) -> None:
        super().__init__(name=name, is_bot=True)
        self._rng = random.Random(seed)

    def next_guess(self, state: GameState) -> int:
        tried = {entry.guess for entry in state.history}
        candidates = [
            n
            for n in range(state.current_low, state.current_high + 1)
            if n not in tried
        ]
        if not candidates:
            return self._rng.randint(state.current_low, state.current_high)
        return self._rng.choice(candidates)


class BinarySearchBot(Player):
    """Optimal worst-case binary search opponent guaranteeing <= ceil(log2(N)) tries."""

    def __init__(self, name: str = "BinarySearchBot") -> None:
        super().__init__(name=name, is_bot=True)

    def next_guess(self, state: GameState) -> int:
        tried = {entry.guess for entry in state.history}
        mid = (state.current_low + state.current_high) // 2
        if mid not in tried:
            return mid
        # Fallback for non-directional modes (Hot/Cold or Hardcore) where bounds don't narrow.
        for offset in range(1, state.high - state.low + 1):
            if mid + offset <= state.current_high and (mid + offset) not in tried:
                return mid + offset
            if mid - offset >= state.current_low and (mid - offset) not in tried:
                return mid - offset
        return mid


_SHARED_DP_CACHE: dict[tuple[int, int, float, tuple[tuple[int, float], ...]], list[list[int]]] = {}


class LearningBot(Player):
    """Bias-exploiting ML bot using Bayesian smoothing + optimal expected-tries DP."""

    def __init__(
        self,
        name: str = "LearningBot",
        *,
        low: int = 1,
        high: int = 100,
        smoothing: float = 0.5,
        prior_weights: Mapping[int, float] | None = None,
    ) -> None:
        super().__init__(name=name, is_bot=True)
        self.low = low
        self.high = high
        self.smoothing = smoothing
        self._base_prior: dict[int, float] = (
            dict(prior_weights) if prior_weights is not None else default_human_prior(low, high)
        )
        self._observed_counts: Counter[int] = Counter()

    def fit(self, counts: Mapping[int, int], *, replace_counts: bool = True) -> None:
        """Fit or update posterior weights from historical human-picked numbers."""
        if replace_counts:
            self._observed_counts = Counter({int(k): int(v) for k, v in counts.items() if v > 0})
        else:
            for k, v in counts.items():
                if v > 0:
                    self._observed_counts[int(k)] += int(v)

    def observe(self, secret: int) -> None:
        """Online update after a round completes."""
        self._observed_counts[secret] += 1

    def _weight_for(self, n: int) -> float:
        if self._observed_counts:
            # Blend empirical human picks with small smoothed prior so unseen values remain valid.
            return float(self._observed_counts.get(n, 0)) + self.smoothing * (
                self._base_prior.get(n, 1.0) / 10.0
            )
        return max(0.05, float(self._base_prior.get(n, 1.0)))

    def _get_dp_table(self, low: int, high: int) -> list[list[int]]:
        if self._observed_counts:
            sig = tuple(sorted((k, float(v)) for k, v in self._observed_counts.items()))
        else:
            sig = tuple(sorted((k, float(v)) for k, v in self._base_prior.items()))
        cache_key = (low, high, self.smoothing, sig)
        cached = _SHARED_DP_CACHE.get(cache_key)
        if cached is not None:
            return cached
        table = self._build_dp_table(low, high)
        if len(_SHARED_DP_CACHE) < 64:
            _SHARED_DP_CACHE[cache_key] = table
        return table

    def _build_dp_table(self, low: int, high: int) -> list[list[int]]:
        """Compute Bayes-optimal guess root[i][j] minimizing expected tries for [low..high]."""
        n = high - low + 1
        weights = [self._weight_for(low + idx) for idx in range(n)]
        prefix = [0.0] * (n + 1)
        for idx, w in enumerate(weights):
            prefix[idx + 1] = prefix[idx] + w

        cost: list[list[float]] = [[0.0] * n for _ in range(n)]
        root: list[list[int]] = [[0] * n for _ in range(n)]

        for i in range(n):
            cost[i][i] = weights[i]
            root[i][i] = i

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                total_w = prefix[j + 1] - prefix[i]
                best_cost = float("inf")
                best_r = i
                for r in range(i, j + 1):
                    left_cost = cost[i][r - 1] if r > i else 0.0
                    right_cost = cost[r + 1][j] if r < j else 0.0
                    candidate = total_w + left_cost + right_cost
                    if candidate < best_cost:
                        best_cost = candidate
                        best_r = r
                cost[i][j] = best_cost
                root[i][j] = best_r

        return root

    def _weighted_median_guess(self, low: int, high: int, tried: set[int]) -> int:
        candidates = [n for n in range(low, high + 1) if n not in tried]
        if not candidates:
            return (low + high) // 2
        weights = [self._weight_for(n) for n in candidates]
        half = sum(weights) / 2.0
        running = 0.0
        for num, weight in zip(candidates, weights):
            running += weight
            if running >= half:
                return num
        return candidates[-1]

    def next_guess(self, state: GameState) -> int:
        tried = {entry.guess for entry in state.history}
        span = state.high - state.low + 1

        if (
            span <= 150
            and state.current_low >= state.low
            and state.current_high <= state.high
        ):
            table = self._get_dp_table(state.low, state.high)
            i = state.current_low - state.low
            j = state.current_high - state.low
            if 0 <= i <= j < span:
                pick = state.low + table[i][j]
                if pick not in tried:
                    return pick

        return self._weighted_median_guess(state.current_low, state.current_high, tried)


@dataclass(frozen=True, slots=True)
class SimulationReport:
    """Aggregated result of running a bot across many simulated games."""

    player_name: str
    games_played: int
    wins: int
    win_rate: float
    min_tries: int
    max_tries: int
    avg_tries: float
    theoretical_max_tries: int
    tries_histogram: dict[int, int] = field(default_factory=dict)


def simulate_games(
    player_factory: Callable[[], Player],
    *,
    num_games: int = 10_000,
    config: GameConfig | None = None,
    mode: GameMode | str = "classic",
    secrets: Sequence[int] | None = None,
) -> SimulationReport:
    """Run num_games headless games and summarize tries/win statistics."""
    if num_games <= 0:
        raise ArenaError("num_games must be positive.")
    base_config = config or GameConfig()
    if secrets is not None and len(secrets) < num_games:
        raise ArenaError("secrets sequence must be at least as long as num_games.")

    rng = random.Random(base_config.seed)
    tries_list: list[int] = []
    wins = 0
    histogram: Counter[int] = Counter()
    player_name = ""

    for game_idx in range(num_games):
        player = player_factory()
        if not player_name:
            player_name = player.name
        chosen_secret = (
            int(secrets[game_idx])
            if secrets is not None
            else rng.randint(base_config.low, base_config.high)
        )
        round_cfg = GameConfig(
            low=base_config.low,
            high=base_config.high,
            max_tries=base_config.max_tries,
            difficulty=base_config.difficulty,
            secret=chosen_secret,
            time_limit_seconds=base_config.time_limit_seconds,
        )
        game = Game(round_cfg, mode=mode)
        while not game.state.game_over:
            guess_val = player.next_guess(game.state)
            game.submit_guess(guess_val)
        summary = game.summary()
        if summary.won:
            wins += 1
        tries_list.append(summary.attempts)
        histogram[summary.attempts] += 1

    theoretical_max = base_config.span.bit_length()
    return SimulationReport(
        player_name=player_name,
        games_played=num_games,
        wins=wins,
        win_rate=wins / num_games,
        min_tries=min(tries_list),
        max_tries=max(tries_list),
        avg_tries=sum(tries_list) / num_games,
        theoretical_max_tries=theoretical_max,
        tries_histogram=dict(sorted(histogram.items())),
    )
