"""Tests for pluggable game modes and Reverse mode (MVP 4)."""

from __future__ import annotations

from datetime import date

import pytest

from arena.engine import Game, GameConfig, ReverseSession
from arena.exceptions import InconsistentFeedbackError
from arena.modes import (
    DailyChallengeMode,
    GameMode,
    GuessOutcome,
    ModeFeedback,
    register_mode,
)
from arena.players import BinarySearchBot


def test_timed_mode_score_decays_with_time() -> None:
    fake_clock = [0.0]

    def clock() -> float:
        return fake_clock[0]

    fast_game = Game(GameConfig(low=1, high=100, secret=50), mode="timed", clock=clock)
    fast_game.submit_guess(50)
    fast_score = fast_game.summary().score

    fake_clock[0] = 0.0
    slow_game = Game(GameConfig(low=1, high=100, secret=50), mode="timed", clock=clock)
    fake_clock[0] = 20.0
    slow_game.submit_guess(50)
    slow_score = slow_game.summary().score

    assert fast_score > slow_score

    # Time limit enforcement
    fake_clock[0] = 0.0
    limited = Game(
        GameConfig(low=1, high=100, secret=50, time_limit_seconds=5.0),
        mode="timed",
        clock=clock,
    )
    fake_clock[0] = 10.0
    res = limited.submit_guess(50)
    assert res.won is False
    assert res.game_over is True


def test_hot_cold_mode_gives_distance_without_narrowing() -> None:
    game = Game(GameConfig(low=1, high=100, secret=50), mode="hot_cold")
    r1 = game.submit_guess(10)
    assert r1.outcome is GuessOutcome.HINT
    assert r1.distance == 40
    assert r1.temperature == "COLD"
    assert r1.current_low == 1
    assert r1.current_high == 100

    r2 = game.submit_guess(48)
    assert r2.distance == 2
    assert r2.temperature == "BOILING"
    assert "getting warmer" in r2.hint

    r3 = game.submit_guess(50)
    assert r3.outcome is GuessOutcome.CORRECT
    assert r3.won is True


def test_daily_challenge_determinism() -> None:
    d1 = DailyChallengeMode(challenge_date=date(2026, 11, 26))
    d2 = DailyChallengeMode(challenge_date=date(2026, 11, 26))
    d3 = DailyChallengeMode(challenge_date=date(2026, 11, 27))

    g1 = Game(GameConfig(low=1, high=1000), mode=d1)
    g2 = Game(GameConfig(low=1, high=1000), mode=d2)
    g3 = Game(GameConfig(low=1, high=1000), mode=d3)

    for g in (g1, g2, g3):
        bot = BinarySearchBot()
        while not g.state.game_over:
            g.submit_guess(bot.next_guess(g.state))

    assert g1.reveal_secret() == g2.reveal_secret()
    assert g1.reveal_secret() != g3.reveal_secret()


def test_hardcore_mode_no_hints_and_max_tries() -> None:
    game = Game(GameConfig(low=1, high=100, secret=99), mode="hardcore")
    assert game.config.max_tries == 6
    res = game.submit_guess(1)
    assert res.outcome is GuessOutcome.HINT
    assert "No hints" in res.hint
    assert res.current_low == 1


def test_reverse_mode_and_contradiction_detection() -> None:
    session = ReverseSession(GameConfig(low=1, high=100, max_tries=10))
    bot = BinarySearchBot()
    human_secret = 73

    while not session.state.game_over:
        guess = bot.next_guess(session.state)
        feedback = session.verify_against_secret(guess, human_secret)
        session.apply_feedback(guess, feedback)

    assert session.state.won is True
    assert session.state.attempts <= 7

    # Contradictory feedback raises InconsistentFeedbackError
    bad_session = ReverseSession(GameConfig(low=1, high=10))
    bad_session.apply_feedback(5, GuessOutcome.HIGHER)  # bounds now [6..10]
    with pytest.raises(InconsistentFeedbackError):
        bad_session.apply_feedback(6, GuessOutcome.LOWER)


def test_custom_pluggable_mode_without_touching_engine() -> None:
    class ParityHintMode(GameMode):
        name = "parity_test"
        description = "Reveals whether secret is even or odd."

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
                return ModeFeedback(outcome=GuessOutcome.CORRECT, hint="Exact!")
            parity = "even" if secret % 2 == 0 else "odd"
            outcome = GuessOutcome.HIGHER if guess < secret else GuessOutcome.LOWER
            return ModeFeedback(outcome=outcome, hint=f"Secret is {parity}")

    register_mode("parity_test", ParityHintMode)
    game = Game(GameConfig(low=1, high=10, secret=4), mode="parity_test")
    res = game.submit_guess(3)
    assert res.outcome is GuessOutcome.HIGHER
    assert res.hint == "Secret is even"
