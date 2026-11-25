"""Tests for the headless Guess Arena game engine (MVP 1)."""

from __future__ import annotations

import pytest

from arena.engine import Game, GameConfig
from arena.exceptions import ArenaError, GameConfigError, GameOverError, InvalidGuessError
from arena.modes import GuessOutcome


def test_full_headless_game_win() -> None:
    game = Game(GameConfig(low=1, high=100, max_tries=10, secret=64))

    assert game.guess(50) is GuessOutcome.HIGHER
    assert game.state.current_low == 51
    assert game.state.current_high == 100

    assert game.guess(75) is GuessOutcome.LOWER
    assert game.state.current_high == 74

    assert game.guess(64) is GuessOutcome.CORRECT
    assert game.state.game_over is True
    assert game.state.won is True
    assert game.reveal_secret() == 64

    summary = game.summary()
    assert summary.attempts == 3
    assert summary.won is True
    assert summary.score > 0


def test_game_loss_when_max_tries_exhausted() -> None:
    game = Game(GameConfig(low=1, high=10, max_tries=2, secret=9))
    assert game.guess(1) is GuessOutcome.HIGHER
    res = game.submit_guess(2)
    assert res.game_over is True
    assert res.won is False
    assert res.score == 0

    with pytest.raises(GameOverError):
        game.guess(9)


def test_config_and_guess_validation() -> None:
    with pytest.raises(GameConfigError):
        GameConfig(low=10, high=10)
    with pytest.raises(GameConfigError):
        GameConfig(low=1, high=10, max_tries=0)
    with pytest.raises(GameConfigError):
        GameConfig(low=1, high=10, secret=11)
    with pytest.raises(GameConfigError):
        GameConfig.from_difficulty("impossible")

    easy = GameConfig.from_difficulty("easy")
    assert (easy.low, easy.high, easy.max_tries) == (1, 50, 8)

    game = Game(GameConfig(low=1, high=10, secret=5))
    with pytest.raises(ArenaError):
        game.reveal_secret()
    with pytest.raises(InvalidGuessError):
        game.guess(0)
    with pytest.raises(InvalidGuessError):
        game.guess(11)
    with pytest.raises(InvalidGuessError):
        game.guess(True)  # type: ignore[arg-type]
