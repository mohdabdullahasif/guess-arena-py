"""Tests for Player interface, 10k-game simulation proof, and LearningBot (MVP 2 & MVP 7)."""

from __future__ import annotations

import math

from arena.engine import Game, GameConfig
from arena.players import (
    BinarySearchBot,
    HumanPlayer,
    LearningBot,
    RandomBot,
    sample_human_biased_secrets,
    simulate_games,
)


def test_human_player_with_injected_prompt() -> None:
    answers = iter([50, 75])
    human = HumanPlayer("Tester", prompt_fn=lambda _: next(answers))
    game = Game(GameConfig(low=1, high=100, secret=75))
    while not game.state.game_over:
        game.submit_guess(human.next_guess(game.state))
    assert game.state.won is True
    assert game.state.attempts == 2


def test_binary_search_bot_10000_games_log2_proof() -> None:
    cfg = GameConfig(low=1, high=100, seed=123)
    report = simulate_games(BinarySearchBot, num_games=10_000, config=cfg)

    assert report.games_played == 10_000
    assert report.wins == 10_000
    assert report.win_rate == 1.0
    assert report.max_tries <= math.ceil(math.log2(100))
    assert report.max_tries <= report.theoretical_max_tries

    # Also prove log2 bound on 1..1000.
    hard_cfg = GameConfig(low=1, high=1000, seed=99)
    hard_report = simulate_games(BinarySearchBot, num_games=1000, config=hard_cfg)
    assert hard_report.win_rate == 1.0
    assert hard_report.max_tries <= math.ceil(math.log2(1000))


def test_random_bot_plays_within_bounds() -> None:
    cfg = GameConfig(low=1, high=50, seed=7)
    report = simulate_games(lambda: RandomBot(seed=7), num_games=200, config=cfg)
    assert report.wins == 200
    assert report.avg_tries > 4.0


def test_learning_bot_beats_binary_search_on_human_biased_numbers() -> None:
    secrets = sample_human_biased_secrets(2000, low=1, high=100, seed=42)
    cfg = GameConfig(low=1, high=100, seed=42)

    binary_report = simulate_games(BinarySearchBot, num_games=2000, config=cfg, secrets=secrets)
    learning_report = simulate_games(
        lambda: LearningBot(low=1, high=100),
        num_games=2000,
        config=cfg,
        secrets=secrets,
    )

    assert learning_report.win_rate == 1.0
    assert learning_report.avg_tries < binary_report.avg_tries


def test_learning_bot_fit_observe_and_large_range_fallback() -> None:
    bot = LearningBot(low=1, high=300)
    bot.fit({250: 100, 10: 5})
    bot.observe(250)
    game = Game(GameConfig(low=1, high=300, secret=250))
    while not game.state.game_over:
        game.submit_guess(bot.next_guess(game.state))
    assert game.state.won is True
    assert game.state.attempts <= 3
