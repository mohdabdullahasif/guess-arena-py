"""Tests for SQLite stats, leaderboards, and Matplotlib analytics (MVP 3 & MVP 7)."""

from __future__ import annotations

from pathlib import Path

from arena.analytics import generate_analytics_report
from arena.engine import Game, GameConfig
from arena.players import BinarySearchBot
from arena.storage import ArenaStorage


def _play_fixed_game(
    secret: int,
    guesses: list[int],
    *,
    difficulty: str = "medium",
    mode: str = "classic",
    max_tries: int | None = 10,
) -> Game:
    low, high = (1, 50) if difficulty == "easy" else (1, 100)
    game = Game(
        GameConfig(low=low, high=high, max_tries=max_tries, difficulty=difficulty, secret=secret),
        mode=mode,
    )
    for g in guesses:
        if game.state.game_over:
            break
        game.submit_guess(g)
    return game


def test_sqlite_persistence_stats_streaks_and_leaderboard(tmp_path: Path) -> None:
    db_file = tmp_path / "arena.db"
    storage = ArenaStorage(db_file)

    # Alice: 2 wins on easy, 1 loss + 1 win on medium
    storage.record_game("Alice", _play_fixed_game(10, [25, 10], difficulty="easy").summary())
    storage.record_game("Alice", _play_fixed_game(20, [20], difficulty="easy").summary())
    storage.record_game("Alice", _play_fixed_game(90, [1, 2], difficulty="medium", max_tries=2).summary())
    storage.record_game("Alice", _play_fixed_game(50, [50], difficulty="medium").summary())

    # Bot records
    bot = BinarySearchBot()
    for secret in (17, 73):
        g = Game(GameConfig(low=1, high=100, difficulty="medium", secret=secret))
        while not g.state.game_over:
            g.submit_guess(bot.next_guess(g.state))
        storage.record_game(bot.name, g.summary(), is_bot=True)

    # Reopen from disk to prove persistence across restarts.
    reopened = ArenaStorage(db_file)
    alice_stats = reopened.get_player_stats("Alice")
    assert alice_stats.total_games == 4
    assert alice_stats.wins == 3
    assert alice_stats.win_rate == 0.75
    assert alice_stats.best_streak == 2
    assert alice_stats.streaks_by_difficulty == {"easy": 2, "medium": 1}

    freq = reopened.get_human_guess_frequencies(low=1, high=100)
    assert freq[25] == 1
    assert freq[50] == 1

    board = reopened.get_leaderboard(limit=10)
    assert len(board) == 2
    assert board[0].player_name == "BinarySearchBot"
    assert board[0].rank == 1
    assert board[1].player_name == "Alice"


def test_matplotlib_analytics_report_generation(tmp_path: Path) -> None:
    storage = ArenaStorage(tmp_path / "analytics.db")
    out_png = tmp_path / "report.png"

    summary = generate_analytics_report(
        storage,
        output_path=out_png,
        low=1,
        high=100,
        simulation_games=300,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 1000
    assert summary.used_simulated_baseline is True

    by_name = {r.player_name: r for r in summary.bot_reports}
    assert by_name["LearningBot"].avg_tries < by_name["BinarySearchBot"].avg_tries
