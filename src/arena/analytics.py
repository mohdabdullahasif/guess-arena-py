"""Matplotlib analytics and guess pattern reports for Guess Arena (MVP 7)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arena.engine import GameConfig
from arena.players import (
    BinarySearchBot,
    LearningBot,
    RandomBot,
    SimulationReport,
    sample_human_biased_secrets,
    simulate_games,
)
from arena.storage import ArenaStorage


@dataclass(frozen=True, slots=True)
class AnalyticsSummary:
    """Summary returned after generating the Matplotlib analytics report."""

    output_path: Path
    human_samples: int
    used_simulated_baseline: bool
    top_numbers: list[tuple[int, int]]
    bot_reports: list[SimulationReport]


def generate_analytics_report(
    storage: ArenaStorage,
    output_path: str | Path = "guess_patterns.png",
    *,
    low: int = 1,
    high: int = 100,
    simulation_games: int = 2000,
) -> AnalyticsSummary:
    """Generate a two-panel Matplotlib report on human guess biases and bot performance."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    guess_counts = storage.get_human_guess_frequencies(low=low, high=high)
    secret_counts = storage.get_human_secret_frequencies(low=low, high=high)
    combined: Counter[int] = Counter(guess_counts)
    combined.update(secret_counts)

    used_simulated = False
    if sum(combined.values()) < 25:
        used_simulated = True
        simulated = sample_human_biased_secrets(500, low=low, high=high, seed=2026)
        combined.update(simulated)

    total_samples = sum(combined.values())
    top_numbers = combined.most_common(10)

    # Benchmark bots on human-biased numbers learned from combined human frequency distribution.
    human_secrets = sample_human_biased_secrets(
        simulation_games,
        low=low,
        high=high,
        seed=99,
        weights={n: max(0.2, float(combined.get(n, 0))) for n in range(low, high + 1)},
    )
    cfg = GameConfig(low=low, high=high, seed=99)

    def make_learning_bot() -> LearningBot:
        bot = LearningBot(low=low, high=high)
        bot.fit(combined)
        return bot

    reports = [
        simulate_games(lambda: RandomBot(seed=99), num_games=simulation_games, config=cfg, secrets=human_secrets),
        simulate_games(BinarySearchBot, num_games=simulation_games, config=cfg, secrets=human_secrets),
        simulate_games(make_learning_bot, num_games=simulation_games, config=cfg, secrets=human_secrets),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    xs = list(range(low, high + 1))
    ys = [combined.get(n, 0) for n in xs]
    uniform_baseline = total_samples / max(1, len(xs))
    top_set = {num for num, _ in top_numbers[:5]}
    colors = ["#ef4444" if n in top_set else "#3b82f6" for n in xs]

    ax1.bar(xs, ys, color=colors, width=0.8)
    ax1.axhline(uniform_baseline, color="#10b981", linestyle="--", linewidth=1.5, label="Uniform avg")
    ax1.set_title("Human Number Selection Pattern" + (" (DB + Prior Sample)" if used_simulated else ""))
    ax1.set_xlabel("Number")
    ax1.set_ylabel("Pick Frequency")
    ax1.legend()

    bot_names = [r.player_name for r in reports]
    avg_tries = [r.avg_tries for r in reports]
    bot_colors = ["#94a3b8", "#3b82f6", "#10b981"]
    bars = ax2.bar(bot_names, avg_tries, color=bot_colors, width=0.5)
    ax2.set_title(f"Bot Comparison on Human-Biased Numbers (n={simulation_games})")
    ax2.set_ylabel("Average Tries (lower is better)")
    for bar, val in zip(bars, avg_tries):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)

    return AnalyticsSummary(
        output_path=target,
        human_samples=total_samples,
        used_simulated_baseline=used_simulated,
        top_numbers=top_numbers,
        bot_reports=reports,
    )
