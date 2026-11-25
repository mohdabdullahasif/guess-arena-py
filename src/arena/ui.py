"""Rich + plotext terminal UI for Guess Arena (MVP 5)."""

from __future__ import annotations

from collections.abc import Sequence

import plotext as plt
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from arena.engine import GameState, GameSummary, GuessResult
from arena.modes import GuessOutcome
from arena.players import SimulationReport
from arena.storage import LeaderboardEntry, PlayerStats


def render_range_bar(
    low: int,
    high: int,
    current_low: int,
    current_high: int,
    *,
    width: int = 36,
) -> Text:
    """Render a colored shrinking range indicator bar."""
    total_span = max(1, high - low + 1)
    active_span = max(0, current_high - current_low + 1)

    start_pos = int(((current_low - low) / total_span) * width)
    end_pos = max(start_pos + 1, int(((current_high - low + 1) / total_span) * width))
    end_pos = min(width, end_pos)

    bar = Text()
    bar.append(f"[{low}] ", style="dim")
    if start_pos > 0:
        bar.append("░" * start_pos, style="dim red")
    bar.append("█" * max(1, end_pos - start_pos), style="bold cyan")
    if end_pos < width:
        bar.append("░" * (width - end_pos), style="dim red")
    bar.append(f" [{high}]", style="dim")
    bar.append(f"  active: {current_low}..{current_high} ({active_span}/{total_span})", style="bold")
    return bar


def render_game_status(state: GameState, *, player_name: str) -> Panel:
    """Build a Rich status panel showing mode, tries, elapsed time, and active range bar."""
    tries_str = (
        f"{state.attempts}/{state.max_tries}"
        if state.max_tries is not None
        else f"{state.attempts} (unlimited)"
    )
    body = Text()
    body.append("Player: ", style="dim")
    body.append(player_name, style="bold white")
    body.append("  |  Mode: ", style="dim")
    body.append(state.mode_name, style="bold magenta")
    body.append("  |  Difficulty: ", style="dim")
    body.append(state.difficulty, style="bold yellow")
    body.append(f"\nTries: {tries_str}  |  Elapsed: {state.elapsed_seconds:.1f}s\n\n")
    body.append_text(
        render_range_bar(state.low, state.high, state.current_low, state.current_high)
    )
    return Panel(body, title="Guess Arena", border_style="cyan")


def render_guess_history_table(
    history: Sequence[GuessResult],
    *,
    title: str = "Guess History",
) -> Table:
    """Render a Rich table of guesses, outcomes, hints, and narrowed bounds."""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Guess", justify="right", style="bold white")
    table.add_column("Outcome", no_wrap=True)
    table.add_column("Hint")
    table.add_column("Active Range", justify="center", style="cyan")
    table.add_column("Time (s)", justify="right", style="dim")

    outcome_styles = {
        GuessOutcome.HIGHER: "cyan",
        GuessOutcome.LOWER: "magenta",
        GuessOutcome.CORRECT: "bold green",
        GuessOutcome.HINT: "yellow",
    }

    for entry in history:
        style = outcome_styles.get(entry.outcome, "white")
        table.add_row(
            str(entry.attempt),
            str(entry.guess),
            Text(entry.outcome.value, style=style),
            entry.hint,
            f"{entry.current_low}..{entry.current_high}",
            f"{entry.elapsed_seconds:.2f}",
        )
    return table


def render_round_summary(summary: GameSummary, *, console: Console) -> None:
    """Print the end-of-round summary panel and guess history."""
    if summary.history:
        console.print(render_guess_history_table(summary.history))
    status = "[bold green]VICTORY![/bold green]" if summary.won else "[bold red]DEFEAT[/bold red]"
    border = "green" if summary.won else "red"
    console.print(
        Panel(
            f"{status}\n"
            f"Secret Number: [bold]{summary.secret}[/bold]\n"
            f"Attempts: [bold]{summary.attempts}[/bold]  |  "
            f"Score: [bold yellow]{summary.score}[/bold yellow]  |  "
            f"Time: [bold]{summary.duration_seconds:.2f}s[/bold]",
            title="Round Summary",
            border_style=border,
        )
    )


def build_terminal_bar_chart(
    labels: Sequence[str],
    values: Sequence[float | int],
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
) -> Panel:
    """Build a plotext bar chart safely wrapped inside a Rich Panel."""
    if hasattr(plt, "figure") and not hasattr(plt, "clf"):
        fig = plt.figure
        fig.clear()
        fig.plot_size(62, 14)
        fig.theme("colorless")
        sig = fig.bar(list(labels), list(values), width=0.6)
        fig.draw(sig)
        fig.title(title)
        if xlabel:
            fig.label(xlabel, "x")
        if ylabel:
            fig.label(ylabel, "y")
        rendered = str(fig.build())
    else:
        plt.clf()
        plt.plotsize(62, 14)
        plt.theme("clear")
        plt.bar(list(labels), list(values), width=0.6)
        plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        rendered = str(plt.build())
    return Panel(Text.from_ansi(rendered), title=title, border_style="blue")


def render_stats(stats: PlayerStats, *, console: Console) -> None:
    """Render player stats, per-difficulty win streaks, and terminal tries chart."""
    if stats.total_games == 0:
        console.print(
            Panel(
                f"No recorded games for [bold]{stats.player_name}[/bold] yet.",
                title="Player Stats",
                border_style="yellow",
            )
        )
        return

    summary_panel = Panel(
        f"Player: [bold]{stats.player_name}[/bold] "
        f"({'[cyan]BOT[/cyan]' if stats.is_bot else '[green]HUMAN[/green]'})\n"
        f"Games: [bold]{stats.total_games}[/bold]  |  "
        f"Wins: [bold]{stats.wins}[/bold] ({stats.win_rate * 100:.1f}%)\n"
        f"Avg Tries: [bold]{stats.avg_tries:.2f}[/bold]  |  "
        f"Best Streak: [bold]{stats.best_streak}[/bold]\n"
        f"Best Score: [bold yellow]{stats.best_score}[/bold yellow]  |  "
        f"Avg Score: [bold]{stats.avg_score:.1f}[/bold]",
        title="Player Statistics",
        border_style="cyan",
    )
    console.print(summary_panel)

    if stats.streaks_by_difficulty:
        diff_table = Table(title="Best Streak by Difficulty", box=box.SIMPLE)
        diff_table.add_column("Difficulty", style="bold yellow")
        diff_table.add_column("Best Streak", justify="right", style="bold green")
        for diff, streak in stats.streaks_by_difficulty.items():
            diff_table.add_row(diff, str(streak))
        console.print(diff_table)

    if stats.tries_distribution:
        labels = [str(k) for k in stats.tries_distribution]
        values = list(stats.tries_distribution.values())
        console.print(
            build_terminal_bar_chart(
                labels,
                values,
                title="Tries Distribution",
                xlabel="Attempts",
                ylabel="Games",
            )
        )


def render_leaderboard(
    entries: Sequence[LeaderboardEntry],
    *,
    console: Console,
    title: str = "Leaderboard",
) -> None:
    """Render ranked leaderboard table."""
    if not entries:
        console.print("[dim]No leaderboard entries recorded yet.[/dim]")
        return

    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Player", style="bold white", no_wrap=True)
    table.add_column("Type", no_wrap=True)
    table.add_column("Games", justify="right")
    table.add_column("Win %", justify="right", style="green")
    table.add_column("Avg Tries", justify="right", style="cyan")
    table.add_column("Best Streak", justify="right", style="magenta")
    table.add_column("Best Score", justify="right", style="yellow")
    table.add_column("Avg Score", justify="right")

    for entry in entries:
        kind = Text("BOT", style="cyan") if entry.is_bot else Text("HUMAN", style="green")
        table.add_row(
            f"#{entry.rank}",
            entry.player_name,
            kind,
            str(entry.total_games),
            f"{entry.win_rate * 100:.1f}%",
            f"{entry.avg_tries:.2f}",
            str(entry.best_streak),
            str(entry.best_score),
            f"{entry.avg_score:.1f}",
        )
    console.print(table)


def render_simulation_comparison(
    reports: Sequence[SimulationReport],
    *,
    console: Console,
) -> None:
    """Display bot simulation results table and plotext comparison chart."""
    table = Table(title="Simulation Results", box=box.ROUNDED)
    table.add_column("Bot", style="bold white", no_wrap=True)
    table.add_column("Games", justify="right")
    table.add_column("Win Rate", justify="right", style="green")
    table.add_column("Min Tries", justify="right")
    table.add_column("Avg Tries", justify="right", style="bold cyan")
    table.add_column("Max Tries", justify="right")
    table.add_column("log2 Bound", justify="right", style="yellow")

    for report in reports:
        table.add_row(
            report.player_name,
            str(report.games_played),
            f"{report.win_rate * 100:.2f}%",
            str(report.min_tries),
            f"{report.avg_tries:.3f}",
            str(report.max_tries),
            str(report.theoretical_max_tries),
        )
    console.print(table)

    if reports:
        console.print(
            build_terminal_bar_chart(
                [r.player_name for r in reports],
                [round(r.avg_tries, 2) for r in reports],
                title="Average Tries by Bot",
                ylabel="Avg Tries",
            )
        )
