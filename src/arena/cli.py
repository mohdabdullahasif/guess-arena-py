"""Typer CLI entry point for Guess Arena."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arena.client import run_interactive_client
from arena.engine import Game, GameConfig, ReverseSession
from arena.exceptions import ArenaError
from arena.modes import GuessOutcome, get_mode, list_modes
from arena.players import (
    BinarySearchBot,
    LearningBot,
    RandomBot,
    sample_human_biased_secrets,
    simulate_games,
)
from arena.server import ArenaServer
from arena.storage import ArenaStorage, resolve_db_path
from arena.ui import (
    render_game_status,
    render_guess_history_table,
    render_leaderboard,
    render_round_summary,
    render_simulation_comparison,
    render_stats,
)

app = typer.Typer(
    name="arena",
    help="Guess Arena: multi-mode guessing game with AI bots, SQLite stats, Rich UI, and multiplayer.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def _storage(db: Optional[Path]) -> ArenaStorage:
    return ArenaStorage(resolve_db_path(db))


def _build_config(
    *,
    difficulty: str,
    low: Optional[int],
    high: Optional[int],
    max_tries: Optional[int],
    seed: Optional[int],
    time_limit: Optional[float],
) -> GameConfig:
    if difficulty.lower() != "custom" and low is None and high is None:
        return GameConfig.from_difficulty(
            difficulty,
            max_tries=max_tries,
            seed=seed,
            time_limit_seconds=time_limit,
        )
    return GameConfig(
        low=low if low is not None else 1,
        high=high if high is not None else 100,
        max_tries=max_tries,
        seed=seed,
        difficulty=difficulty.lower(),
        time_limit_seconds=time_limit,
    )


def _play_reverse_mode(
    *,
    cfg: GameConfig,
    bot_kind: str,
    player_name: str,
    repo: ArenaStorage,
) -> None:
    session = ReverseSession(config=cfg)
    if bot_kind == "learning":
        bot = LearningBot(low=cfg.low, high=cfg.high)
        counts = repo.get_human_secret_frequencies(low=cfg.low, high=cfg.high)
        if counts:
            bot.fit(counts)
    elif bot_kind == "random":
        bot = RandomBot(seed=cfg.seed)
    else:
        bot = BinarySearchBot()

    console.print(
        Panel(
            f"Think of a secret number between [bold]{cfg.low}[/bold] and [bold]{cfg.high}[/bold].\n"
            f"[bold]{bot.name}[/bold] will try to guess it. Reply with "
            "[cyan]higher[/cyan] (h), [magenta]lower[/magenta] (l), or [green]correct[/green] (c).",
            title="Reverse Mode",
            border_style="magenta",
        )
    )

    while not session.state.game_over:
        guess = bot.next_guess(session.state)
        console.print(f"\n[bold]{bot.name}[/bold] guesses: [bold yellow]{guess}[/bold yellow]")
        raw = typer.prompt("Feedback (h=higher / l=lower / c=correct)").strip().lower()
        mapping = {
            "h": GuessOutcome.HIGHER,
            "higher": GuessOutcome.HIGHER,
            "l": GuessOutcome.LOWER,
            "lower": GuessOutcome.LOWER,
            "c": GuessOutcome.CORRECT,
            "correct": GuessOutcome.CORRECT,
        }
        outcome = mapping.get(raw)
        if outcome is None:
            console.print("[red]Please enter h/higher, l/lower, or c/correct.[/red]")
            continue
        result = session.apply_feedback(guess, outcome)
        console.print(f"[dim]Active bounds: {result.current_low}..{result.current_high}[/dim]")

    if session.state.history:
        console.print(render_guess_history_table(session.state.history, title="Reverse Mode History"))
    if session.state.won:
        console.print(
            f"[bold green]{bot.name} found your number in {session.state.attempts} tries![/bold green]"
        )
    else:
        console.print(f"[bold yellow]{bot.name} ran out of tries![/bold yellow]")


@app.command()
def play(
    player: Annotated[str, typer.Option("--player", "-p", help="Player name for stats.")] = "Player",
    mode: Annotated[str, typer.Option("--mode", "-m", help="Game mode (classic, timed, hot_cold, daily, hardcore, reverse).")] = "classic",
    difficulty: Annotated[str, typer.Option("--difficulty", "-d", help="Difficulty: easy, medium, hard, custom.")] = "medium",
    low: Annotated[Optional[int], typer.Option("--low", help="Custom range minimum.")] = None,
    high: Annotated[Optional[int], typer.Option("--high", help="Custom range maximum.")] = None,
    max_tries: Annotated[Optional[int], typer.Option("--max-tries", help="Override max attempts.")] = None,
    seed: Annotated[Optional[int], typer.Option("--seed", help="Optional RNG seed.")] = None,
    time_limit: Annotated[Optional[float], typer.Option("--time-limit", help="Time limit in seconds (timed mode).")] = None,
    bot: Annotated[str, typer.Option("--bot", help="Bot opponent for reverse mode (binary, learning, random).")] = "binary",
    db: Annotated[Optional[Path], typer.Option("--db", help="Path to SQLite stats database.")] = None,
) -> None:
    """Play an interactive terminal round of Guess Arena."""
    try:
        repo = _storage(db)
        cfg = _build_config(
            difficulty=difficulty,
            low=low,
            high=high,
            max_tries=max_tries,
            seed=seed,
            time_limit=time_limit,
        )
        resolved_mode = get_mode(mode)
        if resolved_mode.is_reverse:
            _play_reverse_mode(cfg=cfg, bot_kind=bot.strip().lower(), player_name=player, repo=repo)
            return

        game = Game(cfg, mode=resolved_mode)
        while not game.state.game_over:
            console.print(render_game_status(game.state, player_name=player))
            raw = typer.prompt(
                f"Enter guess ({game.state.current_low}..{game.state.current_high}, or 'q' to quit)"
            ).strip()
            if raw.lower() in {"q", "quit", "exit"}:
                console.print("[yellow]Game aborted.[/yellow]")
                raise typer.Exit(code=0)
            try:
                guess_val = int(raw)
                res = game.submit_guess(guess_val)
            except ValueError:
                console.print("[red]Please enter a valid integer.[/red]")
                continue
            except ArenaError as exc:
                console.print(f"[red]{exc}[/red]")
                continue
            console.print(f"-> [bold]{res.hint}[/bold]")

        summary = game.summary()
        repo.record_game(player, summary, is_bot=False)
        render_round_summary(summary, console=console)
    except ArenaError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def simulate(
    bot: Annotated[str, typer.Option("--bot", "-b", help="Bot: random, binary, learning, or all.")] = "all",
    games: Annotated[int, typer.Option("--games", "-n", help="Number of games to simulate.")] = 10_000,
    low: Annotated[int, typer.Option("--low", help="Range minimum.")] = 1,
    high: Annotated[int, typer.Option("--high", help="Range maximum.")] = 100,
    seed: Annotated[int, typer.Option("--seed", help="RNG seed for reproducibility.")] = 42,
    biased: Annotated[bool, typer.Option("--biased", help="Simulate human-biased numbers instead of uniform random.")] = False,
    record: Annotated[bool, typer.Option("--record", help="Record bot runs to SQLite leaderboard.")] = False,
    db: Annotated[Optional[Path], typer.Option("--db", help="Path to SQLite stats database.")] = None,
) -> None:
    """Run headless bot simulations (MVP 2 & MVP 7 proof)."""
    try:
        cfg = GameConfig(low=low, high=high, seed=seed, difficulty="custom")
        secrets = (
            sample_human_biased_secrets(games, low=low, high=high, seed=seed)
            if biased
            else None
        )
        key = bot.strip().lower()
        targets = ["random", "binary", "learning"] if key == "all" else [key]

        factories = {
            "random": lambda: RandomBot(seed=seed),
            "binary": BinarySearchBot,
            "learning": lambda: LearningBot(low=low, high=high),
        }
        reports = []
        for target in targets:
            if target not in factories:
                raise ArenaError(f"Unknown bot {target!r}. Expected random, binary, learning, or all.")
            reports.append(
                simulate_games(
                    factories[target],
                    num_games=games,
                    config=cfg,
                    secrets=secrets,
                )
            )

        if record:
            repo = _storage(db)
            sample_count = min(50, games)
            for target in targets:
                factory = factories[target]
                for i in range(sample_count):
                    player = factory()
                    g = Game(GameConfig(low=low, high=high, seed=seed + i, difficulty="custom"))
                    while not g.state.game_over:
                        g.submit_guess(player.next_guess(g.state))
                    repo.record_game(player.name, g.summary(), is_bot=True)

        title_note = "Human-Biased Numbers" if biased else "Uniform Random Numbers"
        console.print(f"[bold]Distribution:[/bold] {title_note} ([{low}..{high}])")
        render_simulation_comparison(reports, console=console)
    except ArenaError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def stats(
    player: Annotated[str, typer.Argument(help="Player or bot name.")] = "Player",
    difficulty: Annotated[Optional[str], typer.Option("--difficulty", "-d", help="Filter by difficulty.")] = None,
    db: Annotated[Optional[Path], typer.Option("--db", help="Path to SQLite stats database.")] = None,
) -> None:
    """Show persistent statistics, streaks per difficulty, and tries chart."""
    repo = _storage(db)
    player_stats = repo.get_player_stats(player, difficulty=difficulty)
    render_stats(player_stats, console=console)


@app.command()
def leaderboard(
    difficulty: Annotated[Optional[str], typer.Option("--difficulty", "-d", help="Filter by difficulty.")] = None,
    mode: Annotated[Optional[str], typer.Option("--mode", "-m", help="Filter by mode.")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max rows to display.")] = 10,
    db: Annotated[Optional[Path], typer.Option("--db", help="Path to SQLite stats database.")] = None,
) -> None:
    """Rank humans and bots by win rate, score, and average tries."""
    repo = _storage(db)
    entries = repo.get_leaderboard(difficulty=difficulty, mode=mode, limit=limit)
    render_leaderboard(entries, console=console)


@app.command()
def modes() -> None:
    """List all registered pluggable game modes."""
    table = Table(title="Available Game Modes", box=box.ROUNDED)
    table.add_column("Mode", style="bold cyan")
    table.add_column("Description")
    table.add_column("Reverse", justify="center")
    for mode in list_modes():
        table.add_row(mode.name, mode.description, "Yes" if mode.is_reverse else "No")
    console.print(table)


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind host address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="TCP port.")] = 8765,
    mode: Annotated[str, typer.Option("--mode", "-m", help="Multiplayer mode.")] = "classic",
    difficulty: Annotated[str, typer.Option("--difficulty", "-d", help="Difficulty preset.")] = "medium",
    low: Annotated[Optional[int], typer.Option("--low", help="Custom range minimum.")] = None,
    high: Annotated[Optional[int], typer.Option("--high", help="Custom range maximum.")] = None,
) -> None:
    """Start the asyncio TCP multiplayer server (MVP 6)."""
    try:
        cfg = _build_config(
            difficulty=difficulty,
            low=low,
            high=high,
            max_tries=None,
            seed=None,
            time_limit=None,
        )
        server = ArenaServer(host=host, port=port, config=cfg, mode_name=mode)
        console.print(
            f"[bold green]Guess Arena multiplayer server[/bold green] listening on "
            f"{host}:{port} (mode={mode}, range={cfg.low}..{cfg.high})"
        )
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
    except ArenaError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def join(
    host: Annotated[str, typer.Option("--host", help="Server host address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Server TCP port.")] = 8765,
    name: Annotated[str, typer.Option("--name", "-n", help="Your multiplayer player name.")] = "Player",
) -> None:
    """Join a live Guess Arena TCP multiplayer server (MVP 6)."""
    try:
        asyncio.run(run_interactive_client(host=host, port=port, name=name, console=console))
    except KeyboardInterrupt:
        console.print("\n[yellow]Left arena.[/yellow]")
    except (ArenaError, OSError) as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def report(
    output: Annotated[Path, typer.Option("--output", "-o", help="Output PNG path.")] = Path("guess_patterns.png"),
    games: Annotated[int, typer.Option("--games", "-n", help="Benchmark simulation count.")] = 2000,
    low: Annotated[int, typer.Option("--low", help="Range minimum.")] = 1,
    high: Annotated[int, typer.Option("--high", help="Range maximum.")] = 100,
    db: Annotated[Optional[Path], typer.Option("--db", help="Path to SQLite stats database.")] = None,
) -> None:
    """Generate a Matplotlib report of human guess patterns and LearningBot benchmarks (MVP 7)."""
    from arena.analytics import generate_analytics_report

    repo = _storage(db)
    summary = generate_analytics_report(
        repo,
        output_path=output,
        low=low,
        high=high,
        simulation_games=games,
    )
    console.print(f"[green]Saved analytics report[/green] to [bold]{summary.output_path}[/bold]")
    top_str = ", ".join(f"{num} ({cnt})" for num, cnt in summary.top_numbers[:5])
    console.print(f"Top human picks ({summary.human_samples} samples): {top_str}")
    render_simulation_comparison(summary.bot_reports, console=console)


@app.command()
def web(
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port.")] = 8000,
    db: Annotated[Optional[Path], typer.Option("--db", help="Path to SQLite stats database.")] = None,
) -> None:
    """Start the FastAPI JSON and WebSocket API server (Stretch goal)."""
    import uvicorn

    from arena.web import create_app

    api_app = create_app(storage=_storage(db))
    console.print(
        f"[bold green]Guess Arena API[/bold green] running at http://{host}:{port} "
        f"(OpenAPI docs: http://{host}:{port}/docs)"
    )
    uvicorn.run(api_app, host=host, port=port)


def main() -> None:
    """CLI script entry point."""
    app()


if __name__ == "__main__":
    main()
