"""Asyncio TCP multiplayer client for Guess Arena (MVP 6)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from arena.exceptions import ArenaError


@dataclass(slots=True)
class ArenaClient:
    """Async TCP client communicating via newline-delimited JSON with ArenaServer."""

    host: str = "127.0.0.1"
    port: int = 8765
    name: str = "Player"

    _reader: asyncio.StreamReader | None = field(default=None, init=False)
    _writer: asyncio.StreamWriter | None = field(default=None, init=False)

    async def connect(self) -> dict[str, Any]:
        """Connect to the server and complete the join handshake."""
        self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
        await self._send({"type": "join", "name": self.name})
        reply = await self.read_message()
        if reply is None:
            await self.close()
            raise ArenaError("Server closed connection during join handshake.")
        if reply.get("type") == "error":
            await self.close()
            raise ArenaError(str(reply.get("message", "Failed to join server.")))
        return reply

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._writer is None or self._writer.is_closing():
            raise ArenaError("Not connected to server.")
        self._writer.write((json.dumps(payload) + "\n").encode("utf-8"))
        await self._writer.drain()

    async def send_guess(self, value: int) -> None:
        """Submit an integer guess to the server."""
        await self._send({"type": "guess", "value": value})

    async def request_new_round(self) -> None:
        """Ask the server to begin a new round after one finishes."""
        await self._send({"type": "new_round"})

    async def read_message(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Read one JSON message from the server, or None on disconnect."""
        if self._reader is None:
            return None
        try:
            coro = self._reader.readline()
            raw = await (asyncio.wait_for(coro, timeout=timeout) if timeout else coro)
        except (asyncio.TimeoutError, ConnectionResetError, BrokenPipeError, OSError):
            return None
        if not raw:
            return None
        return dict(json.loads(raw.decode("utf-8").strip()))

    async def close(self) -> None:
        """Send quit notice and close the underlying TCP stream."""
        if self._writer is not None and not self._writer.is_closing():
            with contextlib.suppress(Exception):
                await self._send({"type": "quit"})
            self._writer.close()
            with contextlib.suppress(OSError):
                await self._writer.wait_closed()
        self._reader = None
        self._writer = None


def _print_scoreboard(scoreboard: list[dict[str, Any]], console: Console) -> None:
    table = Table(title="Live Scoreboard")
    table.add_column("Player", style="bold")
    table.add_column("Wins", justify="right", style="green")
    table.add_column("Round Tries", justify="right", style="cyan")
    table.add_column("Total Guesses", justify="right", style="dim")
    for row in scoreboard:
        table.add_row(
            str(row["name"]),
            str(row["wins"]),
            str(row["round_attempts"]),
            str(row["total_guesses"]),
        )
    console.print(table)


async def run_interactive_client(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    name: str = "Player",
    console: Console | None = None,
) -> None:
    """Run an interactive multiplayer terminal session against ArenaServer."""
    active_console = console or Console()
    client = ArenaClient(host=host, port=port, name=name)
    welcome = await client.connect()
    active_console.print(
        f"[bold green]Joined Guess Arena[/bold green] at {host}:{port} as [bold]{name}[/bold] "
        f"(Round {welcome['round']}, Range {welcome['low']}..{welcome['high']}, Mode {welcome['mode']})"
    )
    _print_scoreboard(welcome.get("scoreboard", []), active_console)
    active_console.print("[dim]Enter a number to guess, 'new' for next round, or 'quit' to exit.[/dim]")

    stop_event = asyncio.Event()

    async def receiver() -> None:
        while not stop_event.is_set():
            msg = await client.read_message()
            if msg is None:
                active_console.print("[yellow]Disconnected from server.[/yellow]")
                stop_event.set()
                break
            mtype = msg.get("type")
            if mtype == "hint":
                active_console.print(
                    f"[cyan]Attempt #{msg['attempts']}[/cyan] Guess {msg['guess']}: [bold]{msg['hint']}[/bold]"
                )
            elif mtype == "player_joined":
                active_console.print(f"[green]+ {msg['player']} joined the arena.[/green]")
                _print_scoreboard(msg.get("scoreboard", []), active_console)
            elif mtype == "player_left":
                active_console.print(f"[yellow]- {msg['player']} disconnected.[/yellow]")
                _print_scoreboard(msg.get("scoreboard", []), active_console)
            elif mtype == "round_over":
                active_console.print(
                    f"\n[bold green]ROUND {msg['round']} WINNER: {msg['winner']}[/bold green] "
                    f"in {msg['attempts']} tries! Secret was [bold]{msg['secret']}[/bold]."
                )
                _print_scoreboard(msg.get("scoreboard", []), active_console)
                active_console.print("[dim]Type 'new' to start the next round or 'quit' to leave.[/dim]")
            elif mtype == "new_round":
                active_console.print(
                    f"\n[bold magenta]Round {msg['round']} started![/bold magenta] "
                    f"Range: {msg['low']}..{msg['high']}"
                )
            elif mtype == "error":
                active_console.print(f"[bold red]Server:[/bold red] {msg['message']}")

    async def sender() -> None:
        while not stop_event.is_set():
            try:
                raw = await asyncio.to_thread(input, "")
            except EOFError:
                stop_event.set()
                break
            cleaned = raw.strip().lower()
            if not cleaned:
                continue
            if cleaned in {"quit", "exit", "q"}:
                stop_event.set()
                break
            if cleaned == "new":
                await client.request_new_round()
                continue
            try:
                val = int(cleaned)
            except ValueError:
                active_console.print("[red]Enter an integer guess, 'new', or 'quit'.[/red]")
                continue
            await client.send_guess(val)

    recv_task = asyncio.create_task(receiver())
    send_task = asyncio.create_task(sender())
    try:
        done, pending = await asyncio.wait(
            [recv_task, send_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        stop_event.set()
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for task in done:
            with contextlib.suppress(asyncio.CancelledError):
                await task
    finally:
        await client.close()
