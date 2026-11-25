"""Tests for Rich/plotext UI, Typer CLI, and FastAPI REST + WebSocket API."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from rich.console import Console
from typer.testing import CliRunner

from arena.cli import app
from arena.engine import Game, GameConfig
from arena.players import BinarySearchBot, simulate_games
from arena.storage import ArenaStorage
from arena.ui import (
    render_game_status,
    render_leaderboard,
    render_round_summary,
    render_simulation_comparison,
    render_stats,
)
from arena.web import create_app

runner = CliRunner()


def test_rich_and_plotext_ui_rendering(tmp_path: Path) -> None:
    buf = io.StringIO()
    con = Console(file=buf, width=100, force_terminal=False)
    repo = ArenaStorage(tmp_path / "ui.db")

    game = Game(GameConfig(low=1, high=100, secret=60))
    game.submit_guess(50)
    con.print(render_game_status(game.state, player_name="Alice"))
    game.submit_guess(60)
    summary = game.summary()
    repo.record_game("Alice", summary)

    render_round_summary(summary, console=con)
    render_stats(repo.get_player_stats("Alice"), console=con)
    render_leaderboard(repo.get_leaderboard(), console=con)
    sim = simulate_games(BinarySearchBot, num_games=50, config=GameConfig(low=1, high=100, seed=1))
    render_simulation_comparison([sim], console=con)

    output = buf.getvalue()
    assert "Alice" in output
    assert "VICTORY!" in output
    assert "BinarySearchBot" in output


def test_cli_commands_and_interactive_play(tmp_path: Path) -> None:
    db_path = tmp_path / "cli.db"
    png_path = tmp_path / "cli_report.png"

    modes_res = runner.invoke(app, ["modes"])
    assert modes_res.exit_code == 0
    assert "hot_cold" in modes_res.output

    sim_res = runner.invoke(
        app,
        ["simulate", "--bot", "binary", "--games", "100", "--record", "--db", str(db_path)],
    )
    assert sim_res.exit_code == 0, sim_res.output
    assert "BinarySearchBot" in sim_res.output

    play_res = runner.invoke(
        app,
        ["play", "--player", "Alice", "--seed", "7", "--low", "1", "--high", "5", "--db", str(db_path)],
        input="1\n2\n3\n4\n5\n",
    )
    assert play_res.exit_code == 0, play_res.output
    assert "VICTORY!" in play_res.output

    stats_res = runner.invoke(app, ["stats", "Alice", "--db", str(db_path)])
    assert stats_res.exit_code == 0
    assert "Alice" in stats_res.output

    lb_res = runner.invoke(app, ["leaderboard", "--db", str(db_path)])
    assert lb_res.exit_code == 0
    assert "BinarySearchBot" in lb_res.output

    rep_res = runner.invoke(
        app,
        ["report", "--output", str(png_path), "--games", "200", "--db", str(db_path)],
    )
    assert rep_res.exit_code == 0
    assert png_path.exists()


def test_fastapi_rest_and_websocket_api(tmp_path: Path) -> None:
    storage = ArenaStorage(tmp_path / "web.db")
    client = TestClient(create_app(storage))

    assert client.get("/").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}
    modes_resp = client.get("/modes")
    assert modes_resp.status_code == 200
    assert any(m["name"] == "classic" for m in modes_resp.json())

    # Play a full REST game using binary search against the API
    created = client.post(
        "/games",
        json={"player_name": "WebUser", "mode": "classic", "difficulty": "medium", "seed": 12},
    )
    assert created.status_code == 201
    game_id = created.json()["game_id"]
    state = created.json()["state"]

    while not state["game_over"]:
        guess = (state["current_low"] + state["current_high"]) // 2
        step = client.post(f"/games/{game_id}/guess", json={"guess": guess})
        assert step.status_code == 200
        state = step.json()["state"]

    assert state["won"] is True

    stats = client.get("/stats/WebUser")
    assert stats.status_code == 200
    assert stats.json()["wins"] == 1

    board = client.get("/leaderboard")
    assert board.status_code == 200
    assert board.json()[0]["player_name"] == "WebUser"

    sim = client.post("/simulate", json={"bot": "binary", "games": 100, "seed": 3})
    assert sim.status_code == 200
    assert sim.json()["win_rate"] == 1.0

    # WebSocket play endpoint
    with client.websocket_connect("/ws/play") as ws:
        ws.send_json({"action": "start", "player_name": "WsUser", "low": 1, "high": 20, "seed": 5})
        started = ws.receive_json()
        assert started["type"] == "started"
        ws_state = started["state"]
        while not ws_state["game_over"]:
            g = (ws_state["current_low"] + ws_state["current_high"]) // 2
            ws.send_json({"action": "guess", "value": g})
            msg = ws.receive_json()
            assert msg["type"] == "result"
            ws_state = msg["state"]
        assert ws_state["won"] is True
