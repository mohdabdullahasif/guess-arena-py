# Guess Arena

A multi-mode Python guessing game built with clean layered architecture:

- **Headless Engine** (`src/arena/engine.py`) & **Pluggable Modes** (`src/arena/modes.py`): Zero `input()`/`print()` inside the engine, supporting `classic`, `timed`, `hot_cold`, `daily`, `hardcore`, and `reverse` modes.
- **AI Bots & Simulation** (`src/arena/players.py`): Abstract `Player` interface with `HumanPlayer`, `RandomBot`, `BinarySearchBot` (proving `<= ceil(log2(N))` worst-case tries across 10,000 games), and a Bayesian/DP `LearningBot` that exploits human number biases.
- **Persistent SQLite Stats & Leaderboards** (`src/arena/storage.py`): Tracks `players`, `games`, and `guesses`, computing win rates, average tries, and per-difficulty win streaks.
- **Rich + Plotext Terminal UI** (`src/arena/ui.py`): Live shrinking range bar, color-coded guess history tables, and terminal charts.
- **Matplotlib Analytics** (`src/arena/analytics.py`): Exports visual charts comparing human guess patterns against uniform distribution and bot performance.
- **Asyncio Multiplayer** (`src/arena/server.py`, `src/arena/client.py`): TCP newline-delimited JSON server supporting live multi-client rounds, scoreboards, and clean disconnect handling.
- **FastAPI JSON & WebSocket API** (`src/arena/web.py`) + **GitHub Actions CI** (`.github/workflows/ci.yml`).

## Installation

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

## CLI Commands

```bash
# Play any mode
arena play --player Alice --mode classic --difficulty medium
arena play --player Alice --mode timed --time-limit 30
arena play --player Alice --mode hot_cold
arena play --player Alice --mode daily
arena play --player Alice --mode hardcore
arena play --mode reverse --bot learning

# List registered pluggable modes
arena modes

# Run 10,000-game simulations (uniform vs human-biased)
arena simulate --bot all --games 10000
arena simulate --bot all --games 10000 --biased --record

# Persistent stats and leaderboard
arena stats Alice
arena leaderboard --limit 10

# Multiplayer TCP race (run server in terminal 1, clients in terminals 2 & 3)
arena serve --host 127.0.0.1 --port 8765
arena join --host 127.0.0.1 --port 8765 --name Alice
arena join --host 127.0.0.1 --port 8765 --name Bob

# Matplotlib guess pattern & bot comparison PNG report
arena report --output guess_patterns.png

# FastAPI JSON + WebSocket API (Swagger docs at /docs)
arena web --host 127.0.0.1 --port 8000
```

## Running Tests

```bash
pytest -v
```
