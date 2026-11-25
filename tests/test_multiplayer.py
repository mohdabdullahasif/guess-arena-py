"""Tests for asyncio TCP multiplayer server and client (MVP 6)."""

from __future__ import annotations

import asyncio

from arena.client import ArenaClient
from arena.engine import GameConfig
from arena.exceptions import ArenaError
from arena.server import ArenaServer


async def _wait_for_type(client: ArenaClient, target_type: str, timeout: float = 2.0) -> dict:
    for _ in range(15):
        msg = await client.read_message(timeout=timeout)
        assert msg is not None
        if msg.get("type") == target_type:
            return msg
    raise AssertionError(f"Did not receive message of type {target_type!r}")


def test_multiplayer_race_new_round_and_disconnect() -> None:
    async def scenario() -> None:
        server = ArenaServer(
            host="127.0.0.1",
            port=0,
            config=GameConfig(low=1, high=100),
            preset_secrets=[42, 77],
        )
        await server.start()
        port = server.bound_port

        alice = ArenaClient("127.0.0.1", port, "Alice")
        bob = ArenaClient("127.0.0.1", port, "Bob")
        dup = ArenaClient("127.0.0.1", port, "Alice")

        try:
            welcome_a = await alice.connect()
            assert welcome_a["player"] == "Alice"
            assert welcome_a["round"] == 1

            welcome_b = await bob.connect()
            assert welcome_b["player"] == "Bob"
            joined_msg = await _wait_for_type(alice, "player_joined")
            assert joined_msg["player"] == "Bob"

            # Duplicate name rejected cleanly
            try:
                await dup.connect()
            except ArenaError as exc:
                assert "already taken" in str(exc)
            else:
                raise AssertionError("Expected duplicate player name error")

            # Bob guesses wrong first
            await bob.send_guess(10)
            bob_hint = await _wait_for_type(bob, "hint")
            assert bob_hint["outcome"] == "HIGHER"
            await _wait_for_type(alice, "progress")
            await _wait_for_type(bob, "progress")

            # Alice guesses 42 and wins round 1
            await alice.send_guess(42)
            alice_hint = await _wait_for_type(alice, "hint")
            assert alice_hint["outcome"] == "CORRECT"

            over_a = await _wait_for_type(alice, "round_over")
            over_b = await _wait_for_type(bob, "round_over")
            assert over_a["winner"] == "Alice"
            assert over_b["winner"] == "Alice"
            assert over_a["secret"] == 42

            # Start round 2
            await bob.request_new_round()
            nr_a = await _wait_for_type(alice, "new_round")
            nr_b = await _wait_for_type(bob, "new_round")
            assert nr_a["round"] == 2
            assert nr_b["round"] == 2

            # Bob disconnects; Alice gets player_left and can still win round 2
            await bob.close()
            left_msg = await _wait_for_type(alice, "player_left")
            assert left_msg["player"] == "Bob"

            await alice.send_guess(77)
            round2_over = await _wait_for_type(alice, "round_over")
            assert round2_over["winner"] == "Alice"
            assert round2_over["secret"] == 77
        finally:
            await alice.close()
            await bob.close()
            await server.stop()

    asyncio.run(scenario())
