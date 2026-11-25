"""Custom exceptions for Guess Arena."""

from __future__ import annotations


class ArenaError(Exception):
    """Base exception for all Guess Arena errors."""


class GameConfigError(ArenaError):
    """Raised when game configuration values are invalid."""


class GameOverError(ArenaError):
    """Raised when attempting to submit a guess after a game has ended."""


class InvalidGuessError(ArenaError):
    """Raised when a guess is outside the configured range or not a valid integer."""


class ModeNotFoundError(ArenaError):
    """Raised when requesting an unregistered game mode."""


class InconsistentFeedbackError(ArenaError):
    """Raised in Reverse mode when feedback contradicts prior bounds."""


class StorageError(ArenaError):
    """Raised on database persistence errors."""
