from __future__ import annotations


class RLMError(Exception):
    """Base exception for py-rlm."""

    pass


class RecursionDepthError(RLMError):
    """Raised when max_depth exceeded."""

    pass


class MaxStepsError(RLMError):
    """Raised when max_steps exceeded."""

    pass


class InvalidJSONError(RLMError):
    """Raised when JSON parsing fails."""

    pass


class ExecutionError(RLMError):
    """Raised when LLM call fails."""

    pass


__all__ = [
    "RLMError",
    "RecursionDepthError",
    "MaxStepsError",
    "InvalidJSONError",
    "ExecutionError",
]
