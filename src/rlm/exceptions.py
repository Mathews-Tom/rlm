from __future__ import annotations


class RLMError(Exception):
    """Base exception for py-rlm."""


class RecursionDepthError(RLMError):
    """Raised when max_depth exceeded."""


class MaxStepsError(RLMError):
    """Raised when max_steps exceeded."""


class InvalidJSONError(RLMError):
    """Raised when JSON parsing fails."""


class ExecutionError(RLMError):
    """Raised when LLM call fails."""


__all__ = [
    "RLMError",
    "RecursionDepthError",
    "MaxStepsError",
    "InvalidJSONError",
    "ExecutionError",
]
