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


class BudgetExceeded(RLMError):
    """Base class for budget cap violations."""


class TokenLimitExceeded(BudgetExceeded):
    """Raised when a token limit (prompt, completion, or total) is exceeded."""


class CostLimitExceeded(BudgetExceeded):
    """Raised when the cost ceiling (max_cost USD) is exceeded."""


class CodeExecutionError(RLMError):
    """Raised when REPL code execution fails or times out."""


class MaxREPLIterationsError(RLMError):
    """Raised when REPL loop exceeds max_repl_iterations."""


__all__ = [
    "RLMError",
    "RecursionDepthError",
    "MaxStepsError",
    "InvalidJSONError",
    "ExecutionError",
    "BudgetExceeded",
    "TokenLimitExceeded",
    "CostLimitExceeded",
    "CodeExecutionError",
    "MaxREPLIterationsError",
]
