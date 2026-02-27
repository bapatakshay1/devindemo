"""
Standardized Error Response Module
====================================
Provides a uniform error response format and custom exception hierarchy
for all phases (microservices) of the Devin Issue Triage & Resolution CLI.

Every error raised by the application is represented as a structured
:class:`ErrorResponse` containing:

- ``error_code``  -- a machine-readable identifier (e.g. ``ENV_MISSING``)
- ``message``     -- a short human-readable summary
- ``service``     -- the phase / microservice that raised the error
- ``details``     -- optional extra context (free-form dict)
- ``timestamp``   -- ISO-8601 UTC timestamp of when the error occurred
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("devin_triage")

# ---------------------------------------------------------------------------
# Standardised error codes
# ---------------------------------------------------------------------------

# Environment / configuration
ERR_ENV_MISSING = "ENV_MISSING"

# GitHub phase
ERR_GITHUB_REPO_ACCESS = "GITHUB_REPO_ACCESS"
ERR_GITHUB_COMMENT_FAILED = "GITHUB_COMMENT_FAILED"

# User input phase
ERR_INVALID_INPUT = "INVALID_INPUT"
ERR_ISSUE_NOT_FOUND = "ISSUE_NOT_FOUND"

# Devin API phase
ERR_DEVIN_API_HTTP = "DEVIN_API_HTTP"
ERR_DEVIN_API_UNREACHABLE = "DEVIN_API_UNREACHABLE"

# Generic
ERR_UNEXPECTED = "UNEXPECTED"


# ---------------------------------------------------------------------------
# Standardised error response dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ErrorResponse:
    """Immutable, structured error response used across all services.

    Attributes:
        error_code: Machine-readable error identifier.
        message: Human-readable error summary.
        service: The service / pipeline phase that generated the error.
        details: Optional mapping of additional contextual information.
        timestamp: ISO-8601 UTC timestamp string.
    """

    error_code: str
    message: str
    service: str
    details: dict[str, Any] = dataclasses.field(default_factory=dict)
    timestamp: str = dataclasses.field(default="")

    def __post_init__(self) -> None:
        if not self.timestamp:
            # frozen=True requires object.__setattr__ for late init
            object.__setattr__(
                self,
                "timestamp",
                datetime.now(timezone.utc).isoformat(),
            )

    # Serialisation helpers ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the error response as a plain dictionary."""
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return the error response as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def log(self, level: int = logging.ERROR) -> None:
        """Emit the error through the standard logger."""
        logger.log(
            level,
            "[%s] %s | service=%s | details=%s",
            self.error_code,
            self.message,
            self.service,
            self.details or "{}",
        )


# ---------------------------------------------------------------------------
# Custom exception hierarchy
# ---------------------------------------------------------------------------


class TriageError(Exception):
    """Base exception for all triage pipeline errors.

    Every custom exception carries a fully-formed :class:`ErrorResponse` so
    callers can inspect or serialise the structured error at any point.
    """

    def __init__(self, error_response: ErrorResponse) -> None:
        self.error_response = error_response
        super().__init__(error_response.message)


class ConfigurationError(TriageError):
    """Raised when required environment variables are missing."""


class GitHubAccessError(TriageError):
    """Raised when the GitHub API cannot be reached or returns an error."""


class GitHubCommentError(TriageError):
    """Raised when posting a comment to a GitHub issue fails."""


class InvalidInputError(TriageError):
    """Raised when user-supplied input is invalid."""


class IssueNotFoundError(TriageError):
    """Raised when a selected issue number does not exist in the triage report."""


class DevinAPIError(TriageError):
    """Raised when the Devin API returns an HTTP error or is unreachable."""


# ---------------------------------------------------------------------------
# Convenience factory helpers
# ---------------------------------------------------------------------------


def make_error(
    error_code: str,
    message: str,
    service: str,
    details: dict[str, Any] | None = None,
) -> ErrorResponse:
    """Create an :class:`ErrorResponse` with sensible defaults.

    Args:
        error_code: Machine-readable error identifier.
        message: Human-readable error summary.
        service: The originating service / pipeline phase.
        details: Optional additional context.

    Returns:
        A new :class:`ErrorResponse` instance.
    """
    return ErrorResponse(
        error_code=error_code,
        message=message,
        service=service,
        details=details or {},
    )
