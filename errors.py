"""
Standardized error response format for the Devin Issue Triage CLI.
==================================================================

Provides a unified :class:`ErrorResponse` dataclass and a hierarchy of custom
exceptions so that every phase of the pipeline reports errors in a consistent,
machine-readable structure.

Standard error response schema::

    {
        "error": true,
        "error_code": "GITHUB_REPO_ACCESS_FAILED",
        "message": "Failed to access repository 'owner/repo'",
        "service": "github",
        "timestamp": "2025-01-15T12:34:56.789012+00:00",
        "details": { ... }          # optional extra context
    }
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("devin_triage")

# ---------------------------------------------------------------------------
# Error codes -- one constant per distinct failure scenario
# ---------------------------------------------------------------------------

# Configuration / environment
ERR_MISSING_ENV_VARS = "MISSING_ENV_VARS"

# GitHub service
ERR_GITHUB_REPO_ACCESS = "GITHUB_REPO_ACCESS_FAILED"
ERR_GITHUB_COMMENT_FAILED = "GITHUB_COMMENT_FAILED"

# Devin API service
ERR_DEVIN_API_HTTP = "DEVIN_API_HTTP_ERROR"
ERR_DEVIN_API_UNREACHABLE = "DEVIN_API_UNREACHABLE"
ERR_DEVIN_POLL_FAILED = "DEVIN_POLL_REQUEST_FAILED"
ERR_DEVIN_POLL_TIMEOUT = "DEVIN_POLL_TIMEOUT"

# Dispatch
ERR_DISPATCH_FAILED = "DISPATCH_FAILED"

# Service identifiers
SERVICE_CONFIG = "config"
SERVICE_GITHUB = "github"
SERVICE_DEVIN_API = "devin_api"
SERVICE_DISPATCH = "dispatch"


# ---------------------------------------------------------------------------
# Standardised error response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorResponse:
    """Immutable, JSON-serialisable error envelope.

    Every error produced anywhere in the pipeline MUST be wrapped in this
    structure so that callers can rely on a predictable schema.

    Attributes:
        error: Always ``True`` -- makes it trivial to detect error dicts.
        error_code: A short, uppercase identifier (e.g. ``MISSING_ENV_VARS``).
        message: A human-readable description of what went wrong.
        service: Which logical service produced the error (``config``,
            ``github``, ``devin_api``, ``dispatch``).
        timestamp: ISO-8601 UTC timestamp of when the error was created.
        details: Optional mapping with extra contextual data (HTTP status
            codes, variable names, session IDs, etc.).
    """

    error: bool = field(default=True, init=False)
    error_code: str = ""
    message: str = ""
    service: str = ""
    timestamp: str = field(default="", init=False)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # frozen=True requires object.__setattr__ for post-init mutations
        object.__setattr__(
            self,
            "timestamp",
            datetime.now(tz=timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation."""
        return asdict(self)

    def log(self, level: int = logging.ERROR) -> None:
        """Emit a structured log line at the requested level."""
        logger.log(
            level,
            "[%s] %s | service=%s | details=%s",
            self.error_code,
            self.message,
            self.service,
            self.details,
        )


# ---------------------------------------------------------------------------
# Custom exception hierarchy
# ---------------------------------------------------------------------------


class PipelineError(Exception):
    """Base exception for all pipeline errors.

    Every subclass carries an :class:`ErrorResponse` so that ``except``
    blocks can access the standardised payload without re-constructing it.
    """

    def __init__(self, response: ErrorResponse) -> None:
        self.response = response
        super().__init__(response.message)


class ConfigurationError(PipelineError):
    """Raised when required configuration / environment variables are missing."""


class GitHubServiceError(PipelineError):
    """Raised when a GitHub API call fails."""


class DevinAPIError(PipelineError):
    """Raised when the Devin session API returns an error or is unreachable."""


class DispatchError(PipelineError):
    """Raised when dispatching a Devin session for an issue fails."""


# ---------------------------------------------------------------------------
# Convenience factory helpers
# ---------------------------------------------------------------------------


def make_error(
    error_code: str,
    message: str,
    service: str,
    details: dict[str, Any] | None = None,
) -> ErrorResponse:
    """Create an :class:`ErrorResponse` and log it at ERROR level."""
    resp = ErrorResponse(
        error_code=error_code,
        message=message,
        service=service,
        details=details or {},
    )
    resp.log()
    return resp


def make_warning(
    error_code: str,
    message: str,
    service: str,
    details: dict[str, Any] | None = None,
) -> ErrorResponse:
    """Create an :class:`ErrorResponse` and log it at WARNING level."""
    resp = ErrorResponse(
        error_code=error_code,
        message=message,
        service=service,
        details=details or {},
    )
    resp.log(level=logging.WARNING)
    return resp
