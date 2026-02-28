"""
Standardized Error Response Format
====================================
Provides a unified error response structure and custom exception hierarchy
used across all pipeline phases (microservices) to ensure consistent error
reporting, logging, and handling.

Standard error response schema::

    {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Human-readable description",
            "service": "env_validator",
            "details": { ... }
        }
    }
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("devin_triage")

# ---------------------------------------------------------------------------
# Standardized Error Response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorResponse:
    """Immutable, serialisable error envelope used by every pipeline phase.

    Attributes:
        code: Machine-readable error code (e.g. ``VALIDATION_ERROR``).
        message: Human-readable description of what went wrong.
        service: The pipeline phase / microservice that raised the error.
        details: Optional dict with extra context (stack traces, field names, etc.).
    """

    code: str
    message: str
    service: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical ``{"error": { ... }}`` envelope."""
        return {"error": asdict(self)}

    def __str__(self) -> str:
        base = f"[{self.code}] ({self.service}) {self.message}"
        if self.details:
            base += f" | details={self.details}"
        return base


# ---------------------------------------------------------------------------
# Error Codes
# ---------------------------------------------------------------------------

# Environment / configuration
VALIDATION_ERROR = "VALIDATION_ERROR"

# GitHub integration
GITHUB_AUTH_ERROR = "GITHUB_AUTH_ERROR"
GITHUB_REPO_ERROR = "GITHUB_REPO_ERROR"
GITHUB_COMMENT_ERROR = "GITHUB_COMMENT_ERROR"

# Devin API
DEVIN_API_HTTP_ERROR = "DEVIN_API_HTTP_ERROR"
DEVIN_API_CONNECTION_ERROR = "DEVIN_API_CONNECTION_ERROR"

# User input
USER_INPUT_ERROR = "USER_INPUT_ERROR"


# ---------------------------------------------------------------------------
# Custom Exception Hierarchy
# ---------------------------------------------------------------------------


class PipelineError(Exception):
    """Base exception for all pipeline errors.

    Every subclass carries a fully-formed :class:`ErrorResponse` so that
    callers never need to construct one manually.
    """

    def __init__(self, error_response: ErrorResponse) -> None:
        self.error_response = error_response
        super().__init__(str(error_response))


class ConfigurationError(PipelineError):
    """Raised when required environment variables or config values are missing."""


class GitHubServiceError(PipelineError):
    """Raised when a GitHub API call fails."""


class DevinAPIError(PipelineError):
    """Raised when the Devin API returns an error or is unreachable."""


class UserInputError(PipelineError):
    """Raised when user-supplied input is invalid (non-fatal by default)."""


# ---------------------------------------------------------------------------
# Centralised Error Handler
# ---------------------------------------------------------------------------


def handle_error(
    error_response: ErrorResponse,
    *,
    fatal: bool = True,
) -> None:
    """Log a standardised error and optionally terminate the process.

    This is the **single exit-point** for error reporting across all phases,
    guaranteeing a uniform log format and exit behaviour.

    Args:
        error_response: The structured error to report.
        fatal: If ``True`` (default) the process exits with code 1 after
            logging.  Set to ``False`` for non-critical / recoverable errors.
    """
    if fatal:
        logger.error("%s", error_response)
        sys.exit(1)
    else:
        logger.warning("%s", error_response)


def build_error_response(
    code: str,
    message: str,
    service: str,
    details: dict[str, Any] | None = None,
) -> ErrorResponse:
    """Convenience factory for creating an :class:`ErrorResponse`.

    Args:
        code: Machine-readable error code.
        message: Human-readable error message.
        service: Originating service / phase name.
        details: Optional extra context.

    Returns:
        A new :class:`ErrorResponse` instance.
    """
    return ErrorResponse(
        code=code,
        message=message,
        service=service,
        details=details or {},
    )
