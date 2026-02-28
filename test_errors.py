"""
Tests for the standardised error response module (errors.py).
"""

from __future__ import annotations

import pytest

from errors import (
    DEVIN_API_CONNECTION_ERROR,
    DEVIN_API_HTTP_ERROR,
    GITHUB_AUTH_ERROR,
    GITHUB_COMMENT_ERROR,
    GITHUB_REPO_ERROR,
    USER_INPUT_ERROR,
    VALIDATION_ERROR,
    ConfigurationError,
    DevinAPIError,
    ErrorResponse,
    GitHubServiceError,
    PipelineError,
    UserInputError,
    build_error_response,
    handle_error,
)


# ---------------------------------------------------------------------------
# ErrorResponse dataclass
# ---------------------------------------------------------------------------


class TestErrorResponse:
    """Tests for the ErrorResponse dataclass."""

    def test_basic_creation(self) -> None:
        err = ErrorResponse(
            code="TEST_CODE",
            message="Something broke",
            service="test_service",
        )
        assert err.code == "TEST_CODE"
        assert err.message == "Something broke"
        assert err.service == "test_service"
        assert err.details == {}

    def test_creation_with_details(self) -> None:
        details = {"key": "value", "count": 42}
        err = ErrorResponse(
            code="TEST_CODE",
            message="msg",
            service="svc",
            details=details,
        )
        assert err.details == details

    def test_to_dict_envelope(self) -> None:
        err = ErrorResponse(
            code="ERR",
            message="bad",
            service="svc",
            details={"x": 1},
        )
        d = err.to_dict()
        assert "error" in d
        inner = d["error"]
        assert inner["code"] == "ERR"
        assert inner["message"] == "bad"
        assert inner["service"] == "svc"
        assert inner["details"] == {"x": 1}

    def test_to_dict_empty_details(self) -> None:
        err = ErrorResponse(code="C", message="m", service="s")
        assert err.to_dict()["error"]["details"] == {}

    def test_str_without_details(self) -> None:
        err = ErrorResponse(code="C", message="m", service="s")
        s = str(err)
        assert "[C]" in s
        assert "(s)" in s
        assert "m" in s
        assert "details=" not in s

    def test_str_with_details(self) -> None:
        err = ErrorResponse(code="C", message="m", service="s", details={"k": "v"})
        s = str(err)
        assert "details=" in s

    def test_frozen(self) -> None:
        err = ErrorResponse(code="C", message="m", service="s")
        with pytest.raises(AttributeError):
            err.code = "NEW"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Error codes are defined as expected
# ---------------------------------------------------------------------------


class TestErrorCodes:
    """Ensure all expected error codes are present."""

    def test_validation_error(self) -> None:
        assert VALIDATION_ERROR == "VALIDATION_ERROR"

    def test_github_auth_error(self) -> None:
        assert GITHUB_AUTH_ERROR == "GITHUB_AUTH_ERROR"

    def test_github_repo_error(self) -> None:
        assert GITHUB_REPO_ERROR == "GITHUB_REPO_ERROR"

    def test_github_comment_error(self) -> None:
        assert GITHUB_COMMENT_ERROR == "GITHUB_COMMENT_ERROR"

    def test_devin_api_http_error(self) -> None:
        assert DEVIN_API_HTTP_ERROR == "DEVIN_API_HTTP_ERROR"

    def test_devin_api_connection_error(self) -> None:
        assert DEVIN_API_CONNECTION_ERROR == "DEVIN_API_CONNECTION_ERROR"

    def test_user_input_error(self) -> None:
        assert USER_INPUT_ERROR == "USER_INPUT_ERROR"


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Tests for the custom exception classes."""

    def _make_error_response(self) -> ErrorResponse:
        return ErrorResponse(code="T", message="test", service="test_svc")

    def test_pipeline_error_is_exception(self) -> None:
        err_resp = self._make_error_response()
        exc = PipelineError(err_resp)
        assert isinstance(exc, Exception)
        assert exc.error_response is err_resp

    def test_configuration_error_inherits_pipeline(self) -> None:
        exc = ConfigurationError(self._make_error_response())
        assert isinstance(exc, PipelineError)

    def test_github_service_error_inherits_pipeline(self) -> None:
        exc = GitHubServiceError(self._make_error_response())
        assert isinstance(exc, PipelineError)

    def test_devin_api_error_inherits_pipeline(self) -> None:
        exc = DevinAPIError(self._make_error_response())
        assert isinstance(exc, PipelineError)

    def test_user_input_error_inherits_pipeline(self) -> None:
        exc = UserInputError(self._make_error_response())
        assert isinstance(exc, PipelineError)

    def test_str_representation(self) -> None:
        err_resp = self._make_error_response()
        exc = PipelineError(err_resp)
        assert str(exc) == str(err_resp)


# ---------------------------------------------------------------------------
# build_error_response helper
# ---------------------------------------------------------------------------


class TestBuildErrorResponse:
    """Tests for the build_error_response factory function."""

    def test_without_details(self) -> None:
        err = build_error_response(
            code="C",
            message="msg",
            service="svc",
        )
        assert isinstance(err, ErrorResponse)
        assert err.code == "C"
        assert err.message == "msg"
        assert err.service == "svc"
        assert err.details == {}

    def test_with_details(self) -> None:
        err = build_error_response(
            code="C",
            message="msg",
            service="svc",
            details={"foo": "bar"},
        )
        assert err.details == {"foo": "bar"}

    def test_none_details_normalised(self) -> None:
        err = build_error_response(code="C", message="m", service="s", details=None)
        assert err.details == {}


# ---------------------------------------------------------------------------
# handle_error
# ---------------------------------------------------------------------------


class TestHandleError:
    """Tests for the centralised handle_error function."""

    def test_fatal_exits(self) -> None:
        err = ErrorResponse(code="FATAL", message="boom", service="svc")
        with pytest.raises(SystemExit) as exc_info:
            handle_error(err, fatal=True)
        assert exc_info.value.code == 1

    def test_non_fatal_does_not_exit(self) -> None:
        err = ErrorResponse(code="WARN", message="meh", service="svc")
        # Should NOT raise SystemExit
        handle_error(err, fatal=False)

    def test_fatal_is_default(self) -> None:
        err = ErrorResponse(code="FATAL", message="boom", service="svc")
        with pytest.raises(SystemExit):
            handle_error(err)
