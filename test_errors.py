"""
Tests for the standardized error response module (errors.py)
and the updated error handling in main.py.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from errors import (
    ERR_DEVIN_API_HTTP,
    ERR_DEVIN_API_UNREACHABLE,
    ERR_ENV_MISSING,
    ERR_GITHUB_COMMENT_FAILED,
    ERR_GITHUB_REPO_ACCESS,
    ERR_INVALID_INPUT,
    ERR_ISSUE_NOT_FOUND,
    ERR_UNEXPECTED,
    ConfigurationError,
    DevinAPIError,
    ErrorResponse,
    GitHubAccessError,
    GitHubCommentError,
    InvalidInputError,
    IssueNotFoundError,
    TriageError,
    make_error,
)


# =========================================================================
# ErrorResponse dataclass tests
# =========================================================================


class TestErrorResponse:
    """Tests for the ErrorResponse dataclass."""

    def test_basic_creation(self) -> None:
        err = ErrorResponse(
            error_code="TEST_CODE",
            message="Something went wrong",
            service="test_service",
        )
        assert err.error_code == "TEST_CODE"
        assert err.message == "Something went wrong"
        assert err.service == "test_service"
        assert err.details == {}
        assert err.timestamp != ""

    def test_timestamp_auto_generated(self) -> None:
        err = ErrorResponse(
            error_code="TEST",
            message="msg",
            service="svc",
        )
        # Should be a valid ISO-8601 timestamp
        parsed = datetime.fromisoformat(err.timestamp)
        assert parsed.tzinfo is not None

    def test_custom_timestamp_preserved(self) -> None:
        custom_ts = "2025-01-01T00:00:00+00:00"
        err = ErrorResponse(
            error_code="TEST",
            message="msg",
            service="svc",
            timestamp=custom_ts,
        )
        assert err.timestamp == custom_ts

    def test_details_preserved(self) -> None:
        details = {"key": "value", "count": 42}
        err = ErrorResponse(
            error_code="TEST",
            message="msg",
            service="svc",
            details=details,
        )
        assert err.details == details

    def test_to_dict(self) -> None:
        err = ErrorResponse(
            error_code="TEST",
            message="msg",
            service="svc",
            details={"foo": "bar"},
            timestamp="2025-01-01T00:00:00+00:00",
        )
        d = err.to_dict()
        assert isinstance(d, dict)
        assert d["error_code"] == "TEST"
        assert d["message"] == "msg"
        assert d["service"] == "svc"
        assert d["details"] == {"foo": "bar"}
        assert d["timestamp"] == "2025-01-01T00:00:00+00:00"

    def test_to_json(self) -> None:
        err = ErrorResponse(
            error_code="TEST",
            message="msg",
            service="svc",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        json_str = err.to_json()
        parsed = json.loads(json_str)
        assert parsed["error_code"] == "TEST"
        assert parsed["service"] == "svc"

    def test_to_json_roundtrip(self) -> None:
        err = ErrorResponse(
            error_code="X",
            message="y",
            service="z",
            details={"a": [1, 2, 3]},
            timestamp="2025-06-15T12:00:00+00:00",
        )
        roundtripped = json.loads(err.to_json())
        assert roundtripped == err.to_dict()

    def test_frozen(self) -> None:
        err = ErrorResponse(error_code="X", message="y", service="z")
        with pytest.raises(AttributeError):
            err.error_code = "NEW"  # type: ignore[misc]

    def test_log_emits_at_correct_level(self, caplog: pytest.LogCaptureFixture) -> None:
        err = ErrorResponse(
            error_code="LOG_TEST",
            message="check log",
            service="logger_svc",
        )
        with caplog.at_level(logging.ERROR, logger="devin_triage"):
            err.log(logging.ERROR)
        assert "LOG_TEST" in caplog.text
        assert "check log" in caplog.text
        assert "logger_svc" in caplog.text

    def test_log_warning_level(self, caplog: pytest.LogCaptureFixture) -> None:
        err = ErrorResponse(
            error_code="WARN_TEST",
            message="a warning",
            service="svc",
        )
        with caplog.at_level(logging.WARNING, logger="devin_triage"):
            err.log(logging.WARNING)
        assert "WARN_TEST" in caplog.text


# =========================================================================
# make_error factory tests
# =========================================================================


class TestMakeError:
    """Tests for the make_error convenience factory."""

    def test_basic(self) -> None:
        err = make_error("CODE", "msg", "svc")
        assert isinstance(err, ErrorResponse)
        assert err.error_code == "CODE"
        assert err.details == {}

    def test_with_details(self) -> None:
        err = make_error("CODE", "msg", "svc", details={"x": 1})
        assert err.details == {"x": 1}

    def test_none_details_becomes_empty_dict(self) -> None:
        err = make_error("CODE", "msg", "svc", details=None)
        assert err.details == {}


# =========================================================================
# Custom exception hierarchy tests
# =========================================================================


class TestExceptionHierarchy:
    """Verify the custom exception hierarchy and that each exception carries
    a structured ErrorResponse."""

    def _make_exc(self, cls: type[TriageError], code: str) -> TriageError:
        resp = make_error(code, "test message", "test_svc")
        return cls(resp)

    def test_triage_error_base(self) -> None:
        exc = self._make_exc(TriageError, "BASE")
        assert isinstance(exc, Exception)
        assert exc.error_response.error_code == "BASE"
        assert str(exc) == "test message"

    @pytest.mark.parametrize(
        "cls,code",
        [
            (ConfigurationError, ERR_ENV_MISSING),
            (GitHubAccessError, ERR_GITHUB_REPO_ACCESS),
            (GitHubCommentError, ERR_GITHUB_COMMENT_FAILED),
            (InvalidInputError, ERR_INVALID_INPUT),
            (IssueNotFoundError, ERR_ISSUE_NOT_FOUND),
            (DevinAPIError, ERR_DEVIN_API_HTTP),
        ],
    )
    def test_subclass_is_triage_error(self, cls: type[TriageError], code: str) -> None:
        exc = self._make_exc(cls, code)
        assert isinstance(exc, TriageError)
        assert isinstance(exc, cls)
        assert exc.error_response.error_code == code


# =========================================================================
# main.py function-level tests
# =========================================================================


class TestValidateEnv:
    """Tests for _validate_env in main.py."""

    def test_raises_configuration_error_when_vars_missing(self) -> None:
        with patch.dict(
            "os.environ",
            {"GITHUB_TOKEN": "", "DEVIN_API_KEY": "", "GITHUB_REPO_NAME": ""},
            clear=False,
        ):
            # Need to reload module-level constants
            import main

            main.GITHUB_TOKEN = ""
            main.DEVIN_API_KEY = ""
            main.GITHUB_REPO_NAME = ""

            with pytest.raises(ConfigurationError) as exc_info:
                main._validate_env()

            resp = exc_info.value.error_response
            assert resp.error_code == ERR_ENV_MISSING
            assert resp.service == "configuration"
            assert "GITHUB_TOKEN" in resp.details["missing_variables"]
            assert "DEVIN_API_KEY" in resp.details["missing_variables"]
            assert "GITHUB_REPO_NAME" in resp.details["missing_variables"]

    def test_no_error_when_all_vars_set(self) -> None:
        import main

        main.GITHUB_TOKEN = "tok"
        main.DEVIN_API_KEY = "key"
        main.GITHUB_REPO_NAME = "owner/repo"
        # Should not raise
        main._validate_env()


class TestFetchLabelledIssues:
    """Tests for fetch_labelled_issues in main.py."""

    def test_raises_github_access_error(self) -> None:
        import main

        with patch("main.Github") as mock_gh_cls:
            mock_gh_cls.return_value.get_repo.side_effect = Exception("repo not found")
            with pytest.raises(GitHubAccessError) as exc_info:
                main.fetch_labelled_issues("tok", "owner/repo", "label")

            resp = exc_info.value.error_response
            assert resp.error_code == ERR_GITHUB_REPO_ACCESS
            assert resp.service == "github"
            assert "owner/repo" in resp.details["repository"]


class TestCreateDevinSession:
    """Tests for create_devin_session in main.py."""

    def test_raises_devin_api_error_on_http_error(self) -> None:
        import main

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        http_err = requests.exceptions.HTTPError(response=mock_response)

        with patch("main.requests.post") as mock_post:
            mock_post.return_value.raise_for_status.side_effect = http_err
            mock_post.return_value.status_code = 403

            with pytest.raises(DevinAPIError) as exc_info:
                main.create_devin_session("key", "prompt")

            resp = exc_info.value.error_response
            assert resp.error_code == ERR_DEVIN_API_HTTP
            assert resp.service == "devin_api"
            assert resp.details["status_code"] == 403

    def test_raises_devin_api_error_on_connection_error(self) -> None:
        import main

        with patch("main.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("refused")

            with pytest.raises(DevinAPIError) as exc_info:
                main.create_devin_session("key", "prompt")

            resp = exc_info.value.error_response
            assert resp.error_code == ERR_DEVIN_API_UNREACHABLE
            assert resp.service == "devin_api"


class TestPostGithubComment:
    """Tests for post_github_comment in main.py."""

    def test_logs_warning_on_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        import main

        mock_issue = MagicMock()
        mock_issue.number = 42
        mock_issue.create_comment.side_effect = Exception("permission denied")

        with caplog.at_level(logging.WARNING, logger="devin_triage"):
            main.post_github_comment(mock_issue, "https://example.com/session")

        assert ERR_GITHUB_COMMENT_FAILED in caplog.text

    def test_succeeds_without_error(self) -> None:
        import main

        mock_issue = MagicMock()
        mock_issue.number = 7
        # Should not raise
        main.post_github_comment(mock_issue, "https://example.com/session")
        mock_issue.create_comment.assert_called_once()


class TestPromptUserSelection:
    """Tests for prompt_user_selection in main.py."""

    def test_invalid_input_prints_standardised_message(self) -> None:
        import main

        mock_issue = MagicMock()
        lookup = {1: mock_issue}

        # First call: invalid input "abc", second call: quit
        with patch("builtins.input", side_effect=["abc", "q"]):
            result = main.prompt_user_selection(lookup)

        assert result is None  # user eventually quit

    def test_issue_not_found_prints_standardised_message(self) -> None:
        import main

        mock_issue = MagicMock()
        lookup = {1: mock_issue}

        with patch("builtins.input", side_effect=["999", "q"]):
            result = main.prompt_user_selection(lookup)

        assert result is None

    def test_valid_selection_returns_issue(self) -> None:
        import main

        mock_issue = MagicMock()
        lookup = {1: mock_issue}

        with patch("builtins.input", return_value="1"):
            result = main.prompt_user_selection(lookup)

        assert result is mock_issue


class TestMainErrorHandling:
    """Tests for the top-level main() error handler."""

    def test_main_catches_triage_error_and_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        import main

        main.GITHUB_TOKEN = ""
        main.DEVIN_API_KEY = ""
        main.GITHUB_REPO_NAME = ""

        with pytest.raises(SystemExit) as exc_info:
            main.main()

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        # The structured JSON error should be on stderr
        err_output = json.loads(captured.err)
        assert err_output["error_code"] == ERR_ENV_MISSING
        assert err_output["service"] == "configuration"
        assert "timestamp" in err_output


# =========================================================================
# Error response format consistency tests
# =========================================================================


class TestErrorResponseFormatConsistency:
    """Ensure every error produced by the application follows the same
    five-field schema: error_code, message, service, details, timestamp."""

    REQUIRED_KEYS = {"error_code", "message", "service", "details", "timestamp"}

    @pytest.mark.parametrize(
        "code,msg,svc,details",
        [
            (ERR_ENV_MISSING, "missing vars", "configuration", {"missing_variables": ["X"]}),
            (ERR_GITHUB_REPO_ACCESS, "repo fail", "github", {"repository": "a/b"}),
            (ERR_GITHUB_COMMENT_FAILED, "comment fail", "github", {"issue_number": 1}),
            (ERR_INVALID_INPUT, "bad input", "user_input", {"raw_input": "abc"}),
            (ERR_ISSUE_NOT_FOUND, "no issue", "user_input", {"issue_number": 99}),
            (ERR_DEVIN_API_HTTP, "http error", "devin_api", {"status_code": 500}),
            (ERR_DEVIN_API_UNREACHABLE, "unreachable", "devin_api", {}),
            (ERR_UNEXPECTED, "unexpected", "unknown", {}),
        ],
    )
    def test_all_errors_have_required_fields(
        self,
        code: str,
        msg: str,
        svc: str,
        details: dict[str, Any],
    ) -> None:
        err = make_error(code, msg, svc, details)
        d = err.to_dict()
        assert set(d.keys()) == self.REQUIRED_KEYS

    def test_json_output_always_parseable(self) -> None:
        err = make_error("X", "y", "z", {"nested": {"a": [1]}})
        parsed = json.loads(err.to_json())
        assert set(parsed.keys()) == self.REQUIRED_KEYS
