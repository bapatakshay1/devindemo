"""
Tests for the standardised error response format (``errors.py``) and
its integration into the pipeline functions in ``main.py``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from errors import (
    ERR_DEVIN_API_HTTP,
    ERR_DEVIN_API_UNREACHABLE,
    ERR_DEVIN_POLL_FAILED,
    ERR_DEVIN_POLL_TIMEOUT,
    ERR_DISPATCH_FAILED,
    ERR_GITHUB_COMMENT_FAILED,
    ERR_GITHUB_REPO_ACCESS,
    ERR_MISSING_ENV_VARS,
    SERVICE_CONFIG,
    SERVICE_DEVIN_API,
    SERVICE_DISPATCH,
    SERVICE_GITHUB,
    ConfigurationError,
    DevinAPIError,
    DispatchError,
    ErrorResponse,
    GitHubServiceError,
    PipelineError,
    make_error,
    make_warning,
)


# =========================================================================
# ErrorResponse dataclass
# =========================================================================


class TestErrorResponse:
    """Verify the immutable error envelope."""

    def test_defaults(self) -> None:
        resp = ErrorResponse(
            error_code="TEST_CODE",
            message="something broke",
            service="test_svc",
        )
        assert resp.error is True
        assert resp.error_code == "TEST_CODE"
        assert resp.message == "something broke"
        assert resp.service == "test_svc"
        assert resp.details == {}
        # timestamp should be a valid ISO-8601 string
        dt = datetime.fromisoformat(resp.timestamp)
        assert dt.tzinfo is not None  # timezone-aware

    def test_with_details(self) -> None:
        resp = ErrorResponse(
            error_code="X",
            message="y",
            service="z",
            details={"key": "value", "num": 42},
        )
        assert resp.details == {"key": "value", "num": 42}

    def test_to_dict(self) -> None:
        resp = ErrorResponse(
            error_code="A",
            message="b",
            service="c",
            details={"d": 1},
        )
        d = resp.to_dict()
        assert isinstance(d, dict)
        assert d["error"] is True
        assert d["error_code"] == "A"
        assert d["message"] == "b"
        assert d["service"] == "c"
        assert d["details"] == {"d": 1}
        assert "timestamp" in d

    def test_to_dict_is_json_serialisable(self) -> None:
        resp = ErrorResponse(
            error_code="JSON_TEST",
            message="check json",
            service="test",
        )
        # Should not raise
        serialised = json.dumps(resp.to_dict())
        assert '"error": true' in serialised

    def test_frozen(self) -> None:
        resp = ErrorResponse(error_code="F", message="frozen", service="s")
        with pytest.raises(AttributeError):
            resp.error_code = "MUTATED"  # type: ignore[misc]

    def test_log_error(self, caplog: pytest.LogCaptureFixture) -> None:
        resp = ErrorResponse(
            error_code="LOG_E",
            message="error msg",
            service="svc",
            details={"x": 1},
        )
        with caplog.at_level(logging.ERROR, logger="devin_triage"):
            resp.log()
        assert "LOG_E" in caplog.text
        assert "error msg" in caplog.text

    def test_log_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        resp = ErrorResponse(
            error_code="LOG_W",
            message="warn msg",
            service="svc",
        )
        with caplog.at_level(logging.WARNING, logger="devin_triage"):
            resp.log(level=logging.WARNING)
        assert "LOG_W" in caplog.text


# =========================================================================
# Exception hierarchy
# =========================================================================


class TestExceptionHierarchy:
    """All custom exceptions carry an ErrorResponse and descend from PipelineError."""

    def _make_resp(self, code: str = "EX") -> ErrorResponse:
        return ErrorResponse(error_code=code, message="test", service="test")

    def test_pipeline_error(self) -> None:
        r = self._make_resp("P")
        exc = PipelineError(r)
        assert exc.response is r
        assert str(exc) == "test"

    def test_configuration_error_is_pipeline_error(self) -> None:
        assert issubclass(ConfigurationError, PipelineError)
        exc = ConfigurationError(self._make_resp())
        assert exc.response.error_code == "EX"

    def test_github_service_error(self) -> None:
        assert issubclass(GitHubServiceError, PipelineError)

    def test_devin_api_error(self) -> None:
        assert issubclass(DevinAPIError, PipelineError)

    def test_dispatch_error(self) -> None:
        assert issubclass(DispatchError, PipelineError)


# =========================================================================
# Factory helpers
# =========================================================================


class TestFactoryHelpers:
    def test_make_error(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR, logger="devin_triage"):
            resp = make_error("E1", "msg1", "svc1", details={"a": 1})
        assert resp.error_code == "E1"
        assert resp.message == "msg1"
        assert resp.service == "svc1"
        assert resp.details == {"a": 1}
        assert "E1" in caplog.text

    def test_make_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="devin_triage"):
            resp = make_warning("W1", "msg2", "svc2")
        assert resp.error_code == "W1"
        assert "W1" in caplog.text


# =========================================================================
# Integration: _validate_env
# =========================================================================


class TestValidateEnv:
    """_validate_env should raise ConfigurationError with standard format."""

    @patch("main.GITHUB_TOKEN", "")
    @patch("main.DEVIN_API_KEY", "")
    @patch("main.GITHUB_REPO_NAME", "")
    def test_all_missing(self) -> None:
        from main import _validate_env

        with pytest.raises(ConfigurationError) as exc_info:
            _validate_env()
        resp = exc_info.value.response
        assert resp.error is True
        assert resp.error_code == ERR_MISSING_ENV_VARS
        assert resp.service == SERVICE_CONFIG
        assert set(resp.details["missing_variables"]) == {
            "GITHUB_TOKEN",
            "DEVIN_API_KEY",
            "GITHUB_REPO_NAME",
        }

    @patch("main.GITHUB_TOKEN", "tok")
    @patch("main.DEVIN_API_KEY", "key")
    @patch("main.GITHUB_REPO_NAME", "owner/repo")
    def test_no_error_when_all_set(self) -> None:
        from main import _validate_env

        _validate_env()  # should not raise

    @patch("main.GITHUB_TOKEN", "tok")
    @patch("main.DEVIN_API_KEY", "")
    @patch("main.GITHUB_REPO_NAME", "owner/repo")
    def test_partial_missing(self) -> None:
        from main import _validate_env

        with pytest.raises(ConfigurationError) as exc_info:
            _validate_env()
        resp = exc_info.value.response
        assert resp.details["missing_variables"] == ["DEVIN_API_KEY"]


# =========================================================================
# Integration: fetch_labelled_issues
# =========================================================================


class TestFetchLabelledIssues:
    """fetch_labelled_issues should raise GitHubServiceError on failure."""

    @patch("main.Github")
    def test_repo_access_failure(self, mock_gh_cls: MagicMock) -> None:
        from main import fetch_labelled_issues

        mock_gh_cls.return_value.get_repo.side_effect = Exception("404 Not Found")

        with pytest.raises(GitHubServiceError) as exc_info:
            fetch_labelled_issues("tok", "bad/repo", "label")

        resp = exc_info.value.response
        assert resp.error_code == ERR_GITHUB_REPO_ACCESS
        assert resp.service == SERVICE_GITHUB
        assert resp.details["repo_name"] == "bad/repo"
        assert "404 Not Found" in resp.details["exception"]

    @patch("main.Github")
    def test_success(self, mock_gh_cls: MagicMock) -> None:
        from main import fetch_labelled_issues

        mock_issue = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [mock_issue]
        mock_gh_cls.return_value.get_repo.return_value = mock_repo

        result = fetch_labelled_issues("tok", "owner/repo", "bug")
        assert result == [mock_issue]


# =========================================================================
# Integration: create_devin_session
# =========================================================================


class TestCreateDevinSession:
    """create_devin_session should return standardised error dicts on failure."""

    @patch("main.requests.post")
    def test_http_error(self, mock_post: MagicMock) -> None:
        from main import create_devin_session

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_post.return_value = mock_response

        result = create_devin_session("bad_key", "prompt")

        assert result["error"] is True
        assert result["error_code"] == ERR_DEVIN_API_HTTP
        assert result["service"] == SERVICE_DEVIN_API
        assert result["details"]["status_code"] == 401
        assert "timestamp" in result

    @patch("main.requests.post")
    def test_connection_error(self, mock_post: MagicMock) -> None:
        from main import create_devin_session

        mock_post.side_effect = requests.exceptions.ConnectionError("DNS failed")

        result = create_devin_session("key", "prompt")

        assert result["error"] is True
        assert result["error_code"] == ERR_DEVIN_API_UNREACHABLE
        assert result["service"] == SERVICE_DEVIN_API
        assert "DNS failed" in result["details"]["exception"]

    @patch("main.requests.post")
    def test_success(self, mock_post: MagicMock) -> None:
        from main import create_devin_session

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"session_id": "abc123"}
        mock_post.return_value = mock_response

        result = create_devin_session("key", "prompt")
        assert result == {"session_id": "abc123"}


# =========================================================================
# Integration: post_github_comment
# =========================================================================


class TestPostGithubComment:
    """post_github_comment should produce a standard warning on failure."""

    def test_comment_failure_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from main import post_github_comment

        mock_issue = MagicMock()
        mock_issue.number = 42
        mock_issue.create_comment.side_effect = Exception("Forbidden")

        with caplog.at_level(logging.WARNING, logger="devin_triage"):
            post_github_comment(mock_issue, "https://example.com/session")

        assert ERR_GITHUB_COMMENT_FAILED in caplog.text
        assert "42" in caplog.text

    def test_comment_success(self) -> None:
        from main import post_github_comment

        mock_issue = MagicMock()
        mock_issue.number = 1
        # Should not raise
        post_github_comment(mock_issue, "https://example.com/session")
        mock_issue.create_comment.assert_called_once()


# =========================================================================
# Integration: poll_session_status
# =========================================================================


class TestPollSessionStatus:
    """poll_session_status should use standard error format for poll failures and timeouts."""

    @patch("main.requests.get")
    @patch("main.time.sleep")
    def test_poll_timeout(
        self, mock_sleep: MagicMock, mock_get: MagicMock
    ) -> None:
        from main import poll_session_status

        # Always return a non-terminal status
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status_enum": "running"}
        mock_get.return_value = mock_response

        result = poll_session_status("key", "sess-123", interval=1, timeout=2)

        assert result.get("status") in ("running", "timeout")
        assert "poll_error" in result
        poll_err = result["poll_error"]
        assert poll_err["error"] is True
        assert poll_err["error_code"] == ERR_DEVIN_POLL_TIMEOUT
        assert poll_err["service"] == SERVICE_DEVIN_API

    @patch("main.requests.get")
    @patch("main.time.sleep")
    def test_poll_request_failure_retries(
        self,
        mock_sleep: MagicMock,
        mock_get: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from main import poll_session_status

        # First call fails, second returns terminal
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("timeout"),
            MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"status_enum": "finished"}),
            ),
        ]

        with caplog.at_level(logging.WARNING, logger="devin_triage"):
            result = poll_session_status("key", "sess-456", interval=1, timeout=10)

        assert ERR_DEVIN_POLL_FAILED in caplog.text
        assert result.get("status_enum") == "finished"

    @patch("main.requests.get")
    @patch("main.time.sleep")
    def test_poll_success(
        self, mock_sleep: MagicMock, mock_get: MagicMock
    ) -> None:
        from main import poll_session_status

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status_enum": "finished"}
        mock_get.return_value = mock_response

        result = poll_session_status("key", "sess-789", interval=1, timeout=10)
        assert result["status_enum"] == "finished"
        assert "poll_error" not in result


# =========================================================================
# Integration: _dispatch_single_issue
# =========================================================================


class TestDispatchSingleIssue:
    """_dispatch_single_issue should propagate standard error responses."""

    @patch("main.send_slack_notification")
    @patch("main.post_github_comment")
    @patch("main.create_devin_session")
    def test_api_error_propagated(
        self,
        mock_create: MagicMock,
        mock_comment: MagicMock,
        mock_slack: MagicMock,
    ) -> None:
        from main import _dispatch_single_issue

        mock_create.return_value = ErrorResponse(
            error_code=ERR_DEVIN_API_HTTP,
            message="Devin API returned an HTTP error",
            service=SERVICE_DEVIN_API,
            details={"status_code": 500, "response_body": "Internal Server Error"},
        ).to_dict()

        mock_issue = MagicMock()
        mock_issue.number = 10

        result = _dispatch_single_issue(mock_issue, "owner/repo", "key")

        assert "error" in result
        assert "error_response" in result
        assert result["error_response"]["error_code"] == ERR_DEVIN_API_HTTP
        mock_comment.assert_not_called()
        mock_slack.assert_not_called()

    @patch("main.send_slack_notification")
    @patch("main.post_github_comment")
    @patch("main.create_devin_session")
    def test_success(
        self,
        mock_create: MagicMock,
        mock_comment: MagicMock,
        mock_slack: MagicMock,
    ) -> None:
        from main import _dispatch_single_issue

        mock_create.return_value = {
            "session_id": "sess-abc",
            "url": "https://app.devin.ai/sessions/sess-abc",
        }

        mock_issue = MagicMock()
        mock_issue.number = 5

        result = _dispatch_single_issue(mock_issue, "owner/repo", "key")

        assert result["session_id"] == "sess-abc"
        assert "error" not in result
        mock_comment.assert_called_once()
        mock_slack.assert_called_once()


# =========================================================================
# Integration: dispatch_issues (parallel)
# =========================================================================


class TestDispatchIssues:
    """dispatch_issues should capture exceptions as standardised error dicts."""

    @patch("main._dispatch_single_issue")
    def test_exception_in_thread(self, mock_dispatch: MagicMock) -> None:
        from main import dispatch_issues

        mock_dispatch.side_effect = RuntimeError("boom")

        mock_issue = MagicMock()
        mock_issue.number = 99

        results = dispatch_issues([mock_issue, mock_issue], "owner/repo", "key")

        for r in results:
            assert "error" in r
            assert "error_response" in r
            assert r["error_response"]["error_code"] == ERR_DISPATCH_FAILED


# =========================================================================
# Error code constants
# =========================================================================


class TestErrorCodes:
    """Ensure all error code constants are unique non-empty strings."""

    def test_all_codes_unique(self) -> None:
        codes = [
            ERR_MISSING_ENV_VARS,
            ERR_GITHUB_REPO_ACCESS,
            ERR_GITHUB_COMMENT_FAILED,
            ERR_DEVIN_API_HTTP,
            ERR_DEVIN_API_UNREACHABLE,
            ERR_DEVIN_POLL_FAILED,
            ERR_DEVIN_POLL_TIMEOUT,
            ERR_DISPATCH_FAILED,
        ]
        assert len(codes) == len(set(codes))
        for code in codes:
            assert isinstance(code, str)
            assert len(code) > 0

    def test_all_services_unique(self) -> None:
        services = [SERVICE_CONFIG, SERVICE_GITHUB, SERVICE_DEVIN_API, SERVICE_DISPATCH]
        assert len(services) == len(set(services))


# =========================================================================
# Standardised format consistency
# =========================================================================


class TestStandardFormat:
    """All error responses must contain the mandatory fields."""

    REQUIRED_KEYS = {"error", "error_code", "message", "service", "timestamp", "details"}

    def test_error_response_has_all_required_keys(self) -> None:
        resp = ErrorResponse(
            error_code="CHK",
            message="check",
            service="svc",
        )
        d = resp.to_dict()
        assert self.REQUIRED_KEYS.issubset(d.keys())

    @patch("main.requests.post")
    def test_create_devin_session_error_has_all_keys(
        self, mock_post: MagicMock
    ) -> None:
        from main import create_devin_session

        mock_post.side_effect = requests.exceptions.ConnectionError("fail")
        result = create_devin_session("k", "p")
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_validate_env_error_has_all_keys(self) -> None:
        from main import _validate_env

        with patch("main.GITHUB_TOKEN", ""), patch("main.DEVIN_API_KEY", ""), patch(
            "main.GITHUB_REPO_NAME", ""
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                _validate_env()
            d = exc_info.value.response.to_dict()
            assert self.REQUIRED_KEYS.issubset(d.keys())
