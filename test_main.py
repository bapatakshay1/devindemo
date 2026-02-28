"""
Tests for standardised error handling in main.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from errors import (
    DEVIN_API_CONNECTION_ERROR,
    DEVIN_API_HTTP_ERROR,
    GITHUB_COMMENT_ERROR,
    GITHUB_REPO_ERROR,
    USER_INPUT_ERROR,
    VALIDATION_ERROR,
    ConfigurationError,
    DevinAPIError,
    GitHubServiceError,
)


# ---------------------------------------------------------------------------
# _validate_env
# ---------------------------------------------------------------------------


class TestValidateEnv:
    """Tests for the _validate_env function."""

    @patch("main.GITHUB_TOKEN", "")
    @patch("main.DEVIN_API_KEY", "")
    @patch("main.GITHUB_REPO_NAME", "")
    def test_all_missing(self) -> None:
        from main import _validate_env

        with pytest.raises(ConfigurationError) as exc_info:
            _validate_env()
        err = exc_info.value.error_response
        assert err.code == VALIDATION_ERROR
        assert err.service == "env_validator"
        assert set(err.details["missing_variables"]) == {
            "GITHUB_TOKEN",
            "DEVIN_API_KEY",
            "GITHUB_REPO_NAME",
        }

    @patch("main.GITHUB_TOKEN", "tok")
    @patch("main.DEVIN_API_KEY", "key")
    @patch("main.GITHUB_REPO_NAME", "owner/repo")
    def test_all_present(self) -> None:
        from main import _validate_env

        # Should NOT raise
        _validate_env()

    @patch("main.GITHUB_TOKEN", "tok")
    @patch("main.DEVIN_API_KEY", "")
    @patch("main.GITHUB_REPO_NAME", "owner/repo")
    def test_single_missing(self) -> None:
        from main import _validate_env

        with pytest.raises(ConfigurationError) as exc_info:
            _validate_env()
        err = exc_info.value.error_response
        assert err.details["missing_variables"] == ["DEVIN_API_KEY"]


# ---------------------------------------------------------------------------
# fetch_labelled_issues
# ---------------------------------------------------------------------------


class TestFetchLabelledIssues:
    """Tests for fetch_labelled_issues error path."""

    @patch("main.Github")
    def test_repo_not_found_raises_github_service_error(
        self, mock_github_cls: MagicMock
    ) -> None:
        from main import fetch_labelled_issues

        mock_gh = MagicMock()
        mock_gh.get_repo.side_effect = Exception("Not Found")
        mock_github_cls.return_value = mock_gh

        with pytest.raises(GitHubServiceError) as exc_info:
            fetch_labelled_issues("tok", "bad/repo", "label")

        err = exc_info.value.error_response
        assert err.code == GITHUB_REPO_ERROR
        assert err.service == "github_integration"
        assert err.details["repository"] == "bad/repo"
        assert "Not Found" in err.details["original_error"]

    @patch("main.Github")
    def test_success_returns_issues(self, mock_github_cls: MagicMock) -> None:
        from main import fetch_labelled_issues

        mock_issue = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [mock_issue]
        mock_gh = MagicMock()
        mock_gh.get_repo.return_value = mock_repo
        mock_github_cls.return_value = mock_gh

        result = fetch_labelled_issues("tok", "owner/repo", "label")
        assert result == [mock_issue]


# ---------------------------------------------------------------------------
# create_devin_session
# ---------------------------------------------------------------------------


class TestCreateDevinSession:
    """Tests for create_devin_session error paths."""

    @patch("main.requests.post")
    def test_http_error_raises_devin_api_error(
        self, mock_post: MagicMock
    ) -> None:
        from main import create_devin_session

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        http_err = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_err
        mock_post.return_value = mock_response

        with pytest.raises(DevinAPIError) as exc_info:
            create_devin_session("bad-key", "prompt")

        err = exc_info.value.error_response
        assert err.code == DEVIN_API_HTTP_ERROR
        assert err.service == "devin_api"
        assert err.details["status_code"] == 403
        assert err.details["response_body"] == "Forbidden"

    @patch("main.requests.post")
    def test_connection_error_raises_devin_api_error(
        self, mock_post: MagicMock
    ) -> None:
        from main import create_devin_session

        mock_post.side_effect = requests.exceptions.ConnectionError("refused")

        with pytest.raises(DevinAPIError) as exc_info:
            create_devin_session("key", "prompt")

        err = exc_info.value.error_response
        assert err.code == DEVIN_API_CONNECTION_ERROR
        assert err.service == "devin_api"
        assert "refused" in err.details["original_error"]

    @patch("main.requests.post")
    def test_success_returns_data(self, mock_post: MagicMock) -> None:
        from main import create_devin_session

        mock_response = MagicMock()
        mock_response.json.return_value = {"session_id": "abc"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = create_devin_session("key", "prompt")
        assert result == {"session_id": "abc"}


# ---------------------------------------------------------------------------
# post_github_comment
# ---------------------------------------------------------------------------


class TestPostGithubComment:
    """Tests for post_github_comment error path (non-fatal)."""

    def test_comment_failure_is_non_fatal(self) -> None:
        from main import post_github_comment

        mock_issue = MagicMock()
        mock_issue.number = 42
        mock_issue.create_comment.side_effect = Exception("API rate limit")

        # Should NOT raise -- error is handled non-fatally
        post_github_comment(mock_issue, "https://example.com/session")

    def test_comment_failure_uses_standardised_format(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from main import post_github_comment

        mock_issue = MagicMock()
        mock_issue.number = 7
        mock_issue.create_comment.side_effect = Exception("timeout")

        with caplog.at_level("WARNING"):
            post_github_comment(mock_issue, "https://example.com/s")

        assert GITHUB_COMMENT_ERROR in caplog.text
        assert "github_integration" in caplog.text

    def test_comment_success(self) -> None:
        from main import post_github_comment

        mock_issue = MagicMock()
        mock_issue.number = 1
        mock_issue.create_comment.return_value = MagicMock()

        # Should succeed without any error
        post_github_comment(mock_issue, "https://example.com/s")
        mock_issue.create_comment.assert_called_once()


# ---------------------------------------------------------------------------
# prompt_user_selection
# ---------------------------------------------------------------------------


class TestPromptUserSelection:
    """Tests for prompt_user_selection error paths."""

    @patch("builtins.input", side_effect=["abc", "q"])
    def test_invalid_input_logged_as_user_input_error(
        self, _mock_input: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from main import prompt_user_selection

        mock_issue = MagicMock()
        lookup = {1: mock_issue}

        with caplog.at_level("WARNING"):
            result = prompt_user_selection(lookup)

        assert result is None
        assert USER_INPUT_ERROR in caplog.text
        assert "user_input" in caplog.text

    @patch("builtins.input", side_effect=["999", "q"])
    def test_issue_not_in_lookup_logged_as_user_input_error(
        self, _mock_input: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from main import prompt_user_selection

        mock_issue = MagicMock()
        lookup = {1: mock_issue}

        with caplog.at_level("WARNING"):
            result = prompt_user_selection(lookup)

        assert result is None
        assert USER_INPUT_ERROR in caplog.text

    @patch("builtins.input", return_value="1")
    def test_valid_selection(self, _mock_input: MagicMock) -> None:
        from main import prompt_user_selection

        mock_issue = MagicMock()
        lookup = {1: mock_issue}

        result = prompt_user_selection(lookup)
        assert result is mock_issue


# ---------------------------------------------------------------------------
# main() top-level handler
# ---------------------------------------------------------------------------


class TestMainErrorHandler:
    """Tests that main() catches PipelineError subclasses via handle_error."""

    @patch("main._run_pipeline")
    def test_configuration_error_exits(self, mock_pipeline: MagicMock) -> None:
        from errors import build_error_response
        from main import main

        err = build_error_response(
            code=VALIDATION_ERROR,
            message="missing vars",
            service="env_validator",
        )
        mock_pipeline.side_effect = ConfigurationError(err)

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch("main._run_pipeline")
    def test_github_service_error_exits(self, mock_pipeline: MagicMock) -> None:
        from errors import build_error_response
        from main import main

        err = build_error_response(
            code=GITHUB_REPO_ERROR,
            message="repo gone",
            service="github_integration",
        )
        mock_pipeline.side_effect = GitHubServiceError(err)

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch("main._run_pipeline")
    def test_devin_api_error_exits(self, mock_pipeline: MagicMock) -> None:
        from errors import build_error_response
        from main import main

        err = build_error_response(
            code=DEVIN_API_HTTP_ERROR,
            message="http 500",
            service="devin_api",
        )
        mock_pipeline.side_effect = DevinAPIError(err)

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
