"""Tests for the Devin Issue Triage & Resolution CLI.

Covers the crypto migration from hashlib to the cryptography library (v3+)
and other core helper functions.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from main import (
    _validate_env,
    build_devin_prompt,
    mock_llm_evaluate,
    print_triage_report,
    send_slack_notification,
)


# ---------------------------------------------------------------------------
# mock_llm_evaluate – crypto library migration tests
# ---------------------------------------------------------------------------


class TestMockLlmEvaluate:
    """Verify that mock_llm_evaluate uses the cryptography library and
    produces deterministic, correct results."""

    def test_returns_expected_keys(self) -> None:
        result = mock_llm_evaluate("title", "body")
        assert "complexity" in result
        assert "confidence" in result

    def test_complexity_is_valid(self) -> None:
        result = mock_llm_evaluate("any title", "any body")
        assert result["complexity"] in ("Low", "Medium", "High")

    def test_confidence_range(self) -> None:
        result = mock_llm_evaluate("any title", "any body")
        assert 50 <= result["confidence"] <= 99

    def test_deterministic_output(self) -> None:
        """Same inputs must always produce the same scores."""
        r1 = mock_llm_evaluate("Issue A", "Description A")
        r2 = mock_llm_evaluate("Issue A", "Description A")
        assert r1 == r2

    def test_different_inputs_can_differ(self) -> None:
        """Different inputs should (generally) produce different scores."""
        r1 = mock_llm_evaluate("Issue A", "Description A")
        r2 = mock_llm_evaluate("Issue B", "Description B")
        # At minimum the raw hash differs; scores *may* collide but
        # for these specific inputs they don't.
        assert r1 != r2

    def test_matches_hashlib_sha256(self) -> None:
        """The cryptography-based hash must produce the same digest as
        hashlib.sha256 so that existing scores remain stable."""
        title, body = "Upgrade DB", "Migrate Postgres to v15"
        data = f"{title}:{body}".encode()
        expected_hex = hashlib.sha256(data).hexdigest()

        # Derive the same intermediate value the function uses
        from cryptography.hazmat.primitives import hashes

        h = hashes.Hash(hashes.SHA256())
        h.update(data)
        actual_hex = h.finalize().hex()

        assert actual_hex == expected_hex

    def test_uses_cryptography_library(self) -> None:
        """Ensure the function goes through cryptography.hazmat, not hashlib."""
        with patch("main.hashes") as mock_hashes:
            mock_hash_obj = MagicMock()
            mock_hash_obj.finalize.return_value = b"\x00" * 32
            mock_hashes.Hash.return_value = mock_hash_obj
            mock_hashes.SHA256 = MagicMock()

            mock_llm_evaluate("t", "b")

            mock_hashes.Hash.assert_called_once()
            mock_hash_obj.update.assert_called_once()
            mock_hash_obj.finalize.assert_called_once()

    def test_empty_body(self) -> None:
        result = mock_llm_evaluate("title", "")
        assert result["complexity"] in ("Low", "Medium", "High")
        assert 50 <= result["confidence"] <= 99


# ---------------------------------------------------------------------------
# _validate_env
# ---------------------------------------------------------------------------


class TestValidateEnv:
    def test_exits_when_missing_vars(self) -> None:
        with patch.dict(
            "os.environ",
            {"GITHUB_TOKEN": "", "DEVIN_API_KEY": "", "GITHUB_REPO_NAME": ""},
            clear=False,
        ), patch("main.GITHUB_TOKEN", ""), patch("main.DEVIN_API_KEY", ""), patch(
            "main.GITHUB_REPO_NAME", ""
        ):
            with pytest.raises(SystemExit):
                _validate_env()


# ---------------------------------------------------------------------------
# build_devin_prompt
# ---------------------------------------------------------------------------


class TestBuildDevinPrompt:
    def test_prompt_contains_issue_details(self) -> None:
        issue = MagicMock()
        issue.number = 42
        issue.title = "Fix login bug"
        issue.body = "Users cannot log in"
        prompt = build_devin_prompt(issue, "owner/repo")
        assert "#42" in prompt
        assert "Fix login bug" in prompt
        assert "owner/repo" in prompt


# ---------------------------------------------------------------------------
# print_triage_report
# ---------------------------------------------------------------------------


class TestPrintTriageReport:
    def test_returns_lookup_dict(self, capsys: pytest.CaptureFixture[str]) -> None:
        issue = MagicMock()
        issue.number = 7
        issue.title = "Some issue"
        issue.body = "body"
        lookup = print_triage_report([issue])
        assert 7 in lookup
        assert lookup[7] is issue


# ---------------------------------------------------------------------------
# send_slack_notification
# ---------------------------------------------------------------------------


class TestSendSlackNotification:
    def test_prints_notification(self, capsys: pytest.CaptureFixture[str]) -> None:
        send_slack_notification(99, "https://example.com/session/abc")
        captured = capsys.readouterr()
        assert "Issue #99" in captured.out
        assert "https://example.com/session/abc" in captured.out
