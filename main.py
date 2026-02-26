#!/usr/bin/env python3
"""
Devin Issue Triage & Resolution CLI
====================================
Orchestrates the Devin API to automatically triage and resolve stale GitHub
issues for a mock enterprise client ("FinServ Co").

Phases:
  1. Automated Triage   -- fetch labelled issues, score them
  2. Human-in-the-Loop  -- prompt the operator to pick an issue
  3. Execution           -- kick off a Devin session via the v3 API
  4. Communication       -- mock Slack webhook notification
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv
from github import Github
from github.Issue import Issue

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------

load_dotenv()

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("devin_triage")

GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
DEVIN_API_KEY: str = os.getenv("DEVIN_API_KEY", "")
GITHUB_REPO_NAME: str = os.getenv("GITHUB_REPO_NAME", "")
GITHUB_ISSUE_LABEL: str = os.getenv("GITHUB_ISSUE_LABEL", "devin-backlog")

DEVIN_API_URL = "https://api.devin.ai/v3/sessions"


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------


def _validate_env() -> None:
    """Ensure all required environment variables are set."""
    missing: list[str] = []
    if not GITHUB_TOKEN:
        missing.append("GITHUB_TOKEN")
    if not DEVIN_API_KEY:
        missing.append("DEVIN_API_KEY")
    if not GITHUB_REPO_NAME:
        missing.append("GITHUB_REPO_NAME")
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        sys.exit(1)


def fetch_labelled_issues(
    token: str,
    repo_name: str,
    label: str,
) -> list[Issue]:
    """Connect to GitHub and return all open issues with the given label.

    Args:
        token: GitHub personal-access token.
        repo_name: Full repository name in ``owner/repo`` format.
        label: The label used to filter issues (e.g. ``devin-backlog``).

    Returns:
        A list of :class:`github.Issue.Issue` objects.
    """
    logger.info("Connecting to GitHub repository: %s", repo_name)
    gh = Github(token)
    try:
        repo = gh.get_repo(repo_name)
    except Exception as exc:
        logger.error("Failed to access repository '%s': %s", repo_name, exc)
        sys.exit(1)

    logger.info("Fetching open issues with label '%s'...", label)
    issues: list[Issue] = list(repo.get_issues(state="open", labels=[label]))
    logger.info("Found %d issue(s).", len(issues))
    return issues


def mock_llm_evaluate(title: str, body: str) -> dict[str, Any]:
    """Simulate an LLM evaluation of an issue's complexity.

    Uses a deterministic hash so the same issue always returns the same
    scores, making demos reproducible.

    Args:
        title: The issue title.
        body: The issue body / description.

    Returns:
        A dict with ``complexity`` (Low / Medium / High) and
        ``confidence`` (int percentage 50-99).
    """
    digest = hashlib.sha256(f"{title}:{body}".encode()).hexdigest()
    hash_int = int(digest[:8], 16)

    complexity_levels = ["Low", "Medium", "High"]
    complexity = complexity_levels[hash_int % 3]

    confidence = 50 + (hash_int % 50)  # range 50-99

    return {"complexity": complexity, "confidence": confidence}


def print_triage_report(issues: list[Issue]) -> dict[int, Issue]:
    """Print a formatted triage report and return a lookup dict.

    Args:
        issues: List of GitHub issues to report on.

    Returns:
        A mapping of issue number -> Issue for quick lookup.
    """
    separator = "=" * 80
    print(f"\n{separator}")
    print("  FINSERV CO  --  AUTOMATED ISSUE TRIAGE REPORT")
    print(separator)
    print(f"{'#':<8} {'Complexity':<12} {'Confidence':<12} {'Title'}")
    print("-" * 80)

    lookup: dict[int, Issue] = {}
    for issue in issues:
        scores = mock_llm_evaluate(issue.title, issue.body or "")
        print(
            f"#{issue.number:<7} {scores['complexity']:<12} "
            f"{scores['confidence']}%{'':<10} {issue.title[:50]}"
        )
        lookup[issue.number] = issue

    print(separator + "\n")
    return lookup


# ---------------------------------------------------------------------------
# Phase 2 -- Human-in-the-Loop
# ---------------------------------------------------------------------------


def prompt_user_selection(lookup: dict[int, Issue]) -> Issue | None:
    """Interactively ask the operator to select an issue.

    Args:
        lookup: Mapping of issue number to Issue object.

    Returns:
        The selected :class:`Issue`, or ``None`` if the user quits.
    """
    while True:
        choice = input(
            "Enter the Issue Number you would like Devin to resolve (or 'q' to quit): "
        ).strip()

        if choice.lower() == "q":
            logger.info("User chose to quit. Exiting.")
            return None

        try:
            issue_number = int(choice)
        except ValueError:
            print(f"  Invalid input '{choice}'. Please enter a number or 'q'.")
            continue

        if issue_number not in lookup:
            print(f"  Issue #{issue_number} is not in the triage report. Try again.")
            continue

        return lookup[issue_number]


# ---------------------------------------------------------------------------
# Phase 3 -- Devin API Execution
# ---------------------------------------------------------------------------


def build_devin_prompt(issue: Issue, repo_name: str) -> str:
    """Construct a detailed prompt for the Devin session.

    Args:
        issue: The GitHub issue to resolve.
        repo_name: Full ``owner/repo`` repository name.

    Returns:
        The prompt string to send to the Devin API.
    """
    prompt = (
        f"You are resolving GitHub Issue #{issue.number} in the repository "
        f"'{repo_name}'.\n\n"
        f"## Issue Details\n"
        f"- **Title:** {issue.title}\n"
        f"- **Number:** #{issue.number}\n"
        f"- **Body:**\n{issue.body or '(no description provided)'}\n\n"
        f"## Instructions\n"
        f"1. Clone the repository: `git clone https://github.com/{repo_name}.git`\n"
        f"2. Read and understand Issue #{issue.number} thoroughly.\n"
        f"3. Create a new branch named `fix/issue-{issue.number}`.\n"
        f"4. Implement the fix described in the issue.\n"
        f"5. Write or update tests to cover your changes.\n"
        f"6. Run the full test suite and ensure all tests pass.\n"
        f"7. Commit your changes with a clear commit message referencing "
        f"Issue #{issue.number}.\n"
        f"8. Push the branch and open a Pull Request against `main` with a "
        f"descriptive title and body that links to Issue #{issue.number}.\n"
    )
    return prompt


def create_devin_session(
    api_key: str,
    prompt: str,
) -> dict[str, Any]:
    """Send a POST request to the Devin v3 API to start a session.

    Args:
        api_key: Devin API bearer token.
        prompt: The detailed prompt for Devin.

    Returns:
        The parsed JSON response from the API.

    Raises:
        SystemExit: If the API call fails.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"prompt": prompt}

    logger.info("Creating Devin session via %s ...", DEVIN_API_URL)
    try:
        response = requests.post(
            DEVIN_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        logger.error(
            "Devin API returned HTTP %s: %s",
            exc.response.status_code if exc.response is not None else "N/A",
            exc.response.text if exc.response is not None else str(exc),
        )
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to reach Devin API: %s", exc)
        sys.exit(1)

    data: dict[str, Any] = response.json()
    logger.info("Devin session created successfully.")
    return data


def post_github_comment(issue: Issue, session_url: str) -> None:
    """Post an automation-triggered comment on the GitHub issue.

    Uses PyGithub to add a comment notifying watchers that Devin has
    started working on the issue.

    Args:
        issue: The GitHub issue to comment on.
        session_url: The Devin session URL for tracking progress.
    """
    comment_body = (
        "\U0001f916 **Devin Automation Triggered:** An autonomous agent is "
        "working on this issue.\n\n"
        f"View the session here: {session_url}"
    )
    try:
        issue.create_comment(comment_body)
        logger.info("Posted automation comment on Issue #%d.", issue.number)
    except Exception as exc:
        logger.warning(
            "Failed to post comment on Issue #%d: %s. Continuing anyway.",
            issue.number,
            exc,
        )


# ---------------------------------------------------------------------------
# Phase 4 -- Communication (mock Slack notification)
# ---------------------------------------------------------------------------


def send_slack_notification(issue_number: int, session_url: str) -> None:
    """Print a mock Slack webhook notification to the terminal.

    Formats the message as a Slack-style threaded notification, indicating
    that all subsequent progress updates will be posted as thread replies
    to keep the channel clean.

    Args:
        issue_number: The GitHub issue number being resolved.
        session_url: The Devin session URL for tracking progress.
    """
    border = "*" * 72
    print(f"\n{border}")
    print(
        f"  \U0001f514 SLACK WEBHOOK FIRED: Devin has begun work on "
        f"Issue #{issue_number}."
    )
    print(f"  Track progress here: {session_url}")
    print()
    print(
        "  \U0001f9f5 All progress updates will be posted as thread replies "
        "to this message"
    )
    print("     to avoid channel noise. Follow the thread for real-time status.")
    print(f"{border}\n")


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full triage-and-resolve pipeline."""
    logger.info("Starting FinServ Co Issue Triage Pipeline")

    # 0. Validate environment
    _validate_env()

    # 1. Automated Triage
    issues = fetch_labelled_issues(GITHUB_TOKEN, GITHUB_REPO_NAME, GITHUB_ISSUE_LABEL)
    if not issues:
        logger.warning(
            "No open issues found with label '%s' in %s. Nothing to triage.",
            GITHUB_ISSUE_LABEL,
            GITHUB_REPO_NAME,
        )
        sys.exit(0)

    lookup = print_triage_report(issues)

    # 2. Human-in-the-Loop
    selected_issue = prompt_user_selection(lookup)
    if selected_issue is None:
        sys.exit(0)

    logger.info(
        "User selected Issue #%d: %s",
        selected_issue.number,
        selected_issue.title,
    )

    # 3. Devin API Execution
    prompt = build_devin_prompt(selected_issue, GITHUB_REPO_NAME)
    logger.debug("Devin prompt:\n%s", prompt)

    response_data = create_devin_session(DEVIN_API_KEY, prompt)

    session_id = response_data.get("session_id", "unknown")
    session_url = response_data.get(
        "url", f"https://app.devin.ai/sessions/{session_id}"
    )

    logger.info("Devin Session ID : %s", session_id)
    logger.info("Devin Session URL: %s", session_url)

    # 3b. Post a comment on the GitHub issue
    post_github_comment(selected_issue, session_url)

    # 4. Communication
    send_slack_notification(selected_issue.number, session_url)

    logger.info(
        "Pipeline complete. Devin is now working on Issue #%d.", selected_issue.number
    )


if __name__ == "__main__":
    main()
