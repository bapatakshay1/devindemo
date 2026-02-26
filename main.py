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
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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


def mock_llm_evaluate(issue: Issue) -> dict[str, Any]:
    """Simulate an LLM evaluation using realistic heuristics.

    Scores are derived from observable issue metadata so results feel
    plausible in a demo.  The function is still deterministic (same issue
    always yields the same scores).

    Heuristics used:
    * **Body length** -- longer descriptions suggest more complex work.
    * **Label count** -- more labels often means cross-cutting concerns.
    * **Comment count** -- lots of discussion implies ambiguity / complexity.
    * **Issue age** -- older issues may have bit-rotted and need more context.

    Args:
        issue: A GitHub issue object.

    Returns:
        A dict with keys:
        - ``complexity``  : "Low" / "Medium" / "High"
        - ``confidence``  : int percentage 50-99
        - ``staleness``   : "Fresh" / "Aging" / "Stale"
        - ``age_days``    : int
        - ``signals``     : dict of raw heuristic values
    """
    title = issue.title
    body = issue.body or ""
    label_count = len(list(issue.labels))
    comment_count = issue.comments  # int provided by the API
    body_length = len(body)

    now = datetime.now(tz=timezone.utc)
    created = (
        issue.created_at.replace(tzinfo=timezone.utc)
        if issue.created_at.tzinfo is None
        else issue.created_at
    )
    age_days = (now - created).days

    # --- Complexity score (0-100) -------------------------------------------
    # Each signal contributes a weighted sub-score.
    body_score = min(body_length / 2000, 1.0) * 30  # max 30 pts
    label_score = min(label_count / 5, 1.0) * 20  # max 20 pts
    comment_score = min(comment_count / 10, 1.0) * 25  # max 25 pts
    age_score = min(age_days / 180, 1.0) * 25  # max 25 pts

    raw_complexity = body_score + label_score + comment_score + age_score

    # Add a small deterministic jitter so identical-metadata issues differ
    digest = hashlib.sha256(f"{title}:{body}".encode()).hexdigest()
    jitter = (int(digest[:4], 16) % 10) - 5  # range -5..+4
    raw_complexity = max(0, min(100, raw_complexity + jitter))

    if raw_complexity < 35:
        complexity = "Low"
    elif raw_complexity < 65:
        complexity = "Medium"
    else:
        complexity = "High"

    # --- Confidence score (50-99) -------------------------------------------
    # More data points -> higher confidence.
    data_richness = sum(
        [
            1 if body_length > 100 else 0,
            1 if label_count >= 1 else 0,
            1 if comment_count >= 1 else 0,
            1 if age_days > 7 else 0,
        ]
    )
    base_confidence = 55 + data_richness * 10  # 55-95
    # Deterministic micro-jitter
    confidence = min(99, max(50, base_confidence + (int(digest[4:6], 16) % 5)))

    # --- Staleness label ----------------------------------------------------
    if age_days < 14:
        staleness = "Fresh"
    elif age_days < 90:
        staleness = "Aging"
    else:
        staleness = "Stale"

    return {
        "complexity": complexity,
        "complexity_score": math.floor(raw_complexity),
        "confidence": confidence,
        "staleness": staleness,
        "age_days": age_days,
        "signals": {
            "body_length": body_length,
            "label_count": label_count,
            "comment_count": comment_count,
            "age_days": age_days,
        },
    }


def print_triage_report(issues: list[Issue]) -> dict[int, Issue]:
    """Print a formatted triage report and return a lookup dict.

    The report now includes staleness, age, and the raw complexity score
    alongside the categorical labels, giving the operator more signal.

    Args:
        issues: List of GitHub issues to report on.

    Returns:
        A mapping of issue number -> Issue for quick lookup.
    """
    separator = "=" * 96
    print(f"\n{separator}")
    print("  FINSERV CO  --  AUTOMATED ISSUE TRIAGE REPORT")
    print(separator)
    print(
        f"{'#':<8} {'Complexity':<12} {'Score':<7} {'Confidence':<12} "
        f"{'Staleness':<10} {'Age':<8} {'Title'}"
    )
    print("-" * 96)

    lookup: dict[int, Issue] = {}
    for issue in issues:
        scores = mock_llm_evaluate(issue)
        age_str = f"{scores['age_days']}d"
        print(
            f"#{issue.number:<7} {scores['complexity']:<12} "
            f"{scores['complexity_score']:<7} "
            f"{scores['confidence']}%{'':<10} "
            f"{scores['staleness']:<10} {age_str:<8} "
            f"{issue.title[:40]}"
        )
        lookup[issue.number] = issue

    print(separator)
    print(
        "  Signals: body length, label count, comment count, issue age. "
        "Score range 0-100."
    )
    print(separator + "\n")
    return lookup


# ---------------------------------------------------------------------------
# Phase 2 -- Human-in-the-Loop
# ---------------------------------------------------------------------------


def prompt_user_selection(lookup: dict[int, Issue]) -> list[Issue]:
    """Interactively ask the operator to select one or more issues.

    Supports batch selection via comma-separated issue numbers, the keyword
    ``all`` to select every issue in the triage report, or ``q`` to quit.

    Args:
        lookup: Mapping of issue number to Issue object.

    Returns:
        A list of selected :class:`Issue` objects (may be empty if the user
        quits).
    """
    print(
        "Tip: Enter comma-separated numbers (e.g. 1,4,7) or 'all' to select every issue."
    )
    while True:
        choice = input(
            "Enter Issue Number(s) for Devin to resolve (or 'q' to quit): "
        ).strip()

        if choice.lower() == "q":
            logger.info("User chose to quit. Exiting.")
            return []

        if choice.lower() == "all":
            selected = list(lookup.values())
            logger.info("User selected ALL %d issues.", len(selected))
            return selected

        # Parse comma-separated numbers
        raw_numbers = [tok.strip() for tok in choice.split(",") if tok.strip()]
        selected: list[Issue] = []
        invalid = False
        for tok in raw_numbers:
            try:
                num = int(tok)
            except ValueError:
                print(f"  Invalid input '{tok}'. Please enter numbers or 'q'.")
                invalid = True
                break
            if num not in lookup:
                print(f"  Issue #{num} is not in the triage report. Try again.")
                invalid = True
                break
            selected.append(lookup[num])

        if invalid:
            continue

        if not selected:
            print("  No issues selected. Try again.")
            continue

        # Deduplicate while preserving order
        seen: set[int] = set()
        deduped: list[Issue] = []
        for iss in selected:
            if iss.number not in seen:
                seen.add(iss.number)
                deduped.append(iss)

        logger.info(
            "User selected %d issue(s): %s", len(deduped), [i.number for i in deduped]
        )
        return deduped


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
        The parsed JSON response from the API.  On failure returns a dict
        with an ``error`` key so batch dispatch can continue.
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
        msg = (
            f"HTTP {exc.response.status_code}: {exc.response.text}"
            if exc.response is not None
            else str(exc)
        )
        logger.error("Devin API error: %s", msg)
        return {"error": msg}
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to reach Devin API: %s", exc)
        return {"error": str(exc)}

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


def _dispatch_single_issue(
    issue: Issue,
    repo_name: str,
    api_key: str,
) -> dict[str, Any]:
    """Build prompt, create a Devin session, comment, and notify for one issue.

    Returns a result dict with ``issue_number``, ``session_id``,
    ``session_url``, and optionally ``error``.
    """
    prompt = build_devin_prompt(issue, repo_name)
    response_data = create_devin_session(api_key, prompt)

    if "error" in response_data:
        return {
            "issue_number": issue.number,
            "error": response_data["error"],
        }

    session_id = response_data.get("session_id", "unknown")
    session_url = response_data.get(
        "url", f"https://app.devin.ai/sessions/{session_id}"
    )

    logger.info("Issue #%d -> Session %s (%s)", issue.number, session_id, session_url)

    # Post comment on the GitHub issue
    post_github_comment(issue, session_url)

    # Mock Slack notification
    send_slack_notification(issue.number, session_url)

    return {
        "issue_number": issue.number,
        "session_id": session_id,
        "session_url": session_url,
    }


def dispatch_issues(
    selected_issues: list[Issue],
    repo_name: str,
    api_key: str,
    max_workers: int = 5,
) -> list[dict[str, Any]]:
    """Dispatch Devin sessions for multiple issues in parallel.

    Args:
        selected_issues: Issues chosen by the operator.
        repo_name: Full ``owner/repo`` repository name.
        api_key: Devin API bearer token.
        max_workers: Maximum parallel Devin API calls.

    Returns:
        A list of result dicts, one per issue.
    """
    if len(selected_issues) == 1:
        return [_dispatch_single_issue(selected_issues[0], repo_name, api_key)]

    logger.info(
        "Dispatching %d issues in parallel (max_workers=%d)...",
        len(selected_issues),
        max_workers,
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_issue = {
            executor.submit(_dispatch_single_issue, issue, repo_name, api_key): issue
            for issue in selected_issues
        }
        for future in as_completed(future_to_issue):
            issue = future_to_issue[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error("Dispatch failed for Issue #%d: %s", issue.number, exc)
                result = {"issue_number": issue.number, "error": str(exc)}
            results.append(result)

    return results


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
    selected_issues = prompt_user_selection(lookup)
    if not selected_issues:
        sys.exit(0)

    logger.info(
        "Dispatching Devin for %d issue(s): %s",
        len(selected_issues),
        [i.number for i in selected_issues],
    )

    # 3 & 4. Parallel Devin API Execution + Communication
    results = dispatch_issues(selected_issues, GITHUB_REPO_NAME, DEVIN_API_KEY)

    # Print dispatch summary
    successes = [r for r in results if "error" not in r]
    failures = [r for r in results if "error" in r]

    print("\n" + "=" * 80)
    print("  DISPATCH SUMMARY")
    print("=" * 80)
    print(f"  Total dispatched : {len(results)}")
    print(f"  Succeeded        : {len(successes)}")
    print(f"  Failed           : {len(failures)}")
    if failures:
        for f in failures:
            print(f"    - Issue #{f['issue_number']}: {f['error']}")
    print("=" * 80 + "\n")

    logger.info("Pipeline complete. %d session(s) created.", len(successes))


if __name__ == "__main__":
    main()
