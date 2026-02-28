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
import time
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
DEVIN_POLL_INTERVAL: int = int(os.getenv("DEVIN_POLL_INTERVAL", "30"))  # seconds
DEVIN_POLL_TIMEOUT: int = int(os.getenv("DEVIN_POLL_TIMEOUT", "600"))  # seconds
SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "")


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


def _evaluate_all_issues(
    issues: list[Issue],
) -> list[tuple[Issue, dict[str, Any]]]:
    """Evaluate all issues and return (issue, scores) pairs sorted by score descending."""
    evaluated: list[tuple[Issue, dict[str, Any]]] = []
    for issue in issues:
        scores = mock_llm_evaluate(issue)
        evaluated.append((issue, scores))
    # Sort by complexity_score descending (highest priority first)
    evaluated.sort(key=lambda pair: pair[1]["complexity_score"], reverse=True)
    return evaluated


def prompt_complexity_filter() -> str | None:
    """Ask the operator whether to filter the triage report by complexity.

    Returns:
        A complexity level string ("Low", "Medium", "High") to filter by,
        or ``None`` to show all issues.
    """
    print(
        "\nFilter by complexity? Enter one of: Low, Medium, High "
        "(or press Enter to show all)"
    )
    choice = input("  Filter: ").strip().capitalize()
    if choice in ("Low", "Medium", "High"):
        logger.info("Applying complexity filter: %s", choice)
        return choice
    return None


def print_triage_report(issues: list[Issue]) -> dict[int, Issue]:
    """Print a formatted triage report and return a lookup dict.

    Issues are **sorted by complexity score** (highest first) so the
    operator immediately sees the most impactful work.  An optional
    complexity filter is offered before printing.

    Args:
        issues: List of GitHub issues to report on.

    Returns:
        A mapping of issue number -> Issue for quick lookup.
    """
    evaluated = _evaluate_all_issues(issues)

    # Offer optional complexity filter
    complexity_filter = prompt_complexity_filter()
    if complexity_filter is not None:
        evaluated = [
            (iss, sc) for iss, sc in evaluated if sc["complexity"] == complexity_filter
        ]
        if not evaluated:
            print(
                f"  No issues matched complexity '{complexity_filter}'. "
                "Showing all issues instead."
            )
            evaluated = _evaluate_all_issues(issues)

    separator = "=" * 96
    print(f"\n{separator}")
    print("  FINSERV CO  --  AUTOMATED ISSUE TRIAGE REPORT")
    if complexity_filter:
        print(f"  (Filtered: {complexity_filter} complexity only)")
    print("  Sorted by complexity score (highest first)")
    print(separator)
    print(
        f"{'#':<8} {'Complexity':<12} {'Score':<7} {'Confidence':<12} "
        f"{'Staleness':<10} {'Age':<8} {'Title'}"
    )
    print("-" * 96)

    lookup: dict[int, Issue] = {}
    for issue, scores in evaluated:
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
        f"  {len(evaluated)} issue(s) shown. "
        "Signals: body length, label count, comment count, issue age. "
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


def poll_session_status(
    api_key: str,
    session_id: str,
    interval: int = DEVIN_POLL_INTERVAL,
    timeout: int = DEVIN_POLL_TIMEOUT,
) -> dict[str, Any]:
    """Poll the Devin session endpoint until the session reaches a terminal state.

    Terminal states: ``finished``, ``stopped``, ``failed``.
    Non-terminal states: ``queued``, ``started``, ``running``.

    Args:
        api_key: Devin API bearer token.
        session_id: The session ID to poll.
        interval: Seconds between polls.
        timeout: Maximum total seconds to wait before giving up.

    Returns:
        The last status response dict, which includes at minimum
        ``status`` and may include ``pull_request_url``.
    """
    url = f"{DEVIN_API_URL}/{session_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    terminal_states = {"finished", "stopped", "failed"}
    elapsed = 0
    last_status = "unknown"
    status_data: dict[str, Any] = {}

    logger.info(
        "Polling session %s (every %ds, timeout %ds)...",
        session_id,
        interval,
        timeout,
    )

    while elapsed < timeout:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            status_data = resp.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("Poll request failed: %s. Retrying...", exc)
            time.sleep(interval)
            elapsed += interval
            continue

        current_status = status_data.get(
            "status_enum", status_data.get("status", "unknown")
        )

        if current_status != last_status:
            _print_status_update(session_id, current_status, status_data)
            last_status = current_status

        if current_status in terminal_states:
            logger.info(
                "Session %s reached terminal state: %s", session_id, current_status
            )
            return status_data

        time.sleep(interval)
        elapsed += interval

    logger.warning("Polling timed out after %ds for session %s.", timeout, session_id)
    status_data["status"] = status_data.get("status", "timeout")
    return status_data


def _print_status_update(session_id: str, status: str, data: dict[str, Any]) -> None:
    """Print a formatted status transition to the terminal."""
    status_icons = {
        "queued": "\U0001f4e5",  # inbox tray
        "started": "\U0001f680",  # rocket
        "running": "\U0001f6e0",  # wrench
        "finished": "\u2705",  # check mark
        "stopped": "\u23f9\ufe0f",  # stop button
        "failed": "\u274c",  # cross mark
    }
    icon = status_icons.get(status, "\U0001f504")  # default: arrows
    ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC")
    pr_url = data.get("pull_request_url", "")
    pr_note = f"  PR: {pr_url}" if pr_url else ""
    print(f"  {icon} [{ts}] Session {session_id[:12]}... -> {status.upper()}{pr_note}")


# ---------------------------------------------------------------------------
# Phase 4 -- Communication (mock Slack notification)
# ---------------------------------------------------------------------------


def _post_slack_webhook(
    payload: dict[str, Any],
) -> bool:
    """POST a JSON payload to the configured Slack webhook URL.

    Args:
        payload: The Slack message payload (text and/or blocks).

    Returns:
        ``True`` if the webhook responded with HTTP 200, ``False`` otherwise.
    """
    if not SLACK_WEBHOOK_URL:
        logger.warning("SLACK_WEBHOOK_URL is not set. Skipping real Slack notification.")
        return False

    try:
        resp = requests.post(
            SLACK_WEBHOOK_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("Slack webhook delivered successfully.")
            return True
        logger.warning(
            "Slack webhook returned HTTP %d: %s", resp.status_code, resp.text
        )
    except requests.exceptions.RequestException as exc:
        logger.warning("Failed to deliver Slack webhook: %s", exc)
    return False


def send_slack_notification(
    issue: Issue,
    session_url: str,
    repo_name: str,
) -> None:
    """Send a Slack notification via webhook and print to the terminal.

    If ``SLACK_WEBHOOK_URL`` is configured the message is POSTed to Slack
    using Block Kit formatting.  A human-readable summary is always printed
    to the terminal regardless of webhook availability.

    Args:
        issue: The GitHub issue being resolved.
        session_url: The Devin session URL for tracking progress.
        repo_name: The repository name (owner/repo).
    """
    scores = mock_llm_evaluate(issue)
    complexity = scores["complexity"]
    complexity_score = scores["complexity_score"]
    branch_name = f"devin/fix-issue-{issue.number}"

    # --- Send real Slack webhook ---
    slack_payload: dict[str, Any] = {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"\U0001f916 *Devin Automation Triggered* \u2014 "
                        f"Issue #{issue.number}\n\n"
                        f"*Title:* {issue.title[:60]}\n"
                        f"*Repo:* `{repo_name}`\n"
                        f"*Complexity:* {complexity} ({complexity_score}/100)\n"
                        f"*Branch:* `{branch_name}`\n"
                        f"*Session:* <{session_url}|View Devin session>"
                    ),
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            "\U0001f9f5 All progress updates will be posted as "
                            "thread replies to avoid channel noise."
                        ),
                    }
                ],
            },
        ],
    }
    _post_slack_webhook(slack_payload)

    # --- Terminal output (always printed) ---
    border = "*" * 72
    thin = "-" * 72

    print(f"\n{border}")
    print("  \U0001f514  SLACK  \u2014  #eng-devin-automation")
    print(border)
    print(f"  \U0001f916 *Devin Automation Triggered*  \u2014  Issue #{issue.number}")
    print(f"  *Title:*       {issue.title[:60]}")
    print(f"  *Repo:*        {repo_name}")
    print(f"  *Complexity:*  {complexity} ({complexity_score}/100)")
    print(f"  *Branch:*      `{branch_name}`")
    print(f"  *Session:*     {session_url}")
    print()
    print(
        "  \U0001f9f5 All progress updates will be posted as thread replies "
        "to this message"
    )
    print("     to avoid channel noise. Follow the thread for real-time status.")
    print(border)

    # --- Simulated threaded reply: PR Ready ---
    print(f"  {thin}")
    print("  \U0001f4ac  Thread reply (simulated)")
    print(f"  {thin}")
    print(
        f"  \u2705 *PR Ready* \u2014 Devin has opened a pull request for "
        f"Issue #{issue.number}."
    )
    print(f"  *Branch:* `{branch_name}` \u2192 `main`")
    print(f"  *Review:* https://github.com/{repo_name}/compare/{branch_name}")
    print(f"  {thin}\n")


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
    send_slack_notification(issue, session_url, repo_name)

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

    # Print executive summary dashboard
    print_summary_dashboard(results, selected_issues)

    # 5. Session status polling (optional)
    successes = [r for r in results if "error" not in r]
    if successes:
        _offer_session_polling(successes, DEVIN_API_KEY)

    logger.info("Pipeline complete. %d session(s) created.", len(successes))


# ---------------------------------------------------------------------------
# Summary Dashboard
# ---------------------------------------------------------------------------

# Estimated engineer-hours saved per complexity level (for ROI projection)
_TIME_SAVED_HOURS = {"Low": 1.5, "Medium": 4.0, "High": 8.0}


def print_summary_dashboard(
    results: list[dict[str, Any]],
    dispatched_issues: list[Issue],
) -> None:
    """Print an executive-friendly summary dashboard.

    Includes per-issue dispatch status, complexity breakdown, and an
    estimated engineer-hours saved projection.

    Args:
        results: Dispatch result dicts (one per issue, may contain ``error``).
        dispatched_issues: The Issue objects that were dispatched.
    """
    successes = [r for r in results if "error" not in r]
    failures = [r for r in results if "error" in r]

    # Build a quick lookup: issue_number -> Issue
    issue_map: dict[int, Issue] = {iss.number: iss for iss in dispatched_issues}

    # Compute complexity breakdown & time-saved estimate
    complexity_counts: dict[str, int] = {"Low": 0, "Medium": 0, "High": 0}
    total_hours_saved = 0.0
    for result in successes:
        iss = issue_map.get(result["issue_number"])
        if iss:
            scores = mock_llm_evaluate(iss)
            comp = scores["complexity"]
            complexity_counts[comp] = complexity_counts.get(comp, 0) + 1
            total_hours_saved += _TIME_SAVED_HOURS.get(comp, 2.0)

    sep = "=" * 80
    print(f"\n{sep}")
    print("  FINSERV CO  --  DISPATCH SUMMARY DASHBOARD")
    print(sep)

    # --- Overview ---
    print(f"\n  Total issues dispatched : {len(results)}")
    print(f"  Sessions created        : {len(successes)}")
    print(f"  Failed to dispatch      : {len(failures)}")

    # --- Per-issue detail table ---
    print(f"\n  {'#':<8} {'Status':<12} {'Session / Error':<50} {'Title'}")
    print("  " + "-" * 76)
    for result in results:
        inum = result["issue_number"]
        title = issue_map[inum].title[:35] if inum in issue_map else "?"
        if "error" in result:
            status = "FAILED"
            detail = result["error"][:48]
        else:
            status = "OK"
            detail = result.get("session_id", "?")[:48]
        print(f"  #{inum:<7} {status:<12} {detail:<50} {title}")

    # --- Complexity breakdown ---
    print("\n  Complexity Breakdown:")
    for level in ("Low", "Medium", "High"):
        count = complexity_counts[level]
        bar = "\u2588" * count + "\u2591" * (max(0, 10 - count))
        print(f"    {level:<8} {bar}  {count}")

    # --- ROI estimate ---
    print(f"\n  Estimated engineer-hours saved : {total_hours_saved:.1f}h")
    print(
        f"  (Based on {_TIME_SAVED_HOURS['Low']}h/Low, "
        f"{_TIME_SAVED_HOURS['Medium']}h/Med, "
        f"{_TIME_SAVED_HOURS['High']}h/High per issue)"
    )

    # --- Failures ---
    if failures:
        print("\n  Failed dispatches:")
        for f in failures:
            print(f"    - Issue #{f['issue_number']}: {f['error']}")

    print(f"\n{sep}\n")


def _offer_session_polling(successes: list[dict[str, Any]], api_key: str) -> None:
    """Ask the operator whether to poll Devin session statuses."""
    choice = (
        input("Would you like to monitor session progress in real-time? (y/N): ")
        .strip()
        .lower()
    )
    if choice not in ("y", "yes"):
        logger.info("Skipping session polling.")
        return

    print("\n" + "-" * 80)
    print("  REAL-TIME SESSION MONITORING")
    print(
        f"  Polling every {DEVIN_POLL_INTERVAL}s "
        f"(timeout {DEVIN_POLL_TIMEOUT}s per session)"
    )
    print("-" * 80)

    for result in successes:
        sid = result["session_id"]
        issue_num = result["issue_number"]
        print(f"\n  --- Issue #{issue_num} (session {sid[:12]}...) ---")
        final = poll_session_status(api_key, sid)
        final_status = final.get("status_enum", final.get("status", "unknown"))
        pr_url = final.get("pull_request_url", "N/A")
        print(f"  Final status: {final_status.upper()}  |  PR: {pr_url}")

    print("\n" + "-" * 80)
    print("  Monitoring complete.")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
