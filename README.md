# Devin Issue Triage & Resolution CLI

A command-line tool that orchestrates the [Devin API](https://docs.devin.ai) to
automatically triage and resolve stale GitHub issues for **FinServ Co**.

## How It Works

| Phase | Description |
|-------|-------------|
| **1. Automated Triage** | Fetches open issues with a configurable label from GitHub and scores each one with a mock LLM evaluation (Complexity + Confidence). |
| **2. Human-in-the-Loop** | Pauses and prompts the operator to select which issue Devin should resolve. |
| **3. Execution** | Sends a detailed prompt to the Devin v3 API, instructing it to clone the repo, fix the issue, run tests, and open a PR. |
| **4. Communication** | Fires a mock Slack webhook notification with the Devin session URL so the team can track progress. |

## Prerequisites

* Python 3.10+
* A GitHub personal-access token with read access to the target repo
* A Devin API key

## Quick Start

```bash
# 1. Clone or download this project
git clone <repo-url> && cd devin-issue-triage

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env and fill in your real tokens

# 4. Run the pipeline
python main.py
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | Yes | GitHub PAT with repo read permissions |
| `DEVIN_API_KEY` | Yes | Devin API bearer token |
| `GITHUB_REPO_NAME` | Yes | Target repo in `owner/repo` format |
| `GITHUB_ISSUE_LABEL` | No | Label to filter issues (default: `devin-backlog`) |

## Project Structure

```
devin-issue-triage/
  main.py            # Entry-point with all pipeline phases
  requirements.txt   # Python dependencies
  .env.example       # Template for environment variables
  README.md          # This file
```

## License

MIT
