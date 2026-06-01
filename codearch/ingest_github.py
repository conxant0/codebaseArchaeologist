import json
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


GITHUB_API_BASE_URL = "https://api.github.com"
RAW_DATA_DIR = Path("data/raw")


def parse_github_url(repo_url) -> tuple[str, str]:
    parsed_url = urlparse(repo_url)
    path_parts = [part for part in parsed_url.path.split("/") if part]

    if parsed_url.scheme not in {"http", "https"}:
        raise ValueError("GitHub URL must start with http:// or https://")

    if parsed_url.netloc.lower() != "github.com" or len(path_parts) < 2:
        raise ValueError("Expected a GitHub URL like https://github.com/owner/repo")

    return path_parts[0], path_parts[1].removesuffix(".git")


def _github_get(path: str, params=None):
    load_dotenv()

    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")

    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{GITHUB_API_BASE_URL}{path}"
    response = requests.get(url, headers=headers, params=params, timeout=30)

    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        status_code = response.status_code
        message = response.text.strip() or response.reason
        raise RuntimeError(f"GitHub API request failed ({status_code}): {message}") from error

    return response.json()


def fetch_issues(owner: str, repo: str, limit=50):
    return _github_get(
        f"/repos/{owner}/{repo}/issues",
        params={"per_page": limit, "state": "all"},
    )


def fetch_pull_requests(owner: str, repo: str, limit=50):
    return _github_get(
        f"/repos/{owner}/{repo}/pulls",
        params={"per_page": limit, "state": "all"},
    )


def fetch_commits(owner: str, repo: str, limit=100):
    return _github_get(
        f"/repos/{owner}/{repo}/commits",
        params={"per_page": limit},
    )


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def ingest_repo(repo_url: str) -> Path:
    owner, repo = parse_github_url(repo_url)
    output_dir = RAW_DATA_DIR / f"{owner}_{repo}"

    issues = fetch_issues(owner, repo)
    pull_requests = fetch_pull_requests(owner, repo)
    commits = fetch_commits(owner, repo)

    _save_json(output_dir / "issues.json", issues)
    _save_json(output_dir / "pulls.json", pull_requests)
    _save_json(output_dir / "commits.json", commits)

    return output_dir
