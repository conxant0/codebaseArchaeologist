import json
from pathlib import Path


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def normalize_repo(owner: str, repo: str) -> list[dict]:
    repo_name = f"{owner}/{repo}"
    raw_dir = RAW_DATA_DIR / f"{owner}_{repo}"
    processed_dir = PROCESSED_DATA_DIR / f"{owner}_{repo}"

    issues = _read_json(raw_dir / "issues.json")
    pull_requests = _read_json(raw_dir / "pulls.json")
    commits = _read_json(raw_dir / "commits.json")

    artifacts = []
    artifacts.extend(_normalize_issue(issue, repo_name) for issue in issues)
    artifacts.extend(_normalize_pull_request(pr, repo_name) for pr in pull_requests)
    artifacts.extend(_normalize_commit(commit, repo_name) for commit in commits)

    _write_jsonl(processed_dir / "artifacts.jsonl", artifacts)

    return artifacts


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_issue(issue: dict, repo_name: str) -> dict:
    number = issue["number"]
    title = issue.get("title") or ""
    body = issue.get("body") or ""
    labels = ", ".join(label["name"] for label in issue.get("labels", []))

    text_parts = [
        f"Title: {title}",
        f"Body: {body}",
    ]
    if labels:
        text_parts.append(f"Labels: {labels}")

    return {
        "id": f"issue_{number}",
        "source": f"Issue #{number}",
        "text": "\n\n".join(text_parts),
        "metadata": {
            "type": "issue",
            "repo": repo_name,
            "number": number,
            "url": issue.get("html_url"),
            "title": title,
        },
    }


def _normalize_pull_request(pull_request: dict, repo_name: str) -> dict:
    number = pull_request["number"]
    title = pull_request.get("title") or ""
    body = pull_request.get("body") or ""

    text_parts = [
        f"Title: {title}",
        f"Body: {body}",
        f"State: {pull_request.get('state') or ''}",
        f"Created At: {pull_request.get('created_at') or ''}",
    ]

    if pull_request.get("merged_at"):
        text_parts.append(f"Merged At: {pull_request['merged_at']}")

    return {
        "id": f"pr_{number}",
        "source": f"Pull Request #{number}",
        "text": "\n\n".join(text_parts),
        "metadata": {
            "type": "pull_request",
            "repo": repo_name,
            "number": number,
            "url": pull_request.get("html_url"),
            "title": title,
        },
    }


def _normalize_commit(commit: dict, repo_name: str) -> dict:
    sha = commit["sha"]
    short_sha = sha[:7]
    commit_data = commit.get("commit", {})
    author = commit_data.get("author", {})
    message = commit_data.get("message") or ""
    author_name = author.get("name") or ""
    date = author.get("date") or ""
    url = commit.get("html_url")

    return {
        "id": f"commit_{short_sha}",
        "source": f"Commit {short_sha}",
        "text": "\n\n".join(
            [
                f"Message: {message}",
                f"Author: {author_name}",
                f"Date: {date}",
                f"SHA: {sha}",
                f"URL: {url}",
            ]
        ),
        "metadata": {
            "type": "commit",
            "repo": repo_name,
            "sha": sha,
            "url": url,
            "title": message.splitlines()[0] if message else "",
        },
    }


def _write_jsonl(path: Path, artifacts: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(artifact) for artifact in artifacts]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
