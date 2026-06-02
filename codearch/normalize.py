import hashlib
import json
import re
from datetime import UTC
from datetime import datetime
from pathlib import Path


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
CHROMA_DATA_DIR = Path("data/chroma")


def normalize_repo(
    owner: str,
    repo: str,
    mode: str = "standard",
    fetched_counts: dict | None = None,
    raw_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
    chroma_data_dir: Path | None = None,
    manifest_raw_data_path: Path | None = None,
    manifest_processed_artifacts_path: Path | None = None,
    manifest_chroma_path: Path | None = None,
) -> list[dict]:
    repo_name = f"{owner}/{repo}"
    raw_data_dir = raw_data_dir or RAW_DATA_DIR
    processed_data_dir = processed_data_dir or PROCESSED_DATA_DIR
    chroma_data_dir = chroma_data_dir or CHROMA_DATA_DIR
    raw_dir = raw_data_dir / f"{owner}_{repo}"
    processed_dir = processed_data_dir / f"{owner}_{repo}"

    issues = _read_json(raw_dir / "issues.json")
    pull_requests = _read_json(raw_dir / "pulls.json")
    commits = _read_json(raw_dir / "commits.json")
    docs = _read_json_if_exists(raw_dir / "docs.json")

    print("Normalizing artifacts...")
    print(f"- Raw issues read: {len(issues)}")
    print(f"- Raw pull requests read: {len(pull_requests)}")
    print(f"- Raw commits read: {len(commits)}")
    print(f"- Raw docs read: {len(docs)}")

    artifacts = []
    artifacts.extend(_normalize_issue(issue, repo_name) for issue in issues)
    artifacts.extend(_normalize_pull_request(pr, repo_name) for pr in pull_requests)
    artifacts.extend(_normalize_commit(commit, repo_name) for commit in commits)
    artifacts.extend(_normalize_doc(doc, repo_name) for doc in docs)
    indexed_artifact_counts = _indexed_artifact_counts(artifacts)

    print("Normalized artifact counts:")
    print(f"- Issues: {indexed_artifact_counts['issues']}")
    print(f"- Pull Requests: {indexed_artifact_counts['pull_requests']}")
    print(f"- Commits: {indexed_artifact_counts['commits']}")
    print(f"- Documentation: {indexed_artifact_counts['documentation']}")

    _write_jsonl(processed_dir / "artifacts.jsonl", artifacts)
    _write_manifest(
        processed_dir / "manifest.json",
        repo_name=repo_name,
        mode=mode,
        fetched_counts=fetched_counts
        or _fetched_counts(
            issues=issues,
            pull_requests=pull_requests,
            commits=commits,
            docs=docs,
        ),
        indexed_artifact_counts=indexed_artifact_counts,
        issues=issues,
        pull_requests=pull_requests,
        commits=commits,
        raw_data_path=manifest_raw_data_path or raw_dir,
        processed_artifacts_path=manifest_processed_artifacts_path
        or processed_dir / "artifacts.jsonl",
        chroma_path=manifest_chroma_path or chroma_data_dir / f"{owner}_{repo}",
    )

    return artifacts


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path):
    if not path.exists():
        return []

    return _read_json(path)


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


def _normalize_doc(doc: dict, repo_name: str) -> dict:
    path = doc["path"]
    url = doc.get("url")
    text = doc.get("text") or doc.get("content") or ""

    return {
        "id": f"doc_{_doc_id_slug(path)}",
        "source": f"Doc: {path}",
        "text": f"Path: {path}\n\n{text}",
        "metadata": {
            "type": "documentation",
            "repo": repo_name,
            "path": path,
            "url": url,
        },
    }


def _doc_id_slug(path: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", path.lower()).strip("_")
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:10]
    return f"{slug}_{digest}" if slug else digest


def _fetched_counts(
    issues: list[dict],
    pull_requests: list[dict],
    commits: list[dict],
    docs: list[dict],
) -> dict:
    return {
        "issues": len(issues),
        "pull_requests": len(pull_requests),
        "commits": len(commits),
        "docs": len(docs),
    }


def _indexed_artifact_counts(artifacts: list[dict]) -> dict:
    counts = {
        "issues": 0,
        "pull_requests": 0,
        "commits": 0,
        "documentation": 0,
    }
    type_to_count_key = {
        "issue": "issues",
        "pull_request": "pull_requests",
        "commit": "commits",
        "documentation": "documentation",
    }

    for artifact in artifacts:
        artifact_type = artifact.get("metadata", {}).get("type")
        count_key = type_to_count_key.get(artifact_type)
        if count_key:
            counts[count_key] += 1

    return counts


def _write_manifest(
    path: Path,
    repo_name: str,
    mode: str,
    fetched_counts: dict,
    indexed_artifact_counts: dict,
    issues: list[dict],
    pull_requests: list[dict],
    commits: list[dict],
    raw_data_path: Path,
    processed_artifacts_path: Path,
    chroma_path: Path,
):
    manifest = {
        "repo": repo_name,
        "mode": mode,
        "indexed_at": datetime.now(UTC).isoformat(),
        "fetched_counts": fetched_counts,
        "indexed_artifact_counts": indexed_artifact_counts,
        "raw_data_path": str(raw_data_path),
        "processed_artifacts_path": str(processed_artifacts_path),
        "chroma_path": str(chroma_path),
        "latest_issue_updated_at": _latest_updated_at(issues),
        "latest_pr_updated_at": _latest_updated_at(pull_requests),
        "latest_commit_sha": commits[0].get("sha") if commits else None,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _latest_updated_at(records: list[dict]):
    updated_values = [
        record.get("updated_at") for record in records if record.get("updated_at")
    ]
    return max(updated_values) if updated_values else None


def _write_jsonl(path: Path, artifacts: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(artifact) for artifact in artifacts]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
