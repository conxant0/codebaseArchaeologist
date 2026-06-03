import os
import shutil
import uuid
import warnings
from enum import Enum
from pathlib import Path

import typer

from codearch.eval import evaluate_question
from codearch.eval import load_evaluation_file
from codearch.eval import print_evaluation_summary
from codearch.eval import write_evaluation_csv
from codearch.generate import GroqConfigurationError
from codearch.generate import answer_question
from codearch.generate import compute_evidence_strength
from codearch.generate import generate_context_pack
from codearch.index import CHROMA_DATA_DIR
from codearch.index import build_index
from codearch.ingest_github import github_token_missing
from codearch.ingest_github import ingest_repo
from codearch.ingest_github import INDEXING_MODES
from codearch.ingest_github import parse_github_url
from codearch.ingest_github import RAW_DATA_DIR
from codearch.normalize import PROCESSED_DATA_DIR
from codearch.normalize import normalize_repo
from codearch.retrieve import IndexNotFoundError
from codearch.retrieve import retrieve_relevant_artifacts


def _configure_terminal_output():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings(
        "ignore",
        message="`clean_up_tokenization_spaces` was not set.*",
        category=FutureWarning,
        append=False,
    )


_configure_terminal_output()

app = typer.Typer(help="Codebase Archaeologist CLI.")
# Heuristic Chroma distance threshold; tune as retrieval quality is evaluated.
WEAK_EVIDENCE_DISTANCE_THRESHOLD = 1.3
WEAK_EVIDENCE_WARNING = (
    "Warning: Retrieved artifacts are related, but may not contain a direct "
    "historical explanation."
)
STAGING_DATA_DIR = RAW_DATA_DIR.parent / ".staging"


class IndexingMode(str, Enum):
    recent = "recent"
    standard = "standard"
    deep = "deep"


@app.command()
def index(
    repo_url: str,
    mode: IndexingMode = typer.Option(IndexingMode.standard, "--mode"),
):
    """Fetch, normalize, and index GitHub repository artifacts."""
    mode_value = mode.value
    owner, repo = parse_github_url(repo_url)
    _print_indexing_plan(owner, repo, mode_value)
    _recover_interrupted_index_swap(owner, repo)

    staging_root = _create_staging_root(owner, repo)
    staged_raw_data_dir = staging_root / "raw"
    staged_processed_data_dir = staging_root / "processed"
    staged_chroma_data_dir = staging_root / "chroma"
    repo_dir_name = f"{owner}_{repo}"

    try:
        ingest_result = ingest_repo(
            repo_url,
            mode=mode_value,
            raw_data_dir=staged_raw_data_dir,
        )
        artifacts = normalize_repo(
            owner,
            repo,
            mode=mode_value,
            fetched_counts=ingest_result.fetched_counts,
            raw_data_dir=staged_raw_data_dir,
            processed_data_dir=staged_processed_data_dir,
            chroma_data_dir=staged_chroma_data_dir,
            manifest_raw_data_path=RAW_DATA_DIR / repo_dir_name,
            manifest_processed_artifacts_path=PROCESSED_DATA_DIR
            / repo_dir_name
            / "artifacts.jsonl",
            manifest_chroma_path=CHROMA_DATA_DIR / repo_dir_name,
        )
        build_index(
            owner,
            repo,
            processed_data_dir=staged_processed_data_dir,
            chroma_data_dir=staged_chroma_data_dir,
        )
        _commit_staged_index(owner, repo, staging_root)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)

    indexed_artifact_counts = _indexed_artifact_counts(artifacts)

    typer.echo(f"Indexed {owner}/{repo} using mode: {mode_value}")
    typer.echo("")
    typer.echo("Fetched:")
    typer.echo(f"- Issues: {ingest_result.fetched_counts['issues']}")
    typer.echo(f"- Pull Requests: {ingest_result.fetched_counts['pull_requests']}")
    typer.echo(f"- Commits: {ingest_result.fetched_counts['commits']}")
    typer.echo(f"- Docs: {ingest_result.fetched_counts['docs']}")
    typer.echo("")
    typer.echo("Indexed artifacts:")
    typer.echo(f"- Issues: {indexed_artifact_counts['issues']}")
    typer.echo(f"- Pull Requests: {indexed_artifact_counts['pull_requests']}")
    typer.echo(f"- Commits: {indexed_artifact_counts['commits']}")
    typer.echo(f"- Documentation: {indexed_artifact_counts['documentation']}")
    typer.echo("")
    typer.echo("Storage:")
    storage_paths = _storage_paths(owner, repo)
    typer.echo(f"- Raw data: {storage_paths['raw_data']}/")
    typer.echo(f"- Processed artifacts: {storage_paths['processed_artifacts']}")
    typer.echo(f"- Manifest: {storage_paths['manifest']}")
    typer.echo(f"- Chroma index: {storage_paths['chroma']}/")

    if ingest_result.warnings:
        typer.echo("")
        typer.echo("Warnings:")
        for warning in ingest_result.warnings:
            typer.echo(f"- {warning}")


@app.command()
def ask(question: str, repo: str = typer.Option(..., "--repo")):
    """Retrieve relevant artifacts for a codebase question."""
    _configure_terminal_output()
    owner, repo_name = _parse_repo_option(repo)
    artifacts = _retrieve_artifacts_or_exit(owner, repo_name, question)

    if _retrieved_evidence_is_weak(artifacts):
        typer.echo(WEAK_EVIDENCE_WARNING)

    evidence_strength = compute_evidence_strength(artifacts)

    try:
        answer = answer_question(
            question,
            artifacts,
            evidence_strength=evidence_strength,
        )
    except GroqConfigurationError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    typer.echo("Question:")
    typer.echo(question)
    typer.echo("")
    typer.echo(f"Evidence Strength: {evidence_strength}")
    typer.echo("")
    typer.echo("Historical Findings")
    typer.echo("")
    typer.echo(answer)
    typer.echo("")
    typer.echo("Sources:")
    for source in _sources_used(answer, artifacts):
        typer.echo(f"- {source}")


@app.command()
def context(
    change_request: str,
    repo: str = typer.Option(..., "--repo"),
    out: str = typer.Option(".codearch/context.md", "--out"),
):
    """Prepare historical context for a planned code change."""
    _configure_terminal_output()
    owner, repo_name = _parse_repo_option(repo)
    artifacts = _retrieve_artifacts_or_exit(owner, repo_name, change_request)

    if _retrieved_evidence_is_weak(artifacts):
        typer.echo(WEAK_EVIDENCE_WARNING)

    evidence_strength = compute_evidence_strength(artifacts)

    try:
        context_pack = generate_context_pack(
            change_request,
            artifacts,
            evidence_strength=evidence_strength,
        )
    except GroqConfigurationError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(context_pack, encoding="utf-8")

    typer.echo(f"Created context pack: {out}")
    typer.echo("")
    typer.echo("Suggested next prompt for your AI coding tool:")
    typer.echo(f'"Read {out} first. Then help me with: {change_request}"')


@app.command("eval")
def eval_command(
    evaluation_file: str,
    csv: str | None = typer.Option(None, "--csv"),
):
    """Evaluate whether retrieval finds expected evidence sources."""
    _configure_terminal_output()
    try:
        cases = load_evaluation_file(evaluation_file)
    except (OSError, ValueError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    results = []
    for case in cases:
        owner, repo_name = _parse_evaluation_repo(case["repo"])
        artifacts = _retrieve_artifacts_or_exit(owner, repo_name, case["question"])
        results.append(evaluate_question(case, artifacts))

    print_evaluation_summary(results, output=typer.echo)

    if csv:
        write_evaluation_csv(results, csv)


def _indexed_artifact_counts(artifacts: list[dict]) -> dict[str, int]:
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


def _parse_repo_option(repo: str) -> tuple[str, str]:
    if "/" not in repo:
        raise typer.BadParameter("--repo must use owner/repo format")

    owner, repo_name = repo.split("/", maxsplit=1)
    if not owner or not repo_name:
        raise typer.BadParameter("--repo must use owner/repo format")

    return owner, repo_name


def _parse_evaluation_repo(repo: str) -> tuple[str, str]:
    if "/" in repo:
        return _parse_repo_option(repo)

    if "_" in repo:
        owner, repo_name = repo.split("_", maxsplit=1)
        if owner and repo_name:
            return owner, repo_name

    raise typer.BadParameter(
        "evaluation repo must use owner/repo or owner_repo format"
    )


def _retrieve_artifacts_or_exit(owner: str, repo: str, query: str) -> list[dict]:
    try:
        return retrieve_relevant_artifacts(owner, repo, query)
    except IndexNotFoundError:
        typer.echo(f"No index found for {owner}/{repo}.")
        typer.echo(f"Run: codearch index https://github.com/{owner}/{repo}")
        raise typer.Exit(code=1)


def _retrieved_evidence_is_weak(artifacts: list[dict]) -> bool:
    if not artifacts:
        return False

    artifact_types = {
        (artifact.get("metadata") or {}).get("type") for artifact in artifacts
    }
    only_commit_evidence = artifact_types == {"commit"}
    top_distance_is_weak = artifacts[0]["distance"] > WEAK_EVIDENCE_DISTANCE_THRESHOLD

    return only_commit_evidence or top_distance_is_weak


def _sources_used(answer: str, artifacts: list[dict]) -> list[str]:
    sources = _unique_sources(artifacts)
    cited_sources = [source for source in sources if source in answer]
    return cited_sources or sources


def _unique_sources(artifacts: list[dict]) -> list[str]:
    sources = []
    for artifact in artifacts:
        source = artifact.get("source")
        if source and source not in sources:
            sources.append(source)

    return sources


def _print_indexing_plan(owner: str, repo: str, mode: str):
    mode_config = INDEXING_MODES[mode]
    typer.echo(f"Indexing {owner}/{repo}")
    typer.echo(f"Mode: {mode}")
    typer.echo("This will fetch:")
    typer.echo(f"- Issues: {mode_config['issues']}")
    typer.echo(f"- Pull Requests: {mode_config['pulls']}")
    typer.echo(f"- Documentation: {'yes' if mode_config['docs'] else 'no'}")
    typer.echo(f"- Commits: {mode_config['commits']}")

    if mode in {"standard", "deep"}:
        typer.echo(
            "Building a local historical knowledge base. "
            "This may take several minutes."
        )

    if github_token_missing():
        typer.echo("GitHub token not found. GitHub API rate limits may be lower.")


def _storage_paths(owner: str, repo: str) -> dict[str, str]:
    repo_dir_name = f"{owner}_{repo}"
    processed_dir = PROCESSED_DATA_DIR / repo_dir_name
    return {
        "raw_data": str(RAW_DATA_DIR / repo_dir_name),
        "processed_artifacts": str(processed_dir / "artifacts.jsonl"),
        "manifest": str(processed_dir / "manifest.json"),
        "chroma": str(CHROMA_DATA_DIR / repo_dir_name),
    }


def _create_staging_root(owner: str, repo: str) -> Path:
    repo_dir_name = f"{owner}_{repo}"
    staging_root = STAGING_DATA_DIR / f"{repo_dir_name}.{uuid.uuid4().hex}"
    staging_root.mkdir(parents=True)
    return staging_root


def _commit_staged_index(owner: str, repo: str, staging_root: Path):
    repo_dir_name = f"{owner}_{repo}"
    targets = _index_directory_targets(owner, repo, staging_root)
    backup_root = STAGING_DATA_DIR / f"{repo_dir_name}.swap_backup"

    if backup_root.exists():
        _recover_interrupted_index_swap(owner, repo)

    backup_root.mkdir(parents=True, exist_ok=False)
    moved_targets = []

    try:
        for label, staged_dir, live_dir in targets:
            if not staged_dir.exists():
                raise RuntimeError(f"Missing staged {label} directory: {staged_dir}")

            backup_dir = backup_root / label
            live_dir.parent.mkdir(parents=True, exist_ok=True)

            if live_dir.exists():
                live_dir.rename(backup_dir)
                moved_targets.append((label, backup_dir, live_dir))

            staged_dir.rename(live_dir)

        shutil.rmtree(backup_root, ignore_errors=True)
    except Exception:
        _rollback_index_swap(moved_targets)
        if backup_root.exists():
            shutil.rmtree(backup_root, ignore_errors=True)
        raise


def _recover_interrupted_index_swap(owner: str, repo: str):
    repo_dir_name = f"{owner}_{repo}"
    backup_root = STAGING_DATA_DIR / f"{repo_dir_name}.swap_backup"
    if not backup_root.exists():
        return

    for label, _staged_dir, live_dir in _index_directory_targets(owner, repo):
        backup_dir = backup_root / label
        if not backup_dir.exists():
            continue

        if live_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        else:
            live_dir.parent.mkdir(parents=True, exist_ok=True)
            backup_dir.rename(live_dir)

    if backup_root.exists():
        shutil.rmtree(backup_root, ignore_errors=True)


def _rollback_index_swap(moved_targets: list[tuple[str, Path, Path]]):
    for _label, backup_dir, live_dir in reversed(moved_targets):
        if live_dir.exists():
            shutil.rmtree(live_dir, ignore_errors=True)
        if backup_dir.exists():
            backup_dir.rename(live_dir)


def _index_directory_targets(
    owner: str,
    repo: str,
    staging_root: Path | None = None,
) -> list[tuple[str, Path, Path]]:
    repo_dir_name = f"{owner}_{repo}"
    staging_root = staging_root or STAGING_DATA_DIR / f"{repo_dir_name}.unused"

    return [
        ("raw", staging_root / "raw" / repo_dir_name, RAW_DATA_DIR / repo_dir_name),
        (
            "processed",
            staging_root / "processed" / repo_dir_name,
            PROCESSED_DATA_DIR / repo_dir_name,
        ),
        (
            "chroma",
            staging_root / "chroma" / repo_dir_name,
            CHROMA_DATA_DIR / repo_dir_name,
        ),
    ]


if __name__ == "__main__":
    app()
