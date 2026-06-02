import typer

from codearch.ingest_github import ingest_repo
from codearch.ingest_github import parse_github_url
from codearch.index import build_index
from codearch.normalize import normalize_repo
from codearch.retrieve import IndexNotFoundError
from codearch.retrieve import retrieve_relevant_artifacts


app = typer.Typer(help="Codebase Archaeologist CLI.")
WEAK_RESULT_DISTANCE_THRESHOLD = 1.3


@app.command()
def index(repo_url: str):
    """Fetch, normalize, and index GitHub repository artifacts."""
    output_dir = ingest_repo(repo_url)
    owner, repo = parse_github_url(repo_url)
    artifacts = normalize_repo(owner, repo)
    indexed_count = build_index(owner, repo)
    typer.echo(f"Indexed {indexed_count} artifacts into Chroma")
    typer.echo(f"Saved raw data to {output_dir}")
    typer.echo(f"Normalized artifacts: {len(artifacts)}")


@app.command()
def ask(question: str, repo: str = typer.Option(..., "--repo")):
    """Retrieve relevant artifacts for a codebase question."""
    if "/" not in repo:
        raise typer.BadParameter("--repo must use owner/repo format")

    owner, repo_name = repo.split("/", maxsplit=1)
    if not owner or not repo_name:
        raise typer.BadParameter("--repo must use owner/repo format")

    try:
        artifacts = retrieve_relevant_artifacts(owner, repo_name, question)
    except IndexNotFoundError:
        typer.echo(f"No index found for {owner}/{repo_name}.")
        typer.echo(f"Run: codearch index https://github.com/{owner}/{repo_name}")
        raise typer.Exit(code=1)

    if artifacts and artifacts[0]["distance"] > WEAK_RESULT_DISTANCE_THRESHOLD:
        typer.echo(
            "Warning: retrieved artifacts may be weak matches for this question."
        )

    typer.echo("Top retrieved artifacts:")
    for artifact in artifacts:
        metadata = artifact.get("metadata") or {}
        typer.echo(f"- source: {artifact['source']}")
        typer.echo(f"  type: {metadata.get('type', '')}")
        typer.echo(f"  distance: {artifact['distance']}")
        typer.echo(f"  text: {artifact['text'][:500]}")


@app.command()
def context(change_request: str):
    """Placeholder for future change context generation."""
    typer.echo(f"Context placeholder for: {change_request}")


if __name__ == "__main__":
    app()
