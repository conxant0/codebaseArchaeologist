import typer

from codearch.ingest_github import ingest_repo
from codearch.ingest_github import parse_github_url
from codearch.index import build_index
from codearch.normalize import normalize_repo


app = typer.Typer(help="Codebase Archaeologist CLI.")


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
def ask(question: str):
    """Placeholder for future codebase question answering."""
    typer.echo(f"Ask placeholder for: {question}")


@app.command()
def context(change_request: str):
    """Placeholder for future change context generation."""
    typer.echo(f"Context placeholder for: {change_request}")


if __name__ == "__main__":
    app()
