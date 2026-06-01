import typer

from codearch.ingest_github import ingest_repo


app = typer.Typer(help="Codebase Archaeologist CLI.")


@app.command()
def index(repo_url: str):
    """Fetch raw GitHub metadata for a repository."""
    output_dir = ingest_repo(repo_url)
    typer.echo(f"Indexed {repo_url}")
    typer.echo(f"Saved raw data to {output_dir}")


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
