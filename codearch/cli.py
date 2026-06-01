import typer


app = typer.Typer(help="Codebase Archaeologist CLI.")


@app.command()
def index(repo_url: str):
    """Placeholder for future repository indexing."""
    typer.echo(f"Indexing placeholder for {repo_url}")


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
