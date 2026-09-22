from pathlib import Path
from typing import Annotated

import typer

from paper_rag.bootstrap import build_container
from paper_rag.infrastructure.parsers import parse_file

app = typer.Typer(help="Ingest and query scientific papers.", no_args_is_help=True)


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    title: Annotated[str | None, typer.Option(help="Document title.")] = None,
) -> None:
    """Ingest one PDF, Markdown, or text document."""
    text = parse_file(path)
    count = build_container().ingest_document.execute(
        title=title or path.stem,
        text=text,
        source=str(path),
    )
    typer.echo(f"Ingested {count} chunks from {path.name}.")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question about the indexed papers.")],
    top_k: Annotated[int | None, typer.Option(min=1, max=20)] = None,
) -> None:
    """Ask a question and print the retrieved evidence."""
    answer = build_container().ask_question.execute(question, top_k)
    typer.echo(answer.text)
    if answer.citations:
        typer.echo("\nSources:")
        for index, citation in enumerate(answer.citations, start=1):
            typer.echo(f"[{index}] {citation.title} — {citation.source} ({citation.score:.3f})")


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("paper_rag.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
