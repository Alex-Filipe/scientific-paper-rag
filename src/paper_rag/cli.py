from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import typer

from paper_rag.bootstrap import build_container
from paper_rag.evaluation.dataset import load_dataset
from paper_rag.evaluation.runner import RetrievalEvaluator
from paper_rag.infrastructure.parsers import parse_file
from paper_rag.settings import Settings

app = typer.Typer(help="Ingest and query scientific papers.", no_args_is_help=True)


class EvaluationEmbedding(StrEnum):
    HASHING = "hashing"
    SEMANTIC = "semantic"
    BOTH = "both"


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
def evaluate(
    dataset: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ] = Path("evaluation/retrieval_baseline.json"),
    embedding: Annotated[
        EvaluationEmbedding,
        typer.Option(help="Embedding backend to evaluate; use 'both' to compare."),
    ] = EvaluationEmbedding.HASHING,
    top_k: Annotated[int, typer.Option(min=1, max=20)] = 3,
    min_recall: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.0,
    min_mrr: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.0,
) -> None:
    """Evaluate retrieval or compare hashing with semantic embeddings."""
    evaluation_dataset = load_dataset(dataset)
    configured = Settings.from_environment()
    backends = (
        ("hashing", "semantic") if embedding is EvaluationEmbedding.BOTH else (embedding.value,)
    )
    reports = {}
    with TemporaryDirectory(prefix="paper-rag-evaluation-") as directory:
        for backend in backends:
            run_settings = replace(
                configured,
                database_path=Path(directory) / f"{backend}.db",
                embedding_backend=backend,
            )
            container = build_container(run_settings)
            reports[backend] = RetrievalEvaluator(
                ingest_document=container.ingest_document,
                retrieve_chunks=container.retrieve_chunks,
            ).run(evaluation_dataset, top_k)

    for backend, report in reports.items():
        typer.echo(f"[{backend}]")
        typer.echo(f"Cases: {len(report.cases)}")
        typer.echo(f"Recall@{report.top_k}: {report.recall_at_k:.3f}")
        typer.echo(f"MRR: {report.mean_reciprocal_rank:.3f}")

    if "hashing" in reports and "semantic" in reports:
        hashing_report = reports["hashing"]
        semantic_report = reports["semantic"]
        typer.echo(
            "Delta semantic - hashing: "
            f"Recall@{top_k} {semantic_report.recall_at_k - hashing_report.recall_at_k:+.3f}; "
            f"MRR {semantic_report.mean_reciprocal_rank - hashing_report.mean_reciprocal_rank:+.3f}"
        )

    failures = [
        backend for backend, report in reports.items() if not report.meets(min_recall, min_mrr)
    ]
    if failures:
        typer.echo(
            f"Gate failed for {', '.join(failures)}: expected "
            f"Recall@{top_k} >= {min_recall:.3f} and MRR >= {min_mrr:.3f}.",
            err=True,
        )
        raise typer.Exit(code=1)


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
