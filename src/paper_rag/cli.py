from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import typer

from paper_rag.evaluation.dataset import load_dataset
from paper_rag.evaluation.models import RetrievalReport
from paper_rag.evaluation.runner import RetrievalEvaluator
from paper_rag.service import build_service
from paper_rag.settings import Settings

app = typer.Typer(help="Ingest and query scientific papers.", no_args_is_help=True)


class EvaluationEmbedding(StrEnum):
    HASHING = "hashing"
    SEMANTIC = "semantic"
    BOTH = "both"


class EvaluationRetrieval(StrEnum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    BOTH = "both"


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    title: Annotated[str | None, typer.Option(help="Document title.")] = None,
) -> None:
    """Ingest one PDF, Markdown, or text document."""
    count = build_service().ingest_file(path, title=title)
    typer.echo(f"Ingested {count} chunks from {path.name}.")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question about the indexed papers.")],
    top_k: Annotated[int | None, typer.Option(min=1, max=20)] = None,
    show_steps: Annotated[bool, typer.Option(help="Print the LangGraph steps.")] = False,
) -> None:
    """Ask a question and print the answer with its retrieved sources."""
    try:
        answer = build_service().ask(question, top_k)
    except (ConnectionError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    if show_steps:
        for step in answer.steps:
            typer.echo(f"· {step.node}: {step.detail}")
        typer.echo()
    typer.echo(answer.text)
    if answer.citations:
        typer.echo("\nSources:")
        for citation in answer.citations:
            page = f", p. {citation.page}" if citation.page else ""
            marker = "*" if citation.cited else " "
            typer.echo(f"[{citation.index}]{marker} {citation.title} — {citation.source}{page}")


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
    retrieval_mode: Annotated[
        EvaluationRetrieval,
        typer.Option(help="Retrieval mode to evaluate; use 'both' to compare."),
    ] = EvaluationRetrieval.VECTOR,
    top_k: Annotated[int, typer.Option(min=1, max=20)] = 3,
    min_recall: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.0,
    min_mrr: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.0,
) -> None:
    """Evaluate retrieval or compare embedding and search strategies."""
    evaluation_dataset = load_dataset(dataset)
    configured = Settings.from_environment()
    backends = (
        ("hashing", "semantic") if embedding is EvaluationEmbedding.BOTH else (embedding.value,)
    )
    modes = (
        ("vector", "hybrid")
        if retrieval_mode is EvaluationRetrieval.BOTH
        else (retrieval_mode.value,)
    )
    reports: dict[str, RetrievalReport] = {}
    with TemporaryDirectory(prefix="paper-rag-evaluation-") as directory:
        for backend in backends:
            for mode in modes:
                label = f"{backend}/{mode}"
                run_settings = replace(
                    configured,
                    chroma_dir=Path(directory) / f"{backend}-{mode}",
                    embedding_backend=backend,
                    retrieval_mode=mode,
                    generation_backend="extractive",
                )
                reports[label] = RetrievalEvaluator(build_service(run_settings)).run(
                    evaluation_dataset, top_k
                )

    for label, report in reports.items():
        typer.echo(f"[{label}]")
        typer.echo(f"Cases: {len(report.cases)}")
        typer.echo(f"Recall@{report.top_k}: {report.recall_at_k:.3f}")
        typer.echo(f"MRR: {report.mean_reciprocal_rank:.3f}")

    if retrieval_mode is EvaluationRetrieval.BOTH:
        for backend in backends:
            vector_report = reports[f"{backend}/vector"]
            hybrid_report = reports[f"{backend}/hybrid"]
            recall_delta = hybrid_report.recall_at_k - vector_report.recall_at_k
            mrr_delta = hybrid_report.mean_reciprocal_rank - vector_report.mean_reciprocal_rank
            typer.echo(
                f"Delta hybrid - vector ({backend}): "
                f"Recall@{top_k} {recall_delta:+.3f}; MRR {mrr_delta:+.3f}"
            )

    if embedding is EvaluationEmbedding.BOTH:
        for mode in modes:
            hashing_report = reports[f"hashing/{mode}"]
            semantic_report = reports[f"semantic/{mode}"]
            recall_delta = semantic_report.recall_at_k - hashing_report.recall_at_k
            mrr_delta = semantic_report.mean_reciprocal_rank - hashing_report.mean_reciprocal_rank
            typer.echo(
                f"Delta semantic - hashing ({mode}): "
                f"Recall@{top_k} {recall_delta:+.3f}; MRR {mrr_delta:+.3f}"
            )

    failures = [label for label, report in reports.items() if not report.meets(min_recall, min_mrr)]
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
