import json
from dataclasses import replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import typer

from paper_rag.evaluation.dataset import load_dataset, load_generation_dataset
from paper_rag.evaluation.models import GenerationRecord, RetrievalReport
from paper_rag.evaluation.report import format_report, load_records
from paper_rag.evaluation.runner import (
    GENERATION_CONFIGS,
    GenerationEvaluator,
    RetrievalEvaluator,
    reset_ollama_cache,
)
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


class GraphConfig(StrEnum):
    LINEAR = "linear"
    GRADE = "grade"
    GRADE_REWRITE = "grade-rewrite"


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
def evaluate_generation(
    dataset: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ] = Path("evaluation/generation_resnet.json"),
    papers_dir: Annotated[
        Path | None,
        typer.Option(help="Folder with the dataset PDFs (default: the dataset's folder)."),
    ] = None,
    config: Annotated[
        list[GraphConfig] | None,
        typer.Option(help="Graph configuration; repeat to compare (default: all three)."),
    ] = None,
    embedding: Annotated[
        EvaluationEmbedding,
        typer.Option(help="Embedding backend; use 'both' to evaluate each."),
    ] = EvaluationEmbedding.HASHING,
    runs: Annotated[int, typer.Option(min=1, max=20, help="Repetitions of every case.")] = 3,
    top_k: Annotated[int, typer.Option(min=1, max=20)] = 4,
    case: Annotated[
        list[str] | None, typer.Option(help="Only these case ids (repeatable).")
    ] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Only the first N cases.")] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Results JSONL (default: .data/evaluation/generation-*.jsonl)."),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="Print the report of an existing results JSONL."
        ),
    ] = None,
    cold: Annotated[
        bool,
        typer.Option(help="Ollama: reload the model before each question to empty its cache."),
    ] = True,
) -> None:
    """Compare graph configurations on answers, citations, abstention and cost (calls the LLM)."""
    if report is not None:
        typer.echo(format_report(load_records(report)))
        return

    configured = Settings.from_environment()
    if configured.generation_backend == "extractive":
        typer.echo("Set RAG_GENERATION_BACKEND to 'ollama' or 'openai'.", err=True)
        raise typer.Exit(code=2)
    model = (
        configured.ollama_model
        if configured.generation_backend == "ollama"
        else configured.openai_model
    )

    evaluation_dataset = load_generation_dataset(dataset)
    cases = evaluation_dataset.cases
    if case:
        unknown = sorted(set(case) - {item.id for item in cases})
        if unknown:
            typer.echo(f"Unknown case ids: {', '.join(unknown)}", err=True)
            raise typer.Exit(code=2)
        cases = tuple(item for item in cases if item.id in case)
    evaluation_dataset = replace(evaluation_dataset, cases=cases[:limit])

    configs = list(dict.fromkeys(config or list(GraphConfig)))
    backends = (
        ("hashing", "semantic") if embedding is EvaluationEmbedding.BOTH else (embedding.value,)
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = (
        output or Path(".data/evaluation") / f"generation-{evaluation_dataset.name}-{stamp}.jsonl"
    )
    results.parent.mkdir(parents=True, exist_ok=True)

    grader_model = configured.grader_model.strip()
    before_case = None
    if cold and configured.generation_backend == "ollama":
        loaded = list(dict.fromkeys(filter(None, (configured.ollama_model, grader_model))))

        def before_case() -> None:
            for name in loaded:
                reset_ollama_cache(configured.ollama_base_url, name)

    records: list[GenerationRecord] = []
    try:
        with (
            TemporaryDirectory(prefix="paper-rag-generation-") as directory,
            results.open("w", encoding="utf-8") as sink,
        ):

            def save(record: GenerationRecord) -> None:
                sink.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                sink.flush()
                typer.echo(
                    f"[{record.embedding} {record.config} run {record.run}] {record.case_id}: "
                    f"{record.verdict.value} ({record.latency_s:.1f} s)",
                    err=True,
                )

            for backend in backends:
                services = {
                    name.value: build_service(
                        replace(
                            configured,
                            chroma_dir=Path(directory) / backend,
                            embedding_backend=backend,
                            top_k=top_k,
                            grade_documents=GENERATION_CONFIGS[name.value].grade_documents,
                            max_rewrites=GENERATION_CONFIGS[name.value].max_rewrites,
                        )
                    )
                    for name in configs
                }
                evaluator = GenerationEvaluator(
                    services=services,
                    model=model,
                    embedding=backend,
                    top_k=top_k,
                    before_case=before_case,
                    grader_model=grader_model,
                )
                evaluator.ingest(evaluation_dataset, papers_dir or dataset.parent)
                if before_case is None:  # warm-up, so the first latency excludes model loading
                    next(iter(services.values())).ask(evaluation_dataset.cases[0].question, top_k)
                records += evaluator.run(evaluation_dataset, runs, on_record=save)
    except (ConnectionError, RuntimeError, FileNotFoundError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    typer.echo(format_report(records))
    typer.echo(f"\nResults: {results}")


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
