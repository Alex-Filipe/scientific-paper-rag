import hashlib
import json
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from paper_rag.evaluation.metrics import (
    contains_evidence,
    contains_fact,
    matched_fact_groups,
    recall_at_k,
    reciprocal_rank,
    verdict_for,
)
from paper_rag.evaluation.models import (
    CaseResult,
    GenerationCase,
    GenerationDataset,
    GenerationRecord,
    GradeAudit,
    PaperDocument,
    RetrievalDataset,
    RetrievalReport,
)
from paper_rag.models import AnswerStatus
from paper_rag.service import RagService


@dataclass(slots=True)
class RetrievalEvaluator:
    service: RagService

    def run(self, dataset: RetrievalDataset, top_k: int) -> RetrievalReport:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        for document in dataset.documents:
            self.service.ingest_text(document.title, document.text, document.source)

        results = tuple(
            self._evaluate_case(case.question, case.relevant_sources, top_k)
            for case in dataset.cases
        )
        return RetrievalReport(
            top_k=top_k,
            recall_at_k=fmean(result.recall_at_k for result in results),
            mean_reciprocal_rank=fmean(result.reciprocal_rank for result in results),
            cases=results,
        )

    def _evaluate_case(
        self,
        question: str,
        relevant_sources: tuple[str, ...],
        top_k: int,
    ) -> CaseResult:
        retrieved = self.service.retrieve(question, top_k)
        sources = tuple(str(document.metadata.get("source", "")) for document in retrieved)
        return CaseResult(
            question=question,
            retrieved_sources=sources,
            recall_at_k=recall_at_k(sources, relevant_sources),
            reciprocal_rank=reciprocal_rank(sources, relevant_sources),
        )


class LLMUsage(BaseCallbackHandler):
    """Count chat-model calls per graph node and the tokens they report."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.input_tokens = 0
        self.output_tokens = 0

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.calls[str((metadata or {}).get("langgraph_node", "other"))] += 1

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    self.input_tokens += int(usage.get("input_tokens", 0))
                    self.output_tokens += int(usage.get("output_tokens", 0))


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    name: str
    grade_documents: bool
    max_rewrites: int


GENERATION_CONFIGS: dict[str, GenerationConfig] = {
    config.name: config
    for config in (
        GenerationConfig("linear", grade_documents=False, max_rewrites=0),
        GenerationConfig("grade", grade_documents=True, max_rewrites=0),
        GenerationConfig("grade-rewrite", grade_documents=True, max_rewrites=1),
    )
}


def resolve_paper(document: PaperDocument, papers_dir: Path) -> Path:
    """Find a dataset PDF and check its hash, so pages and quotes match the annotated version."""
    assert document.file is not None
    path = papers_dir / document.file
    if not path.is_file():
        raise FileNotFoundError(
            f"Paper not found: {path}. Pass --papers-dir with the folder that holds it."
        )
    if document.sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != document.sha256:
            raise ValueError(f"{path.name} differs from the annotated version (sha256 {digest})")
    return path


def reset_ollama_cache(base_url: str, model: str, timeout: float = 120.0) -> None:
    """Unload and reload the model so each question starts with an empty prompt cache.

    Ollama reuses the KV cache of any earlier identical prompt (a repeated 1,600-token prompt
    drops from 2 s to 0.05 s), and the configurations share prompts, so without this the
    latency would depend on the order of the runs. Reloading takes about 0.7 s, not timed.
    """
    url = f"{base_url.rstrip('/')}/api/generate"
    for body in (
        {"model": model, "keep_alive": 0},
        {"model": model, "prompt": "ok", "stream": False, "options": {"num_predict": 1}},
    ):
        request = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()


@dataclass(slots=True)
class GenerationEvaluator:
    """Ask every case with each graph configuration; all services share one index."""

    services: Mapping[str, RagService]
    model: str
    embedding: str
    top_k: int
    before_case: Callable[[], None] | None = None
    grader_model: str = ""
    """Runs before each question, outside the timing (e.g. `reset_ollama_cache`)."""

    def ingest(self, dataset: GenerationDataset, papers_dir: Path) -> None:
        service = next(iter(self.services.values()))
        for document in dataset.documents:
            if document.text is not None:
                service.ingest_text(document.title, document.text, document.source)
            else:
                service.ingest_file(
                    resolve_paper(document, papers_dir),
                    title=document.title,
                    source=document.source,
                )
        check_dataset_against_index(dataset, service.search.index.all_chunks())

    def run(
        self,
        dataset: GenerationDataset,
        runs: int,
        on_record: Callable[[GenerationRecord], None] | None = None,
    ) -> list[GenerationRecord]:
        """Interleave configurations inside each run so drift over time hits all of them."""
        if runs <= 0:
            raise ValueError("runs must be positive")
        records: list[GenerationRecord] = []
        for run in range(1, runs + 1):
            for config in self.services:
                for case in dataset.cases:
                    if self.before_case is not None:
                        self.before_case()
                    record = self.evaluate_case(config, run, case)
                    records.append(record)
                    if on_record is not None:
                        on_record(record)
        return records

    def evaluate_case(self, config: str, run: int, case: GenerationCase) -> GenerationRecord:
        usage = LLMUsage()
        started = perf_counter()
        trace = self.services[config].trace(case.question, self.top_k, callbacks=[usage])
        latency = perf_counter() - started
        answer = trace.answer

        facts_matched = matched_fact_groups(answer.text, case.facts)
        verdict = verdict_for(answer.status, facts_matched, len(case.facts))
        cited = tuple(citation.index for citation in answer.citations if citation.cited)
        first_retrieval = first_contexts(trace.updates)
        answered = answer.status is AnswerStatus.ANSWERED
        return GenerationRecord(
            model=self.model,
            grader_model=self.grader_model or self.model,
            embedding=self.embedding,
            config=config,
            top_k=self.top_k,
            run=run,
            case_id=case.id,
            category=case.category,
            answerable=case.answerable,
            question=case.question,
            answer=answer.text,
            status=answer.status.value,
            verdict=verdict,
            facts_matched=facts_matched,
            facts_total=len(case.facts),
            cited=cited,
            cited_pages=tuple(answer.citations[index - 1].page for index in cited),
            cited_gold=(
                any(is_gold(trace.contexts[index - 1], case) for index in cited)
                if answered and case.answerable
                else None
            ),
            gold_retrieved=(
                any(is_gold(chunk, case) for chunk in first_retrieval) if case.answerable else None
            ),
            grade=audit_grade(trace.updates, case),
            rewrites=sum(node == "rewrite" for node, _ in trace.updates),
            search_query=answer.search_query,
            latency_s=round(latency, 3),
            llm_calls=dict(usage.calls),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            steps=tuple(f"{step.node}: {step.detail}" for step in answer.steps),
            pdf_parser=self.services[config].settings.pdf_parser,
            rerank_model=getattr(self.services[config].reranker, "model_name", ""),
        )


def is_gold(chunk: Document, case: GenerationCase) -> bool:
    page = chunk.metadata.get("page")
    return any(
        contains_evidence(chunk.page_content, page if isinstance(page, int) else None, evidence)
        for evidence in case.evidence
    )


RETRIEVAL_NODES = ("retrieve", "rerank")


def first_contexts(updates: Sequence[tuple[str, dict[str, Any]]]) -> list[Document]:
    """Chunks of the first retrieval as the next node sees them (after rerank, when it runs)."""
    contexts: list[Document] = []
    for node, update in updates:
        if node not in RETRIEVAL_NODES:
            break
        contexts = update.get("contexts", contexts)
    return contexts


def audit_grade(updates: Sequence[tuple[str, dict[str, Any]]], case: GenerationCase) -> GradeAudit:
    """Compare what each grade step kept with the chunks the preceding retrieve/rerank returned."""
    counts = Counter[str]()
    retrieved: list[Document] = []
    for node, update in updates:
        if node in RETRIEVAL_NODES:
            retrieved = update.get("contexts", retrieved)
        elif node == "grade" and "contexts" in update:  # absent when grading was skipped
            kept = {_chunk_id(chunk) for chunk in update["contexts"]}
            for chunk in retrieved:
                kind = "gold" if is_gold(chunk, case) else "other"
                counts[f"{kind}_{'kept' if _chunk_id(chunk) in kept else 'dropped'}"] += 1
    return GradeAudit(**counts)


def check_dataset_against_index(dataset: GenerationDataset, chunks: Sequence[Document]) -> None:
    """Fail early when a quote or fact is not in the ingested text (e.g. after a parser change)."""
    problems: list[str] = []
    corpus = "\n".join(chunk.page_content for chunk in chunks)
    for case in dataset.cases:
        for evidence in case.evidence:
            if not any(
                contains_evidence(
                    chunk.page_content,
                    page if isinstance(page := chunk.metadata.get("page"), int) else None,
                    evidence,
                )
                for chunk in chunks
            ):
                where = f" (p. {evidence.page})" if evidence.page else ""
                problems.append(f"{case.id}: quote not found{where}: {evidence.text!r}")
        for group in case.facts:
            if not any(contains_fact(corpus, alternative) for alternative in group):
                problems.append(f"{case.id}: no alternative of {list(group)} in the corpus")
    if problems:
        raise ValueError("Dataset does not match the ingested text:\n" + "\n".join(problems))


def _chunk_id(chunk: Document) -> str:
    return str(chunk.metadata.get("chunk_id", chunk.id or ""))
