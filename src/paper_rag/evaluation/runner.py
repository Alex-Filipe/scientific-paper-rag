import re
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean

from paper_rag.adapters.generation import ABSTENTION_ANSWER
from paper_rag.core.ask import AskQuestion
from paper_rag.core.ingest import IngestDocument
from paper_rag.core.retrieval import RetrieveChunks
from paper_rag.evaluation.metrics import recall_at_k, reciprocal_rank
from paper_rag.evaluation.models import (
    CaseResult,
    GenerationCase,
    GenerationCaseResult,
    GenerationDataset,
    GenerationReport,
    RetrievalDataset,
    RetrievalReport,
)


@dataclass(slots=True)
class RetrievalEvaluator:
    ingest_document: IngestDocument
    retrieve_chunks: RetrieveChunks

    def run(self, dataset: RetrievalDataset, top_k: int) -> RetrievalReport:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        for document in dataset.documents:
            self.ingest_document.execute(document.title, document.text, document.source)

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
        retrieved = self.retrieve_chunks.execute(question, top_k)
        sources = tuple(result.chunk.source for result in retrieved)
        return CaseResult(
            question=question,
            retrieved_sources=sources,
            recall_at_k=recall_at_k(sources, relevant_sources),
            reciprocal_rank=reciprocal_rank(sources, relevant_sources),
        )


@dataclass(slots=True)
class GenerationEvaluator:
    ingest_document: IngestDocument
    ask_question: AskQuestion
    backend: str
    model: str

    def run(self, dataset: GenerationDataset) -> GenerationReport:
        for document in dataset.documents:
            self.ingest_document.execute(document.title, document.text, document.source)

        results = tuple(self._evaluate_case(case) for case in dataset.cases)
        answerable_results = tuple(
            result for result in results if result.answer_facts_match is not None
        )
        return GenerationReport(
            backend=self.backend,
            model=self.model,
            answer_fact_match_rate=_rate(
                result.answer_facts_match for result in answerable_results
            ),
            citation_valid_rate=_rate(result.citation_valid for result in answerable_results),
            citation_support_rate=_rate(
                result.citation_supported for result in answerable_results
            ),
            abstention_accuracy=fmean(float(result.abstention_correct) for result in results),
            cases=results,
        )

    def _evaluate_case(self, case: GenerationCase) -> GenerationCaseResult:
        answer = self.ask_question.execute(case.question)
        abstained = answer.text.strip().casefold() == ABSTENTION_ANSWER.casefold()
        cited_indexes = tuple(int(value) for value in re.findall(r"\[(\d+)\]", answer.text))
        citation_valid = bool(cited_indexes) and all(
            1 <= index <= len(answer.citations) for index in cited_indexes
        )
        cited_sources = tuple(
            answer.citations[index - 1].source
            for index in cited_indexes
            if 1 <= index <= len(answer.citations)
        )

        if case.expected_abstention:
            answer_facts_match = None
            case_citation_valid = None
            citation_supported = None
        else:
            normalized_answer = answer.text.casefold()
            answer_facts_match = not abstained and all(
                fact.casefold() in normalized_answer for fact in case.expected_facts
            )
            case_citation_valid = not abstained and citation_valid
            cited_excerpts = "\n".join(
                answer.citations[index - 1].excerpt
                for index in cited_indexes
                if 1 <= index <= len(answer.citations)
            ).casefold()
            citation_supported = (
                not abstained
                and citation_valid
                and all(fact.casefold() in cited_excerpts for fact in case.expected_facts)
            )

        return GenerationCaseResult(
            question=case.question,
            answer=answer.text,
            abstained=abstained,
            cited_indexes=cited_indexes,
            retrieved_sources=tuple(citation.source for citation in answer.citations),
            cited_sources=cited_sources,
            answer_facts_match=answer_facts_match,
            citation_valid=case_citation_valid,
            citation_supported=citation_supported,
            abstention_correct=case.expected_abstention == abstained,
        )


def _rate(values: Iterable[bool | None]) -> float | None:
    eligible = tuple(value for value in values if value is not None)
    if not eligible:
        return None
    return fmean(float(value) for value in eligible)
