from dataclasses import dataclass
from statistics import fmean

from paper_rag.core.ingest import IngestDocument
from paper_rag.core.retrieval import RetrieveChunks
from paper_rag.evaluation.metrics import recall_at_k, reciprocal_rank
from paper_rag.evaluation.models import CaseResult, RetrievalDataset, RetrievalReport


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
