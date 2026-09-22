from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationDocument:
    title: str
    text: str
    source: str


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    question: str
    relevant_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalDataset:
    documents: tuple[EvaluationDocument, ...]
    cases: tuple[RetrievalCase, ...]


@dataclass(frozen=True, slots=True)
class CaseResult:
    question: str
    retrieved_sources: tuple[str, ...]
    recall_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    top_k: int
    recall_at_k: float
    mean_reciprocal_rank: float
    cases: tuple[CaseResult, ...]

    def meets(self, minimum_recall: float, minimum_mrr: float) -> bool:
        return self.recall_at_k >= minimum_recall and self.mean_reciprocal_rank >= minimum_mrr
