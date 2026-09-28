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
class GenerationCase:
    question: str
    expected_facts: tuple[str, ...]
    expected_abstention: bool


@dataclass(frozen=True, slots=True)
class GenerationDataset:
    documents: tuple[EvaluationDocument, ...]
    cases: tuple[GenerationCase, ...]


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


@dataclass(frozen=True, slots=True)
class GenerationCaseResult:
    question: str
    answer: str
    abstained: bool
    cited_indexes: tuple[int, ...]
    retrieved_sources: tuple[str, ...]
    cited_sources: tuple[str, ...]
    answer_facts_match: bool | None
    citation_valid: bool | None
    citation_supported: bool | None
    abstention_correct: bool


@dataclass(frozen=True, slots=True)
class GenerationReport:
    backend: str
    model: str
    answer_fact_match_rate: float | None
    citation_valid_rate: float | None
    citation_support_rate: float | None
    abstention_accuracy: float
    cases: tuple[GenerationCaseResult, ...]
