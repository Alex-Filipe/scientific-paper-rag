from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


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


@dataclass(frozen=True, slots=True)
class PaperDocument:
    """A generation-evaluation document: inline `text` or a `file` found in the papers folder."""

    title: str
    source: str
    text: str | None = None
    file: str | None = None
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class Evidence:
    """A literal quote that supports the answer; `page` is 1-based (None for inline text)."""

    text: str
    page: int | None = None


@dataclass(frozen=True, slots=True)
class GenerationCase:
    id: str
    category: str
    question: str
    reference_answer: str
    facts: tuple[tuple[str, ...], ...]
    """Every group must appear in a correct answer; a group lists accepted alternatives."""
    evidence: tuple[Evidence, ...]

    @property
    def answerable(self) -> bool:
        return bool(self.facts)


@dataclass(frozen=True, slots=True)
class GenerationDataset:
    name: str
    documents: tuple[PaperDocument, ...]
    cases: tuple[GenerationCase, ...]


class Verdict(StrEnum):
    CORRECT = "correct"
    PARTIAL = "partial"
    """Some, but not all, expected facts."""
    WRONG = "wrong"
    """An answer with valid citations but none of the expected facts (or any answer to an
    unanswerable question)."""
    ABSTAINED = "abstained"
    FALLBACK = "fallback"
    """Invalid citations, so the evidence was shown instead of an answer."""


@dataclass(frozen=True, slots=True)
class GradeAudit:
    """Grader decisions on the chunks it saw; `gold` chunks contain one of the evidence quotes."""

    gold_kept: int = 0
    gold_dropped: int = 0
    other_kept: int = 0
    other_dropped: int = 0

    @property
    def graded(self) -> int:
        return self.gold_kept + self.gold_dropped + self.other_kept + self.other_dropped


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """One question answered once by one configuration (a line of the results JSONL)."""

    model: str
    grader_model: str
    embedding: str
    config: str
    top_k: int
    run: int
    case_id: str
    category: str
    answerable: bool
    question: str
    answer: str
    status: str
    verdict: Verdict
    facts_matched: int
    facts_total: int
    cited: tuple[int, ...]
    cited_pages: tuple[int | None, ...]
    cited_gold: bool | None
    """Answered and answerable: does a cited chunk contain an evidence quote?"""
    gold_retrieved: bool | None
    """Answerable: did the first retrieval return a chunk with an evidence quote?"""
    grade: GradeAudit
    rewrites: int
    search_query: str
    latency_s: float
    llm_calls: dict[str, int]
    input_tokens: int
    output_tokens: int
    steps: tuple[str, ...]

    @property
    def total_llm_calls(self) -> int:
        return sum(self.llm_calls.values())

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GenerationRecord":
        values = dict(data)
        values.setdefault("grader_model", values["model"])  # files written before the option
        values["verdict"] = Verdict(values["verdict"])
        values["grade"] = GradeAudit(**values["grade"])
        for field in ("cited", "cited_pages", "steps"):
            values[field] = tuple(values[field])
        return cls(**values)
