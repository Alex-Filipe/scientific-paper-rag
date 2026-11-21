from dataclasses import asdict, dataclass
from enum import StrEnum


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    """An LLM synthesized the answer and every citation points to a retrieved chunk."""
    EXTRACTIVE = "extractive"
    """No LLM configured: the answer lists the retrieved evidence."""
    FALLBACK = "fallback"
    """The LLM answer had missing or invalid citations, so the evidence is shown instead."""
    ABSTAINED = "abstained"
    """No sufficient evidence was found."""


@dataclass(frozen=True, slots=True)
class Citation:
    index: int
    chunk_id: str
    title: str
    source: str
    page: int | None
    excerpt: str
    rank: int
    score: float
    """Reciprocal Rank Fusion score; only meaningful for ordering, not as a probability."""
    cited: bool


@dataclass(frozen=True, slots=True)
class Step:
    node: str
    detail: str


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    status: AnswerStatus
    citations: tuple[Citation, ...]
    steps: tuple[Step, ...]
    search_query: str

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.text,
            "status": self.status.value,
            "search_query": self.search_query,
            "citations": [asdict(citation) for citation in self.citations],
            "steps": [asdict(step) for step in self.steps],
        }
