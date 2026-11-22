from dataclasses import asdict, dataclass
from enum import StrEnum


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    """An LLM synthesized the answer and every citation points to a retrieved chunk."""
    EXTRACTIVE = "extractive"
    """No LLM configured: the answer lists the retrieved evidence."""
    FALLBACK = "fallback"
    """The LLM answer had missing or invalid citations, or numbers its cited chunks do not
    contain, so the evidence is shown instead."""
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
    text: str = ""
    """The whole chunk; `excerpt` holds only its beginning, which may miss the cited number."""


@dataclass(frozen=True, slots=True)
class Step:
    node: str
    detail: str


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    """One indexed document, as listed by `GET /documents`."""

    document_id: str
    title: str
    source: str
    chunks: int
    pages: int | None
    """Highest page number among its chunks; None for text documents."""


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    """Read-only view of the service settings, as returned by `GET /config`."""

    generation_backend: str
    model: str | None
    """Generation model; None in extractive mode."""
    grader_model: str | None
    """Model of the grade and rewrite nodes; None when they reuse the generation model."""
    embedding_backend: str
    retrieval_mode: str
    top_k: int
    grade_documents: bool
    max_rewrites: int
    validate_numbers: bool


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
