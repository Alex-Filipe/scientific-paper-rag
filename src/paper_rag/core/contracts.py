from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    text: str
    source: str


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    document_title: str
    source: str
    position: int
    text: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    title: str
    source: str
    excerpt: str
    score: float


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Convert texts into vectors in the same embedding space."""


class ChunkRepository(Protocol):
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Persist chunks and their vectors."""

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        """Return the most relevant chunks, ordered by descending score."""

    def search_lexical(self, query: str, top_k: int) -> list[Chunk]:
        """Return chunks ordered by lexical relevance."""


class AnswerGenerator(Protocol):
    def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        """Generate a grounded answer from retrieved contexts."""
