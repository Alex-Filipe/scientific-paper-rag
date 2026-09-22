from dataclasses import dataclass


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
