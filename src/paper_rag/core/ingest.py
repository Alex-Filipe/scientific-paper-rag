import hashlib
from dataclasses import dataclass

from paper_rag.core.contracts import Chunk, ChunkRepository, Document, Embedder


@dataclass(frozen=True, slots=True)
class WordChunker:
    size: int = 220
    overlap: int = 40

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.overlap < 0 or self.overlap >= self.size:
            raise ValueError("Chunk overlap must be between zero and size - 1")

    def split(self, document: Document) -> list[Chunk]:
        words = document.text.split()
        step = self.size - self.overlap
        chunks: list[Chunk] = []

        for position, start in enumerate(range(0, len(words), step)):
            text = " ".join(words[start : start + self.size])
            if not text:
                continue
            chunks.append(
                Chunk(
                    id=f"{document.id}:{position}",
                    document_id=document.id,
                    document_title=document.title,
                    source=document.source,
                    position=position,
                    text=text,
                )
            )

        return chunks


@dataclass(slots=True)
class IngestDocument:
    chunker: WordChunker
    embedder: Embedder
    repository: ChunkRepository

    def execute(self, title: str, text: str, source: str) -> int:
        if not title.strip():
            raise ValueError("Document title cannot be empty")
        if not text.strip():
            raise ValueError("Document text cannot be empty")

        identity = hashlib.sha256(f"{source}:{title}".encode()).hexdigest()[:16]
        document = Document(id=identity, title=title.strip(), text=text, source=source)
        chunks = self.chunker.split(document)
        embeddings = self.embedder.embed([chunk.text for chunk in chunks])
        self.repository.add(chunks, embeddings)
        return len(chunks)
