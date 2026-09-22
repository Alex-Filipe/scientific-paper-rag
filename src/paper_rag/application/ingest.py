import hashlib
from dataclasses import dataclass

from paper_rag.application.chunking import WordChunker
from paper_rag.domain.models import Document
from paper_rag.domain.ports import ChunkRepository, Embedder


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
