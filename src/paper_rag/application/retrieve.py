from dataclasses import dataclass

from paper_rag.domain.models import RetrievedChunk
from paper_rag.domain.ports import ChunkRepository, Embedder


@dataclass(slots=True)
class RetrieveChunks:
    embedder: Embedder
    repository: ChunkRepository

    def execute(self, question: str, top_k: int) -> list[RetrievedChunk]:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self.embedder.embed([question])[0]
        return self.repository.search(query_embedding, top_k)
