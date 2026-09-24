from dataclasses import dataclass, field
from enum import StrEnum

from paper_rag.application.ranking import ReciprocalRankFusion
from paper_rag.domain.models import RetrievedChunk
from paper_rag.domain.ports import ChunkRepository, Embedder


class RetrievalMode(StrEnum):
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass(slots=True)
class RetrieveChunks:
    embedder: Embedder
    repository: ChunkRepository
    mode: RetrievalMode = RetrievalMode.HYBRID
    fusion: ReciprocalRankFusion = field(default_factory=ReciprocalRankFusion)

    def execute(self, question: str, top_k: int) -> list[RetrievedChunk]:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self.embedder.embed([question])[0]
        vector_results = self.repository.search(query_embedding, top_k)
        if self.mode is RetrievalMode.VECTOR:
            return vector_results

        lexical_results = self.repository.search_lexical(question, top_k)
        return self.fusion.fuse(
            [
                [result.chunk for result in vector_results],
                lexical_results,
            ],
            top_k,
        )
