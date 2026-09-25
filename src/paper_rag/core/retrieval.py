from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from paper_rag.core.contracts import Chunk, ChunkRepository, Embedder, RetrievedChunk


@dataclass(frozen=True, slots=True)
class ReciprocalRankFusion:
    """Combine ranked lists without comparing their incompatible raw scores."""

    rank_constant: int = 60

    def __post_init__(self) -> None:
        if self.rank_constant <= 0:
            raise ValueError("rank_constant must be positive")

    def fuse(self, rankings: Sequence[Sequence[Chunk]], top_k: int) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        chunks_by_id: dict[str, Chunk] = {}
        scores_by_id: dict[str, float] = {}
        for ranking in rankings:
            for rank, chunk in enumerate(ranking, start=1):
                chunks_by_id[chunk.id] = chunk
                scores_by_id[chunk.id] = scores_by_id.get(chunk.id, 0.0) + 1 / (
                    self.rank_constant + rank
                )

        fused = [
            RetrievedChunk(chunk=chunks_by_id[chunk_id], score=score)
            for chunk_id, score in scores_by_id.items()
        ]
        return sorted(fused, key=lambda result: result.score, reverse=True)[:top_k]


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
