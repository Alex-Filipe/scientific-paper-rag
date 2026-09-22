from dataclasses import dataclass

from paper_rag.application.retrieve import RetrieveChunks
from paper_rag.domain.models import Answer, Citation
from paper_rag.domain.ports import AnswerGenerator


@dataclass(slots=True)
class AskQuestion:
    retriever: RetrieveChunks
    generator: AnswerGenerator
    default_top_k: int = 4

    def execute(self, question: str, top_k: int | None = None) -> Answer:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        limit = top_k or self.default_top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")

        contexts = self.retriever.execute(question, limit)
        text = self.generator.generate(question, contexts)
        citations = tuple(
            Citation(
                chunk_id=result.chunk.id,
                title=result.chunk.document_title,
                source=result.chunk.source,
                excerpt=result.chunk.text[:280],
                score=result.score,
            )
            for result in contexts
        )
        return Answer(text=text, citations=citations)
