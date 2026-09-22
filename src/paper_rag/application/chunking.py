from dataclasses import dataclass

from paper_rag.domain.models import Chunk, Document


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
