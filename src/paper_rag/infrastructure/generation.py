from collections.abc import Sequence
from dataclasses import dataclass

from paper_rag.domain.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class ExtractiveAnswerGenerator:
    """Safe local baseline that exposes retrieval evidence without calling an LLM."""

    max_excerpt_chars: int = 420

    def generate(self, question: str, contexts: Sequence[RetrievedChunk]) -> str:
        del question
        if not contexts:
            return "Não encontrei evidências no corpus para responder à pergunta."

        excerpts = [
            f"[{index}] {result.chunk.text[: self.max_excerpt_chars].strip()}"
            for index, result in enumerate(contexts, start=1)
        ]
        return "Evidências mais relevantes encontradas:\n\n" + "\n\n".join(excerpts)
