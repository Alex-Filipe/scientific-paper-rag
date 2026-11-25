"""Cross-encoder reranking of the hybrid search candidates (FastEmbed/ONNX, optional extra).

The hybrid search ranks chunks by fusing BM25 and vector rankings, without ever reading the
question and the chunk together. A cross-encoder does: it scores each (question, chunk) pair in a
single forward pass, so it can reorder the ~20 fused candidates before the cut at `top_k`.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document


class Reranker(Protocol):
    model_name: str

    def rerank(self, query: str, documents: Sequence[Document], top_k: int) -> list[Document]:
        """Best `top_k` documents for the query, with `rank`, `search_rank` and `rerank_score`."""
        ...


CUSTOM_CROSS_ENCODERS: dict[str, dict[str, Any]] = {
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": {
        "model_file": "onnx/model.onnx",
        "description": "Multilingual MiniLM (L12, H384) fine-tuned on mMARCO (14 languages).",
        "license": "apache-2.0",
        "size_in_gb": 0.47,
    },
}
"""Cross-encoders missing from FastEmbed's list, registered on first use."""


class CrossEncoderReranker:
    """Score each (question, chunk) pair with a cross-encoder; the model loads lazily."""

    def __init__(self, model_name: str, cache_dir: Path, threads: int | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._model: Any | None = None

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        return [float(score) for score in self._load().rerank(query, list(texts))]

    def rerank(self, query: str, documents: Sequence[Document], top_k: int) -> list[Document]:
        """Reorder by score (ties keep the search order) and keep the best `top_k`.

        `rank` becomes the position after reranking; the search position moves to `search_rank`.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = self.scores(query, [document.page_content for document in documents])
        order = sorted(range(len(documents)), key=lambda index: -scores[index])
        return [
            Document(
                page_content=documents[index].page_content,
                metadata={
                    **documents[index].metadata,
                    "search_rank": documents[index].metadata.get("rank", index + 1),
                    "rerank_score": scores[index],
                    "rank": rank,
                },
                id=documents[index].id,
            )
            for rank, index in enumerate(order[:top_k], start=1)
        ]

    def _load(self) -> Any:
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as error:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    "The reranker requires the optional dependency. "
                    'Install it with: python -m pip install -e ".[rerank]"'
                ) from error

            _register_custom_model(TextCrossEncoder, self.model_name)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextCrossEncoder(
                model_name=self.model_name, cache_dir=str(self.cache_dir), threads=self.threads
            )
        return self._model


def _register_custom_model(encoder_class: Any, model_name: str) -> None:
    spec = CUSTOM_CROSS_ENCODERS.get(model_name)
    if spec is None:
        return
    registered = {model["model"] for model in encoder_class.list_supported_models()}
    if model_name in registered:
        return
    from fastembed.common.model_description import ModelSource

    encoder_class.add_custom_model(model=model_name, sources=ModelSource(hf=model_name), **spec)
