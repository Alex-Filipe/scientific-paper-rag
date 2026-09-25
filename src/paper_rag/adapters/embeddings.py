import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paper_rag.core.contracts import Embedder

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class HashingEmbedder:
    """Small deterministic baseline; replace with a semantic embedding model later."""

    dimensions: int = 256

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_PATTERN.findall(text.casefold()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest) % self.dimensions
            vector[bucket] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


@dataclass(slots=True)
class FastEmbedder:
    """Optional CPU semantic embedder backed by FastEmbed and ONNX Runtime."""

    model_name: str
    cache_dir: Path
    threads: int = 2
    _model: Any | None = field(default=None, init=False, repr=False)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        return [[float(value) for value in vector] for vector in model.embed(list(texts))]

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as error:
                raise RuntimeError(
                    "Semantic embeddings require the optional dependency. "
                    'Install it with: python -m pip install -e ".[semantic]"'
                ) from error

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
                threads=self.threads,
            )
        return self._model


@dataclass(frozen=True, slots=True)
class ConfiguredEmbedder:
    embedder: Embedder
    embedding_space: str


def create_embedder(backend: str, model_name: str, cache_dir: Path) -> ConfiguredEmbedder:
    if backend == "hashing":
        return ConfiguredEmbedder(HashingEmbedder(), "hashing-v1")
    if backend == "semantic":
        return ConfiguredEmbedder(
            FastEmbedder(model_name=model_name, cache_dir=cache_dir),
            f"fastembed:{model_name}",
        )
    raise ValueError("Embedding backend must be 'hashing' or 'semantic'")
