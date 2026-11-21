import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class HashingEmbeddings(Embeddings):
    """Deterministic baseline with no downloads; useful for CI and as a reference point."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_PATTERN.findall(text.casefold()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest) % self.dimensions] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class FastEmbedEmbeddings(Embeddings):
    """Semantic embeddings on CPU via FastEmbed (ONNX Runtime); the model loads lazily."""

    def __init__(self, model_name: str, cache_dir: Path, threads: int = 2) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._model: Any | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [[float(value) for value in vector] for vector in self._load().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _load(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as error:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    "Semantic embeddings require the optional dependency. "
                    'Install it with: python -m pip install -e ".[semantic]"'
                ) from error

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(
                model_name=self.model_name, cache_dir=str(self.cache_dir), threads=self.threads
            )
        return self._model


@dataclass(frozen=True, slots=True)
class ConfiguredEmbeddings:
    embeddings: Embeddings
    embedding_space: str


def create_embeddings(backend: str, model_name: str, cache_dir: Path) -> ConfiguredEmbeddings:
    if backend == "hashing":
        return ConfiguredEmbeddings(HashingEmbeddings(), "hashing-v1")
    if backend == "semantic":
        return ConfiguredEmbeddings(
            FastEmbedEmbeddings(model_name=model_name, cache_dir=cache_dir),
            f"fastembed:{model_name}",
        )
    raise ValueError("Embedding backend must be 'hashing' or 'semantic'")
