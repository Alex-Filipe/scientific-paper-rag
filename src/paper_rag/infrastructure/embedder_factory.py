from dataclasses import dataclass
from pathlib import Path

from paper_rag.domain.ports import Embedder
from paper_rag.infrastructure.embeddings import HashingEmbedder
from paper_rag.infrastructure.fastembed import FastEmbedder


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
