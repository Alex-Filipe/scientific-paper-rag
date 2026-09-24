from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
                from fastembed import TextEmbedding  # type: ignore[import-not-found]
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
