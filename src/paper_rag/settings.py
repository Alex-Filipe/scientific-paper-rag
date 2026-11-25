import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on", "sim"}


@dataclass(frozen=True, slots=True)
class Settings:
    chroma_dir: Path = Path(".data/chroma")
    chunk_size: int = 220
    chunk_overlap: int = 40
    top_k: int = 4
    candidate_k: int = 20
    embedding_backend: str = "hashing"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_cache_dir: Path = Path(".data/models")
    retrieval_mode: str = "hybrid"
    generation_backend: str = "extractive"
    openai_model: str = "gpt-6-astra"
    ollama_model: str = "qwen2.5:1.5b-instruct"
    ollama_base_url: str = "http://localhost:11434"
    # Linear by default: with qwen2.5:1.5b the grader kept gold and other chunks alike (27%)
    # and turned right answers into abstentions; see docs/architecture.md.
    grade_documents: bool = False
    max_rewrites: int = 0
    grader_model: str = ""
    """Model for the grade and rewrite nodes; empty means the generation model."""
    validate_numbers: bool = True
    pdf_parser: str = "pypdf"
    """`pypdf` (text layer) or `docling` (layout and table models, optional extra)."""
    index_references: bool = False
    """Keep the bibliography in the index; only parsers with sections (docling) can drop it."""
    rerank: bool = False
    """Reorder the `candidate_k` fused candidates with a cross-encoder before the cut at top_k."""
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    @classmethod
    def from_environment(cls) -> "Settings":
        defaults = cls()
        return cls(
            chroma_dir=Path(os.getenv("RAG_CHROMA_DIR", str(defaults.chroma_dir))),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", str(defaults.chunk_size))),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", str(defaults.chunk_overlap))),
            top_k=int(os.getenv("RAG_TOP_K", str(defaults.top_k))),
            candidate_k=int(os.getenv("RAG_CANDIDATE_K", str(defaults.candidate_k))),
            embedding_backend=os.getenv("RAG_EMBEDDING_BACKEND", defaults.embedding_backend),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", defaults.embedding_model),
            model_cache_dir=Path(os.getenv("RAG_MODEL_CACHE_DIR", str(defaults.model_cache_dir))),
            retrieval_mode=os.getenv("RAG_RETRIEVAL_MODE", defaults.retrieval_mode),
            generation_backend=os.getenv("RAG_GENERATION_BACKEND", defaults.generation_backend),
            openai_model=os.getenv("RAG_OPENAI_MODEL", defaults.openai_model),
            ollama_model=os.getenv("RAG_OLLAMA_MODEL", defaults.ollama_model),
            ollama_base_url=os.getenv("RAG_OLLAMA_BASE_URL", defaults.ollama_base_url),
            grade_documents=_env_bool("RAG_GRADE_DOCUMENTS", defaults.grade_documents),
            max_rewrites=int(os.getenv("RAG_MAX_REWRITES", str(defaults.max_rewrites))),
            grader_model=os.getenv("RAG_GRADER_MODEL", defaults.grader_model),
            validate_numbers=_env_bool("RAG_VALIDATE_NUMBERS", defaults.validate_numbers),
            pdf_parser=os.getenv("RAG_PDF_PARSER", defaults.pdf_parser),
            index_references=_env_bool("RAG_INDEX_REFERENCES", defaults.index_references),
            rerank=_env_bool("RAG_RERANK", defaults.rerank),
            rerank_model=os.getenv("RAG_RERANK_MODEL", defaults.rerank_model),
        )
