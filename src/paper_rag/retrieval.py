import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr
from rank_bm25 import BM25Plus

from paper_rag.embeddings import TOKEN_PATTERN
from paper_rag.vectorstore import ChunkIndex

LEXICAL_STOP_WORDS = frozenset(
    {
        "a", "as", "ao", "aos", "com", "da", "das", "de", "do", "dos", "e", "em", "na", "nas",
        "no", "nos", "o", "os", "para", "por", "que", "qual", "quais", "se", "um", "uma",
        "the", "and", "or", "of", "to", "in", "on", "for", "from", "with", "by", "is", "are",
        "was", "were", "how", "what", "which", "when", "where", "why",
    }
)  # fmt: skip


def lexical_tokens(text: str) -> list[str]:
    """Casefold, strip accents and drop stop words so BM25 matches 'sistêmica' to 'sistemica'."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    tokens = TOKEN_PATTERN.findall(without_accents)
    return [token for token in tokens if token not in LEXICAL_STOP_WORDS]


def chunk_key(document: Document) -> str:
    return str(document.metadata.get("chunk_id") or document.id or document.page_content)


class BM25Retriever(BaseRetriever):
    """Lexical retriever over an in-memory BM25+ index.

    Only chunks sharing at least one query term are returned, so unrelated chunks never
    enter the rank fusion. BM25+ keeps IDF positive even in tiny corpora.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[Document]
    k: int = 20
    _index: BM25Plus | None = PrivateAttr(default=None)
    _token_sets: list[frozenset[str]] = PrivateAttr(default_factory=list)

    def model_post_init(self, context: Any, /) -> None:
        tokenized = [lexical_tokens(document.page_content) for document in self.documents]
        self._token_sets = [frozenset(tokens) for tokens in tokenized]
        if any(tokenized):
            self._index = BM25Plus(tokenized)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        del run_manager
        tokens = lexical_tokens(query)
        if self._index is None or not tokens:
            return []
        query_terms = set(tokens)
        scores = self._index.get_scores(tokens)
        matching = [index for index, terms in enumerate(self._token_sets) if terms & query_terms]
        matching.sort(
            key=lambda index: (-scores[index], self.documents[index].metadata.get("position", 0))
        )
        return [self.documents[index] for index in matching[: self.k]]


class RRFRetriever(BaseRetriever):
    """Fuse several retrievers with (weighted) Reciprocal Rank Fusion.

    RRF combines rankings without comparing their raw scores, which live on different scales
    for BM25 and cosine similarity.
    """

    retrievers: list[BaseRetriever]
    weights: list[float]
    c: int = 60
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        rankings = [
            retriever.invoke(query, config={"callbacks": run_manager.get_child()})
            for retriever in self.retrievers
        ]
        return reciprocal_rank_fusion(rankings, self.weights, self.c)[: self.k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Document]], weights: Sequence[float], c: int = 60
) -> list[Document]:
    if len(rankings) != len(weights):
        raise ValueError("Each ranking needs one weight")
    if c <= 0:
        raise ValueError("RRF constant must be positive")

    scores: dict[str, float] = defaultdict(float)
    documents: dict[str, Document] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, document in enumerate(ranking, start=1):
            key = chunk_key(document)
            documents.setdefault(key, document)
            scores[key] += weight / (c + rank)

    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [
        Document(
            page_content=documents[key].page_content,
            metadata={**documents[key].metadata, "score": scores[key]},
            id=documents[key].id,
        )
        for key in ordered
    ]


class RetrievalMode(StrEnum):
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass(slots=True)
class HybridSearch:
    """Vector search in Chroma, optionally fused with BM25 through RRF."""

    index: ChunkIndex
    mode: RetrievalMode = RetrievalMode.HYBRID
    candidate_k: int = 20
    rrf_constant: int = 60

    def as_retriever(self, top_k: int) -> BaseRetriever:
        # Fusion only helps when each ranker contributes more candidates than the final cut.
        candidates = max(self.candidate_k, top_k)
        vector_retriever = self.index.store.as_retriever(search_kwargs={"k": candidates})
        if self.mode is RetrievalMode.VECTOR:
            return RRFRetriever(retrievers=[vector_retriever], weights=[1.0], k=top_k)
        lexical_retriever = BM25Retriever(documents=self.index.all_chunks(), k=candidates)
        return RRFRetriever(
            retrievers=[lexical_retriever, vector_retriever],
            weights=[0.5, 0.5],
            c=self.rrf_constant,
            k=top_k,
        )

    def search(self, query: str, top_k: int) -> list[Document]:
        if not query.strip():
            raise ValueError("Question cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.index.count() == 0:
            return []
        results = self.as_retriever(top_k).invoke(query)
        return [
            Document(
                page_content=document.page_content,
                metadata={**document.metadata, "rank": rank},
                id=document.id,
            )
            for rank, document in enumerate(results, start=1)
        ]
