import pytest
from conftest import ServiceFactory
from langchain_core.documents import Document

from paper_rag.retrieval import BM25Retriever, lexical_tokens, reciprocal_rank_fusion


def test_lexical_tokens_strip_accents_and_stop_words() -> None:
    assert lexical_tokens("A responsabilidade sistêmica DA rede") == [
        "responsabilidade",
        "sistemica",
        "rede",
    ]


def test_hybrid_search_matches_diacritic_variants(make_service: ServiceFactory) -> None:
    service = make_service()
    service.ingest_text("Notas", "A responsabilidade sistêmica define o limite.", "target")
    service.ingest_text("Outras", "O procedimento descreve a manutenção preventiva.", "other")

    results = service.retrieve("responsabilidade sistemica", top_k=1)

    assert [document.metadata["source"] for document in results] == ["target"]
    assert results[0].metadata["rank"] == 1


def test_reingesting_a_document_removes_its_old_chunks(make_service: ServiceFactory) -> None:
    service = make_service(chunk_size=3, chunk_overlap=0)
    service.ingest_text("Paper", "solar array offline today again here", "paper.txt")
    assert service.search.index.count() == 2

    service.ingest_text("Paper", "wind turbine", "paper.txt")

    assert service.search.index.count() == 1
    assert service.retrieve("wind turbine", top_k=3)[0].page_content == "wind turbine"


def test_hybrid_search_promotes_lexical_matches(make_service: ServiceFactory) -> None:
    hybrid = make_service()
    for source, text in [
        ("catalyst", "The cobalt catalyst reduces reaction time."),
        ("noise", "Reaction reaction reaction kinetics overview."),
        ("other", "Unrelated survey of datasets."),
    ]:
        hybrid.ingest_text(source.title(), text, source)

    results = hybrid.retrieve("cobalt catalyst", top_k=2)

    assert results[0].metadata["source"] == "catalyst"


def test_vector_mode_skips_lexical_search(make_service: ServiceFactory) -> None:
    service = make_service(retrieval_mode="vector")
    service.ingest_text("Paper", "Embeddings represent text as vectors.", "paper")

    assert service.retrieve("vectors", top_k=3)[0].metadata["source"] == "paper"


def test_empty_corpus_returns_nothing(make_service: ServiceFactory) -> None:
    assert make_service().retrieve("anything", top_k=3) == []


@pytest.mark.parametrize(("question", "top_k"), [("  ", 3), ("ok", 0)])
def test_search_validates_input(make_service: ServiceFactory, question: str, top_k: int) -> None:
    with pytest.raises(ValueError):
        make_service().retrieve(question, top_k)


def make_document(chunk_id: str) -> Document:
    return Document(page_content=chunk_id, metadata={"chunk_id": chunk_id})


def test_rrf_promotes_chunks_shared_by_both_rankers() -> None:
    first, shared, last = (make_document(name) for name in ("first", "shared", "last"))

    fused = reciprocal_rank_fusion([[first, shared], [shared, last]], weights=[0.5, 0.5])

    assert [document.metadata["chunk_id"] for document in fused] == ["shared", "first", "last"]
    assert fused[0].metadata["score"] > fused[1].metadata["score"]


def test_rrf_validates_arguments() -> None:
    with pytest.raises(ValueError, match="weight"):
        reciprocal_rank_fusion([[make_document("a")]], weights=[0.5, 0.5])
    with pytest.raises(ValueError, match="constant"):
        reciprocal_rank_fusion([[make_document("a")]], weights=[1.0], c=0)


def test_bm25_ignores_stop_words_and_unrelated_chunks() -> None:
    retriever = BM25Retriever(
        documents=[
            Document(page_content="A reranker receives candidates.", metadata={"chunk_id": "a"}),
            Document(page_content="Solar panels on the roof.", metadata={"chunk_id": "b"}),
        ]
    )

    assert retriever.invoke("a o de que and the") == []
    assert [document.metadata["chunk_id"] for document in retriever.invoke("reranker")] == ["a"]
