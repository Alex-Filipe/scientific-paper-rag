from paper_rag.core.contracts import Document
from paper_rag.core.ingest import WordChunker


def test_chunker_preserves_overlap() -> None:
    document = Document(
        id="paper-1",
        title="Paper",
        text="one two three four five six seven",
        source="paper.txt",
    )

    chunks = WordChunker(size=4, overlap=2).split(document)

    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "three four five six"
    assert chunks[0].id == "paper-1:0"


def test_chunker_rejects_overlap_equal_to_size() -> None:
    try:
        WordChunker(size=4, overlap=4)
    except ValueError as error:
        assert "overlap" in str(error).lower()
    else:
        raise AssertionError("Expected invalid overlap to raise ValueError")
