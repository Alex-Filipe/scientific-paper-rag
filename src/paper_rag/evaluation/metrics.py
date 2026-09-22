from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    expected = set(relevant)
    if not expected:
        raise ValueError("At least one relevant source is required")
    return len(expected.intersection(retrieved)) / len(expected)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    expected = set(relevant)
    if not expected:
        raise ValueError("At least one relevant source is required")

    for rank, source in enumerate(retrieved, start=1):
        if source in expected:
            return 1.0 / rank
    return 0.0
