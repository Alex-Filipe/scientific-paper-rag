import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from paper_rag.evaluation.models import EvaluationDocument, RetrievalCase, RetrievalDataset


def load_dataset(path: Path) -> RetrievalDataset:
    root = _as_mapping(json.loads(path.read_text(encoding="utf-8")), "dataset")
    documents = tuple(
        _parse_document(item) for item in _as_list(root.get("documents"), "documents")
    )
    cases = tuple(_parse_case(item) for item in _as_list(root.get("cases"), "cases"))

    if not documents:
        raise ValueError("Evaluation dataset must contain documents")
    if not cases:
        raise ValueError("Evaluation dataset must contain cases")
    return RetrievalDataset(documents=documents, cases=cases)


def _parse_document(value: object) -> EvaluationDocument:
    item = _as_mapping(value, "document")
    return EvaluationDocument(
        title=_required_string(item, "title"),
        text=_required_string(item, "text"),
        source=_required_string(item, "source"),
    )


def _parse_case(value: object) -> RetrievalCase:
    item = _as_mapping(value, "case")
    sources = tuple(
        _as_string(source, "relevant_sources item")
        for source in _as_list(item.get("relevant_sources"), "relevant_sources")
    )
    if not sources:
        raise ValueError("Evaluation case must contain relevant_sources")
    return RetrievalCase(question=_required_string(item, "question"), relevant_sources=sources)


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _as_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return cast(list[object], value)


def _required_string(item: Mapping[str, object], field: str) -> str:
    return _as_string(item.get(field), field)


def _as_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
