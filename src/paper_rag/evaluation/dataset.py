import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from paper_rag.evaluation.models import (
    EvaluationDocument,
    GenerationCase,
    GenerationDataset,
    RetrievalCase,
    RetrievalDataset,
)


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


def load_generation_dataset(path: Path) -> GenerationDataset:
    root = _as_mapping(json.loads(path.read_text(encoding="utf-8")), "dataset")
    documents = tuple(
        _parse_document(item) for item in _as_list(root.get("documents"), "documents")
    )
    cases = tuple(
        _parse_generation_case(item) for item in _as_list(root.get("cases"), "cases")
    )

    if not documents:
        raise ValueError("Generation dataset must contain documents")
    if not cases:
        raise ValueError("Generation dataset must contain cases")

    corpus = "\n".join(document.text.casefold() for document in documents)
    for case in cases:
        missing_facts = [fact for fact in case.expected_facts if fact.casefold() not in corpus]
        if missing_facts:
            raise ValueError(
                "Expected facts for answerable cases must appear in the evaluation documents: "
                + ", ".join(missing_facts)
            )

    return GenerationDataset(documents=documents, cases=cases)


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


def _parse_generation_case(value: object) -> GenerationCase:
    item = _as_mapping(value, "case")
    facts = tuple(
        _as_string(fact, "expected_facts item")
        for fact in _as_list(item.get("expected_facts"), "expected_facts")
    )
    expected_abstention = item.get("expected_abstention")
    if not isinstance(expected_abstention, bool):
        raise ValueError("expected_abstention must be a boolean")
    if expected_abstention and facts:
        raise ValueError("Abstention cases cannot contain expected_facts")
    if not expected_abstention and not facts:
        raise ValueError("Answerable cases must contain expected_facts")

    return GenerationCase(
        question=_required_string(item, "question"),
        expected_facts=facts,
        expected_abstention=expected_abstention,
    )


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
