import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from paper_rag.evaluation.models import (
    EvaluationDocument,
    Evidence,
    GenerationCase,
    GenerationDataset,
    PaperDocument,
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
    """Load a generation dataset; content checks against the ingested chunks happen later."""
    root = _as_mapping(json.loads(path.read_text(encoding="utf-8")), "dataset")
    name = root.get("name")
    documents = tuple(
        _parse_paper_document(item) for item in _as_list(root.get("documents"), "documents")
    )
    cases = tuple(_parse_generation_case(item) for item in _as_list(root.get("cases"), "cases"))

    if not documents:
        raise ValueError("Generation dataset must contain documents")
    if not cases:
        raise ValueError("Generation dataset must contain cases")
    duplicated = sorted({case.id for case in cases if [c.id for c in cases].count(case.id) > 1})
    if duplicated:
        raise ValueError(f"Duplicated case ids: {', '.join(duplicated)}")
    return GenerationDataset(
        name=name.strip() if isinstance(name, str) and name.strip() else path.stem,
        documents=documents,
        cases=cases,
    )


def _parse_paper_document(value: object) -> PaperDocument:
    item = _as_mapping(value, "document")
    title = _required_string(item, "title")
    has_text, has_file = item.get("text") is not None, item.get("file") is not None
    if has_text == has_file:
        raise ValueError(f"Document {title!r} needs exactly one of 'text' or 'file'")
    if has_text:
        return PaperDocument(
            title=title,
            source=_required_string(item, "source"),
            text=_required_string(item, "text"),
        )
    file = _required_string(item, "file")
    sha256 = item.get("sha256")
    return PaperDocument(
        title=title,
        source=_as_string(item["source"], "source") if "source" in item else file,
        file=file,
        sha256=_as_string(sha256, "sha256") if sha256 is not None else None,
    )


def _parse_generation_case(value: object) -> GenerationCase:
    item = _as_mapping(value, "case")
    case_id = _required_string(item, "id")
    facts = tuple(_parse_fact_group(group) for group in _as_list(item.get("facts"), "facts"))
    evidence = tuple(_parse_evidence(entry) for entry in _as_list(item.get("evidence"), "evidence"))
    if bool(facts) != bool(evidence):
        raise ValueError(
            f"Case {case_id!r}: answerable cases need facts and evidence; "
            "unanswerable cases need neither"
        )
    return GenerationCase(
        id=case_id,
        category=_required_string(item, "category"),
        question=_required_string(item, "question"),
        reference_answer=_required_string(item, "reference_answer"),
        facts=facts,
        evidence=evidence,
    )


def _parse_fact_group(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (_as_string(value, "fact"),)
    group = tuple(_as_string(alternative, "fact") for alternative in _as_list(value, "facts"))
    if not group:
        raise ValueError("A fact group must list at least one alternative")
    return group


def _parse_evidence(value: object) -> Evidence:
    item = _as_mapping(value, "evidence")
    page = item.get("page")
    if page is not None and (not isinstance(page, int) or isinstance(page, bool) or page < 1):
        raise ValueError("evidence page must be a positive integer")
    return Evidence(text=_required_string(item, "text"), page=page)


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
