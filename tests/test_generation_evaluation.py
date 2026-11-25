import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from conftest import ServiceFactory, fake_cross_encoder, make_pdf
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import SimpleChatModel
from langchain_core.messages import BaseMessage
from pytest import MonkeyPatch
from typer.testing import CliRunner

import paper_rag.cli as cli
from paper_rag.evaluation.dataset import load_generation_dataset
from paper_rag.evaluation.metrics import contains_evidence, contains_fact, verdict_for
from paper_rag.evaluation.models import Evidence, GenerationDataset, GenerationRecord, Verdict
from paper_rag.evaluation.report import format_report, load_records
from paper_rag.evaluation.runner import (
    GENERATION_CONFIGS,
    GenerationEvaluator,
    reset_ollama_cache,
    resolve_paper,
)
from paper_rag.generation import GenerationBackend
from paper_rag.graph import RagNodes, build_graph
from paper_rag.models import AnswerStatus
from paper_rag.service import RagService, build_service
from paper_rag.settings import Settings

BASELINE = Path("evaluation/generation_baseline.json")
RESNET = Path("evaluation/generation_resnet.json")


class RoutedChatModel(SimpleChatModel):
    """Fake LLM that answers by role: grader, query rewriter or answer generator."""

    grade: str = "sim"
    rewrite: str = "consulta reescrita"
    answer: str = "Não encontrei evidências suficientes no corpus para responder à pergunta."

    @property
    def _llm_type(self) -> str:
        return "routed-fake"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        system = messages[0].text
        if "avalia se um trecho" in system:
            return self.grade
        if "Reescreva a pergunta" in system:
            return self.rewrite
        return self.answer


def with_llm(service: RagService, llm: RoutedChatModel, config: str) -> RagService:
    options = GENERATION_CONFIGS[config]
    nodes = RagNodes(
        search=service.search,
        llm=llm,
        backend=GenerationBackend.OLLAMA,
        grade_documents=options.grade_documents,
        max_rewrites=options.max_rewrites,
    )
    return replace(service, graph=build_graph(nodes))


@pytest.fixture
def baseline() -> GenerationDataset:
    return load_generation_dataset(BASELINE)


def evaluator_for(
    make_service: ServiceFactory, llm: RoutedChatModel, *configs: str
) -> GenerationEvaluator:
    service = make_service()
    return GenerationEvaluator(
        services={config: with_llm(service, llm, config) for config in configs},
        model="fake",
        embedding="hashing",
        top_k=4,
    )


def test_datasets_load() -> None:
    resnet = load_generation_dataset(RESNET)
    categories = {case.category for case in resnet.cases}

    assert len(resnet.cases) == 26
    assert categories == {"direct", "vague", "table", "portuguese", "unanswerable"}
    assert sum(not case.answerable for case in resnet.cases) == 5
    assert resnet.documents[0].file == "He_Deep_Residual_Learning_CVPR_2016_paper.pdf"
    assert len(load_generation_dataset(BASELINE).cases) == 6


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ({"facts": [["x"]], "evidence": []}, "answerable cases need facts and evidence"),
        ({"facts": [], "evidence": [{"text": "x"}]}, "answerable cases need facts and evidence"),
        ({"facts": [["x"]], "evidence": [{"text": "x", "page": 0}]}, "positive integer"),
    ],
)
def test_dataset_validation(tmp_path: Path, case: dict[str, object], message: str) -> None:
    path = tmp_path / "dataset.json"
    base = {"id": "a", "category": "direct", "question": "q?", "reference_answer": "r"}
    documents = [{"title": "T", "source": "s", "text": "x"}]
    path.write_text(json.dumps({"documents": documents, "cases": [base | case]}))

    with pytest.raises(ValueError, match=message):
        load_generation_dataset(path)


def test_dataset_rejects_duplicate_ids_and_ambiguous_documents(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    case = {"id": "a", "category": "c", "question": "q", "reference_answer": "r", "facts": []}
    case["evidence"] = []
    documents = [{"title": "T", "source": "s", "text": "x"}]
    path.write_text(json.dumps({"documents": documents, "cases": [case, case]}))
    with pytest.raises(ValueError, match="Duplicated case ids: a"):
        load_generation_dataset(path)

    documents = [{"title": "T", "source": "s", "text": "x", "file": "x.pdf"}]
    path.write_text(json.dumps({"documents": documents, "cases": [case]}))
    with pytest.raises(ValueError, match="exactly one of 'text' or 'file'"):
        load_generation_dataset(path)


@pytest.mark.parametrize(
    ("text", "fact", "expected"),
    [
        ("O erro foi de 4,49%.", "4.49", True),
        ("Erro de 14.49%.", "4.49", False),
        ("10.1", "0.1", False),
        ("Starts from 0.1.", "0.1", True),
        ("28.54% top-1", "28", False),
        ("a 28% gain", "28", True),
        ("1.7M parameters", "1.7", True),
        ("blocks 1×1 and 3×3", "1x1", True),
        ("o problema de degradação", "degradação", True),
        ("degradations", "degradation", False),
    ],
)
def test_contains_fact(text: str, fact: str, expected: bool) -> None:
    assert contains_fact(text, fact) is expected


def test_evidence_ignores_pdf_spacing_and_checks_page() -> None:
    evidence = Evidence(text="See Table 2 for details", page=5)

    assert contains_evidence("See T able 2 for\ndetails.", 5, evidence)
    assert not contains_evidence("See T able 2 for details.", 4, evidence)


def test_verdicts() -> None:
    assert verdict_for(AnswerStatus.ANSWERED, 2, 2) is Verdict.CORRECT
    assert verdict_for(AnswerStatus.ANSWERED, 1, 2) is Verdict.PARTIAL
    assert verdict_for(AnswerStatus.ANSWERED, 0, 1) is Verdict.WRONG
    assert verdict_for(AnswerStatus.ANSWERED, 0, 0) is Verdict.WRONG  # unanswerable
    assert verdict_for(AnswerStatus.ABSTAINED, 0, 1) is Verdict.ABSTAINED
    assert verdict_for(AnswerStatus.FALLBACK, 1, 1) is Verdict.FALLBACK


def test_linear_config_answers_with_one_call(
    make_service: ServiceFactory, baseline: GenerationDataset
) -> None:
    evaluator = evaluator_for(make_service, RoutedChatModel(answer="Redução de 37% [1]."), "linear")
    evaluator.ingest(baseline, BASELINE.parent)

    record = evaluator.evaluate_case("linear", 1, baseline.cases[0])

    assert record.verdict is Verdict.CORRECT
    assert record.llm_calls == {"generate": 1}
    assert record.cited == (1,)
    assert record.cited_gold is True
    assert record.gold_retrieved is True
    assert record.grade.graded == 0


def test_grade_audit_counts_dropped_gold_chunks(
    make_service: ServiceFactory, baseline: GenerationDataset
) -> None:
    llm = RoutedChatModel(grade="não")
    evaluator = evaluator_for(make_service, llm, "grade", "grade-rewrite")
    evaluator.ingest(baseline, BASELINE.parent)

    grade = evaluator.evaluate_case("grade", 1, baseline.cases[0])
    rewrite = evaluator.evaluate_case("grade-rewrite", 1, baseline.cases[0])

    assert grade.verdict is Verdict.ABSTAINED
    assert grade.grade.gold_dropped == 1
    assert grade.llm_calls == {"grade": grade.grade.graded}
    assert rewrite.rewrites == 1
    assert rewrite.search_query == "consulta reescrita"
    assert rewrite.llm_calls["rewrite"] == 1
    assert rewrite.grade.graded > grade.grade.graded  # graded again after the rewrite


def test_unanswerable_answer_counts_as_wrong(
    make_service: ServiceFactory, baseline: GenerationDataset
) -> None:
    evaluator = evaluator_for(make_service, RoutedChatModel(answer="A FAPEMIG [1]."), "linear")
    evaluator.ingest(baseline, BASELINE.parent)
    unanswerable = next(case for case in baseline.cases if not case.answerable)

    record = evaluator.evaluate_case("linear", 1, unanswerable)

    assert record.verdict is Verdict.WRONG
    assert record.cited_gold is None
    assert record.gold_retrieved is None


def test_run_interleaves_configs_and_reports(
    make_service: ServiceFactory, baseline: GenerationDataset, tmp_path: Path
) -> None:
    evaluator = evaluator_for(make_service, RoutedChatModel(), "linear", "grade")
    evaluator.ingest(baseline, BASELINE.parent)
    seen: list[str] = []

    records = evaluator.run(baseline, runs=2, on_record=lambda record: seen.append(record.config))

    assert len(records) == 2 * 2 * len(baseline.cases)
    assert seen[: len(baseline.cases) + 1] == ["linear"] * len(baseline.cases) + ["grade"]
    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(json.dumps(record.to_dict()) for record in records))
    assert load_records(results) == records
    assert records[0].grader_model == "fake"
    legacy = {key: value for key, value in records[0].to_dict().items() if key != "grader_model"}
    assert GenerationRecord.from_dict(legacy).grader_model == "fake"  # older JSONL files
    assert records[0].pdf_parser == "pypdf"
    before_parser = {key: value for key, value in legacy.items() if key != "pdf_parser"}
    assert GenerationRecord.from_dict(before_parser).pdf_parser == "pypdf"
    report = format_report(records)
    assert "PDF parser pypdf" in report
    assert "| Metric | linear | grade |" in report
    assert "| Correct abstention (unanswerable) | 100% ± 0 (100%–100%) |" in report
    assert "linear: 0 of 6 cases changed the answer text" in report
    # The fake grader keeps every chunk, so both configurations answer alike.
    assert "- grade: 0.0 better and 0.0 worse per run" in report


def test_ingest_checks_quotes_against_the_index(
    make_service: ServiceFactory, baseline: GenerationDataset
) -> None:
    case = replace(baseline.cases[0], evidence=(Evidence(text="frase que não existe"),))
    broken = replace(baseline, cases=(case,))
    evaluator = evaluator_for(make_service, RoutedChatModel(), "linear")

    with pytest.raises(ValueError, match="quote not found"):
        evaluator.ingest(broken, BASELINE.parent)


def test_pdf_documents_are_found_and_hash_checked(
    make_service: ServiceFactory, tmp_path: Path
) -> None:
    (tmp_path / "paper.pdf").write_bytes(
        make_pdf([["Intro"], ["Residual nets reach 4.49% error."]])
    )
    case = {
        "id": "n1",
        "category": "direct",
        "question": "What error?",
        "reference_answer": "4.49%",
        "facts": [["4.49"]],
        "evidence": [{"page": 2, "text": "reach 4.49% error"}],
    }
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps({"documents": [{"title": "Paper", "file": "paper.pdf"}], "cases": [case]})
    )
    dataset = load_generation_dataset(path)
    evaluator = evaluator_for(make_service, RoutedChatModel(answer="4,49% [1]."), "linear")

    evaluator.ingest(dataset, tmp_path)
    record = evaluator.evaluate_case("linear", 1, dataset.cases[0])

    assert record.verdict is Verdict.CORRECT
    assert record.cited_pages == (2,)
    with pytest.raises(FileNotFoundError, match="--papers-dir"):
        resolve_paper(dataset.documents[0], tmp_path / "missing")
    with pytest.raises(ValueError, match="differs from the annotated version"):
        resolve_paper(replace(dataset.documents[0], sha256="0" * 64), tmp_path)


def test_trace_keeps_node_updates(make_service: ServiceFactory) -> None:
    service = make_service()
    service.ingest_text("RAG", "RAG combines retrieval and generation.", "rag.md")
    traced = with_llm(service, RoutedChatModel(answer="RAG combina [1]."), "grade")

    trace = traced.trace("What is RAG?")

    assert [node for node, _ in trace.updates] == ["retrieve", "grade", "generate", "validate"]
    assert trace.answer.status is AnswerStatus.ANSWERED
    assert len(trace.contexts) == 1


def test_reset_ollama_cache_unloads_then_reloads(monkeypatch: MonkeyPatch) -> None:
    sent: list[tuple[str, dict[str, object]]] = []

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request: Any, timeout: float) -> Response:
        sent.append((request.full_url, json.loads(request.data)))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    reset_ollama_cache("http://localhost:11434/", "qwen")

    assert [url for url, _ in sent] == ["http://localhost:11434/api/generate"] * 2
    assert sent[0][1] == {"model": "qwen", "keep_alive": 0}
    assert sent[1][1]["prompt"] == "ok"


runner = CliRunner()


def test_cli_requires_an_llm_backend() -> None:
    result = runner.invoke(cli.app, ["evaluate-generation", str(BASELINE)])

    assert result.exit_code == 2
    assert "RAG_GENERATION_BACKEND" in result.output


def test_cli_runs_and_reprints_report(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    def fake_build_service(settings: Settings | None = None) -> RagService:
        assert settings is not None
        service = build_service(replace(settings, generation_backend="extractive"))
        config = (
            "linear"
            if not settings.grade_documents
            else "grade-rewrite"
            if settings.max_rewrites
            else "grade"
        )
        return with_llm(service, RoutedChatModel(answer="Foi 37% [1]."), config)

    resets: list[tuple[str, str]] = []
    monkeypatch.setenv("RAG_GENERATION_BACKEND", "ollama")
    monkeypatch.setattr(cli, "build_service", fake_build_service)
    monkeypatch.setattr(cli, "reset_ollama_cache", lambda *args: resets.append(args))
    results = tmp_path / "results.jsonl"

    result = runner.invoke(
        cli.app,
        [
            "evaluate-generation",
            str(BASELINE),
            "--config",
            "linear",
            "--config",
            "grade",
            "--case",
            "aurora-reduction",
            "--case",
            "aurora-funding",
            "--runs",
            "2",
            "--output",
            str(results),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "| Metric | linear | grade |" in result.stdout
    assert len(results.read_text().splitlines()) == 2 * 2 * 2
    assert len(resets) == 2 * 2 * 2  # an empty prompt cache before every question
    reprinted = runner.invoke(cli.app, ["evaluate-generation", "--report", str(results)])
    assert reprinted.exit_code == 0
    assert "| Correct (answerable) | 100% ± 0 (100%–100%) |" in reprinted.stdout


def test_reranked_contexts_are_the_first_retrieval(
    make_service: ServiceFactory, baseline: GenerationDataset
) -> None:
    service = make_service(rerank=True)
    fake_cross_encoder(service)
    nodes = RagNodes(
        search=service.search,
        llm=RoutedChatModel(grade="não"),
        backend=GenerationBackend.OLLAMA,
        grade_documents=True,
        max_rewrites=0,
        reranker=service.reranker,
    )
    evaluator = GenerationEvaluator(
        services={"grade": replace(service, graph=build_graph(nodes))},
        model="fake",
        embedding="hashing",
        top_k=2,
    )
    evaluator.ingest(baseline, BASELINE.parent)
    assert service.search.index.count() > 2

    record = evaluator.evaluate_case("grade", 1, baseline.cases[0])

    assert record.steps[1].startswith("rerank: ")
    assert record.grade.graded == 2  # the reranked top-2, not every candidate
    assert record.rerank_model == service.settings.rerank_model
    assert f"reranker {record.rerank_model}" in format_report([record])
    older = {key: value for key, value in record.to_dict().items() if key != "rerank_model"}
    assert GenerationRecord.from_dict(older).rerank_model == ""
