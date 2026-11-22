"""Aggregate generation-evaluation records into Markdown tables (mean ± sd across runs)."""

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev

from paper_rag.evaluation.models import GenerationRecord, Verdict

Records = Sequence[GenerationRecord]


@dataclass(frozen=True, slots=True)
class Spread:
    mean: float
    sd: float
    low: float
    high: float

    @classmethod
    def of(cls, values: Sequence[float]) -> "Spread":
        return cls(
            mean=fmean(values),
            sd=stdev(values) if len(values) > 1 else 0.0,
            low=min(values),
            high=max(values),
        )


def _share(
    records: Iterable[GenerationRecord], test: Callable[[GenerationRecord], bool]
) -> float | None:
    selected = list(records)
    return sum(map(test, selected)) / len(selected) if selected else None


def _answerable(records: Records) -> list[GenerationRecord]:
    return [record for record in records if record.answerable]


def _unanswerable(records: Records) -> list[GenerationRecord]:
    return [record for record in records if not record.answerable]


def _grade_rate(records: Records, kind: str) -> float | None:
    kept = sum(getattr(record.grade, f"{kind}_kept") for record in records)
    dropped = sum(getattr(record.grade, f"{kind}_dropped") for record in records)
    return (dropped if kind == "gold" else kept) / (kept + dropped) if kept + dropped else None


def _verdict(verdict: Verdict) -> Callable[[GenerationRecord], bool]:
    return lambda record: record.verdict is verdict


PERCENT_METRICS: tuple[tuple[str, Callable[[Records], float | None]], ...] = (
    ("Correct (answerable)", lambda rs: _share(_answerable(rs), _verdict(Verdict.CORRECT))),
    ("Partial (answerable)", lambda rs: _share(_answerable(rs), _verdict(Verdict.PARTIAL))),
    (
        "Wrong with valid citations (answerable)",
        lambda rs: _share(_answerable(rs), _verdict(Verdict.WRONG)),
    ),
    (
        "False abstention (answerable)",
        lambda rs: _share(_answerable(rs), _verdict(Verdict.ABSTAINED)),
    ),
    ("Fallback (answerable)", lambda rs: _share(_answerable(rs), _verdict(Verdict.FALLBACK))),
    (
        "Correct abstention (unanswerable)",
        lambda rs: _share(_unanswerable(rs), _verdict(Verdict.ABSTAINED)),
    ),
    (
        "Answered with citations (unanswerable)",
        lambda rs: _share(_unanswerable(rs), _verdict(Verdict.WRONG)),
    ),
    ("Fallback (unanswerable)", lambda rs: _share(_unanswerable(rs), _verdict(Verdict.FALLBACK))),
    (
        "Cited a gold chunk (answered, answerable)",
        lambda rs: _share(
            (r for r in _answerable(rs) if r.cited_gold is not None),
            lambda r: bool(r.cited_gold),
        ),
    ),
    (
        "Gold chunk in first retrieval",
        lambda rs: _share(_answerable(rs), lambda r: bool(r.gold_retrieved)),
    ),
    ("Grade dropped gold chunks", lambda rs: _grade_rate(_answerable(rs), "gold")),
    ("Grade kept non-gold chunks", lambda rs: _grade_rate(rs, "other")),
    (
        "False abstention after grade dropped gold",
        lambda rs: _share(
            _answerable(rs),
            lambda r: r.verdict is Verdict.ABSTAINED and r.grade.gold_dropped > 0,
        ),
    ),
)

AMOUNT_METRICS: tuple[tuple[str, str, Callable[[Records], float]], ...] = (
    ("Rewrites per question", "{:.2f}", lambda rs: fmean(r.rewrites for r in rs)),
    ("LLM calls per question", "{:.1f}", lambda rs: fmean(r.total_llm_calls for r in rs)),
    (
        "Tokens per question",
        "{:.0f}",
        lambda rs: fmean(r.input_tokens + r.output_tokens for r in rs),
    ),
    ("Latency per question (s)", "{:.1f}", lambda rs: fmean(r.latency_s for r in rs)),
    ("Max latency (s)", "{:.1f}", lambda rs: max(r.latency_s for r in rs)),
)


def load_records(path: Path) -> list[GenerationRecord]:
    return [
        GenerationRecord.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def format_report(records: Records) -> str:
    """One section per embedding backend; columns are the graph configurations."""
    if not records:
        return "No records."
    groups: dict[tuple[str, str, str, str], list[GenerationRecord]] = defaultdict(list)
    for record in records:
        groups[(record.embedding, record.model, record.grader_model, record.pdf_parser)].append(
            record
        )
    return "\n\n".join(_format_embedding(group) for group in groups.values())


def _format_embedding(records: Records) -> str:
    configs = list(dict.fromkeys(record.config for record in records))
    by_run: dict[str, dict[int, list[GenerationRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_run[record.config][record.run].append(record)

    first = records[0]
    runs = sorted({record.run for record in records})
    cases = list(dict.fromkeys(record.case_id for record in records))
    answerable = len({record.case_id for record in records if record.answerable})
    lines = [
        f"### {first.embedding} — {first.model}"
        + (f" (grade/rewrite: {first.grader_model})" if first.grader_model != first.model else "")
        + f", top_k={first.top_k}, PDF parser {first.pdf_parser}, {len(runs)} run(s), "
        f"{len(cases)} cases ({answerable} answerable, {len(cases) - answerable} unanswerable)",
        "",
        "Rates are mean ± sd across runs (min–max).",
        "",
        "| Metric | " + " | ".join(configs) + " |",
        "|---" * (len(configs) + 1) + "|",
    ]
    for label, metric in PERCENT_METRICS:
        cells = [_percent([metric(rs) for rs in by_run[config].values()]) for config in configs]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for label, pattern, amount in AMOUNT_METRICS:
        cells = []
        for config in configs:
            spread = Spread.of([amount(rs) for rs in by_run[config].values()])
            cells.append(
                pattern.format(spread.mean)
                + (f" ± {pattern.format(spread.sd)}" if len(runs) > 1 else "")
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "Per category: correct (answerable) or correct abstention (unanswerable).", ""]
    lines += ["| Category | " + " | ".join(configs) + " |", "|---" * (len(configs) + 1) + "|"]
    for category in dict.fromkeys(record.category for record in records):
        cells = []
        for config in configs:
            values = []
            for run_records in by_run[config].values():
                selected = [r for r in run_records if r.category == category]
                values.append(_share(selected, is_success))
            cells.append(_percent(values))
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    if len(configs) > 1:
        reference = configs[0]
        lines += ["", f"Paired with {reference} (same question, same run):", ""]
        for config in configs[1:]:
            better: list[str] = []
            worse: list[str] = []
            for run in runs:
                before = {r.case_id: is_success(r) for r in by_run[reference][run]}
                for record in by_run[config][run]:
                    if record.case_id in before and is_success(record) != before[record.case_id]:
                        (better if is_success(record) else worse).append(record.case_id)
            lines.append(
                f"- {config}: {len(better) / len(runs):.1f} better and "
                f"{len(worse) / len(runs):.1f} worse per run"
                + _case_list("better", better)
                + _case_list("worse", worse)
            )

    if len(runs) > 1:
        lines += ["", "Stability across runs:", ""]
        for config in configs:
            verdicts: dict[str, set[Verdict]] = defaultdict(set)
            answers: dict[str, set[str]] = defaultdict(set)
            for record in records:
                if record.config == config:
                    verdicts[record.case_id].add(record.verdict)
                    answers[record.case_id].add(record.answer)
            unstable = [case for case in cases if len(verdicts[case]) > 1]
            changed = sum(len(answers[case]) > 1 for case in cases)
            lines.append(
                f"- {config}: {changed} of {len(cases)} cases changed the answer text; "
                f"{len(unstable)} changed the verdict"
                + (f" ({', '.join(unstable)})" if unstable else "")
            )
    return "\n".join(lines)


def is_success(record: GenerationRecord) -> bool:
    """Correct answer to an answerable question, or abstention on an unanswerable one."""
    return record.verdict is (Verdict.CORRECT if record.answerable else Verdict.ABSTAINED)


def _case_list(label: str, case_ids: Sequence[str]) -> str:
    if not case_ids:
        return ""
    counts = Counter(case_ids)
    return f"; {label}: " + ", ".join(
        case if count == 1 else f"{case} ×{count}" for case, count in counts.items()
    )


def _percent(values: Sequence[float | None]) -> str:
    present = [value for value in values if value is not None]
    if not present:
        return "—"
    spread = Spread.of(present)
    if len(present) == 1:
        return f"{spread.mean:.0%}"
    return f"{spread.mean:.0%} ± {spread.sd * 100:.0f} ({spread.low:.0%}–{spread.high:.0%})"
