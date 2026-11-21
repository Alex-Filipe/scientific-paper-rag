import re
import unicodedata
from collections.abc import Sequence

from paper_rag.evaluation.models import Evidence, Verdict
from paper_rag.models import AnswerStatus

_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")
_SYMBOLS = str.maketrans({"×": "x", "−": "-", "–": "-", "’": "'", "“": '"', "”": '"'})


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


def normalize_text(text: str) -> str:
    """Casefold, drop accents, unify symbols and decimal commas (4,49 -> 4.49)."""
    decomposed = unicodedata.normalize("NFKD", text.translate(_SYMBOLS).casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(_DECIMAL_COMMA.sub(".", plain).split())


def squash(text: str) -> str:
    """Normalized text without whitespace: PDF extraction splits words (`T able`, `W e`)."""
    return "".join(normalize_text(text).split())


def contains_fact(text: str, fact: str) -> bool:
    """Whole-token match: `0.1` does not match `10.1` and `28` does not match `28.54`, but a
    number may carry a unit (`1.7` matches `1.7M`)."""
    needle = normalize_text(fact)
    if not needle:
        return False
    before = r"(?<![0-9])(?<![0-9]\.)" if needle[0].isdigit() else r"(?<![0-9a-z])"
    after = r"(?![0-9])(?!\.[0-9])" if needle[-1].isdigit() else r"(?![0-9a-z])"
    return re.search(before + re.escape(needle) + after, normalize_text(text)) is not None


def matched_fact_groups(text: str, facts: Sequence[Sequence[str]]) -> int:
    return sum(any(contains_fact(text, alternative) for alternative in group) for group in facts)


def contains_evidence(chunk_text: str, chunk_page: int | None, evidence: Evidence) -> bool:
    if evidence.page is not None and chunk_page != evidence.page:
        return False
    return squash(evidence.text) in squash(chunk_text)


def verdict_for(status: AnswerStatus, facts_matched: int, facts_total: int) -> Verdict:
    """Lexical verdict; any answer to an unanswerable question (no facts) is wrong."""
    if status is AnswerStatus.ABSTAINED:
        return Verdict.ABSTAINED
    if status in {AnswerStatus.FALLBACK, AnswerStatus.EXTRACTIVE}:
        return Verdict.FALLBACK
    if facts_total and facts_matched == facts_total:
        return Verdict.CORRECT
    if facts_matched:
        return Verdict.PARTIAL
    return Verdict.WRONG
