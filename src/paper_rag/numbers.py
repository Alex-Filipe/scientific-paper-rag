"""Number reading shared by answer validation and the screen (plain `re`, no LangChain)."""

import re

CITATION_BLOCK = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
NUMBER = re.compile(
    r"(?<![\w.,])(\d+(?:[.,]\d+)*)\s*(k|mil|thousand|m|million|milh\w*|billion|bilh\w*)?(?![a-z])",
    re.IGNORECASE,
)
SCALES = {"k": 1e3, "mil": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "billion": 1e9}


def number_readings(text: str) -> set[float]:
    """Every plausible value of every number in `text`.

    `4,49` and `4.49` read as 4.49; `50.000` as 50 or 50000 (Portuguese thousands); `50k`, `50 mil`
    and `1.7M` also as 50000 and 1700000. Comparing readings tolerates the formats a model uses
    when it copies a number from an English paper into a Portuguese answer.
    """
    readings: set[float] = set()
    for match in NUMBER.finditer(text):
        digits, scale = match.group(1), (match.group(2) or "").casefold()
        values = []
        groups = re.split(r"[.,]", digits)
        if len(groups) <= 2:
            values.append(float(".".join(groups)))
        if len(groups) > 1 and all(len(group) == 3 for group in groups[1:]):
            values.append(float("".join(groups)))
        factor = SCALES.get(scale) or (1e6 if scale.startswith("milh") else 0.0)
        factor = factor or (1e9 if scale.startswith("bilh") else 0.0)
        for value in values:
            readings.add(round(value, 6))
            if factor:
                readings.add(round(value * factor, 6))
    return readings


def answer_numbers(answer: str) -> list[str]:
    """Numbers stated in an answer, ignoring citations `[1]` and paper references `[10, 41]`."""
    tokens = [match.group(0).strip() for match in NUMBER.finditer(CITATION_BLOCK.sub(" ", answer))]
    return list(dict.fromkeys(tokens))


def claimed_readings(answer: str, question: str) -> set[float]:
    """Readings of the numbers the answer adds to the question: the ones validation checks."""
    known = number_readings(question)
    claimed: set[float] = set()
    for token in answer_numbers(answer):
        readings = number_readings(token)
        if not readings & known:
            claimed |= readings
    return claimed
