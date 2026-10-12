"""Evidence finder: locates the words in a pledge that satisfy each of the four checks.

The patterns come from src/rules.py, so a check is reported as found here exactly when
rules.detect_signals reports it as present. Each span is a dictionary with the matched
text and its character positions in the original string, which the web page uses to
highlight the words. Positions are Python string indices, so they match JavaScript
indices for text without characters outside the basic multilingual plane.
"""
import re
from typing import Optional

from src import rules

# Maps the internal signal names to the check ids used by the web page.
CHECK_IDS = {"target_year": "year", "scope3": "scope3", "offsets": "offsets", "milestones": "milestones"}

def _as_span(match: re.Match) -> dict:
    """Converts a regular expression match into an evidence span."""
    return {"evidence": match.group(0), "start": match.start(), "end": match.end()}


def _year_of(match: re.Match) -> int:
    """Returns the target year captured by a TARGET_YEAR or DATED_TARGET match."""
    return int(next(group for group in match.groups() if group))


# Words that introduce a year and are highlighted together with it, for example "by 2040".
_YEAR_LEAD = re.compile(
    r"(?:by the end of|by end of|no later than|at the latest|before|by|in|for|of)\s+(?:the\s+year\s+)?(?:fy\s*)?$", re.I)


def _year_span(text: str, match: re.Match) -> dict:
    """Returns the span of the year inside a match, extended back over its introducing words."""
    group = next(i for i, value in enumerate(match.groups(), start=1) if value)
    start, end = match.start(group), match.end(group)
    lead = _YEAR_LEAD.search(text[match.start():start])
    if lead:
        start = match.start() + lead.start()
    return {"evidence": text[start:end], "start": start, "end": end}


def find_year(text: str) -> Optional[dict]:
    """Returns the span of the most likely target year, or None when no target year is present.

    Only years tied to target wording are considered, which is the definition used by
    rules.detect_signals. The latest such year is chosen, because the final goal is
    normally the furthest date.
    """
    matches = list(rules.TARGET_YEAR.finditer(text))
    return _year_span(text, max(matches, key=_year_of)) if matches else None


def find_milestone(text: str) -> Optional[dict]:
    """Returns the span of an interim milestone, or None when there is none.

    Explicit wording such as "interim target" counts directly. Otherwise two or more
    distinct target years count, and the earliest dated target is reported as the
    interim step. This mirrors rules._has_interim_milestone.
    """
    explicit = rules.EXPLICIT_MILESTONE.search(text)
    if explicit:
        return _as_span(explicit)
    dated = list(rules.DATED_TARGET.finditer(text))
    if len({_year_of(m) for m in dated}) >= 2:
        return _year_span(text, min(dated, key=_year_of))
    return None


def _first_span(pattern: re.Pattern, text: str) -> Optional[dict]:
    """Returns the span of the first match of the pattern, or None when there is no match."""
    match = pattern.search(text)
    return _as_span(match) if match else None


def find_evidence(text: str) -> dict:
    """Returns the evidence span for each check id, or None for a check that is not met."""
    return {
        "year": find_year(text),
        "scope3": _first_span(rules.SCOPE3, text),
        "offsets": _first_span(rules.OFFSETS, text),
        "milestones": find_milestone(text),
    }


def find_target_phrase(text: str, label: str) -> Optional[str]:
    """Returns the words that made the rule based method choose the given label, or None."""
    pattern = {"net_zero": rules.NET_ZERO, "reduction": rules.REDUCTION}.get(label)
    match = pattern.search(text) if pattern else None
    return re.sub(r"\s+", " ", match.group(0)).strip() if match else None
