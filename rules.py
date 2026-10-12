"""Rule based baseline: target classification and four robustness signal checks.

The rules are intentionally deterministic and interpretable.  Target classification
uses regular expressions.  The robustness checks follow the definitions in the
labelling guide so that a positive flag means the sentence contains evidence for the
signal rather than merely a related keyword.
"""
import re

# Pattern for net zero style language, checked before reduction language.
# Net-zero style language is checked before reduction language.
#
# The alternatives cover common equivalent forms found during validation
# error analysis, including neutrality and explicit zero-emission wording.
NET_ZERO = re.compile(
    r"(?:"
    # Net-zero, allowing small punctuation variations such as "(net)-zero".
    r"\bnet\W{0,3}zero\b"

    # Carbon/climate neutrality terminology.
    r"|\bcarbon[\s\-]?neutral(?:ity)?\b(?!\s+growth\b)"
    r"|\bclimate[\s\-]?neutral(?:ity)?\b"
    r"|\bclimate[\s\-]?positive\b"

    # Net greenhouse-gas/carbon neutrality.
    r"|\bnet\s+(?:greenhouse\s+gas|ghg|carbon)\s+neutral(?:ity)?\b"

    # Emission-neutral wording.
    r"|\bemissions?[\s\-]?neutral\b"

    # Reversed forms such as 'neutrality of greenhouse gas emissions'.
    r"|\bneutrality\s+(?:of\s+)?"
    r"(?:greenhouse\s+gas|ghg|carbon|co2)\s+emissions?\b"

    # Examples such as 'neutral carbon city'.
    r"|\bneutral\s+carbon\b"

    # Explicit zero-emission terminology.
    r"|\bzero\s+(?:net\s+)?"
    r"(?:carbon|emissions?|ghg|greenhouse\s+gas(?:es)?|co2)"
    r"(?:\s+emissions?)?\b"

    # Examples such as 'real zero by 2050'.
    r"|\breal\s+zero\b"

    # Explicit commitment to eliminate emissions.
    r"|\beliminat\w*.{0,30}?"
    r"(?:co2|carbon|ghg|greenhouse\s+gas(?:es)?)\s+emissions?\b"

    # Existing stronger-than-zero terminology.
    r"|\bcarbon[\s\-]?negative\b"

    # Wording occasionally used for a net-zero target.
    r"|\bnet\s+neutrality\b"

    r")",
    re.I | re.S,
)

PERCENT = (
    r"(?:"
    r"\d{1,3}(?:\.\d+)?\s*(?:%|percent)"
    r"|"
    r"\d{1,3}(?:\.\d+)?\s*[-–]\s*"
    r"\d{1,3}(?:\.\d+)?\s*%"
    r")"
)

CLIMATE_NOUN = (
    r"(?:"
    r"emissions?|"
    r"ghg|"
    r"greenhouse\s+gas(?:es)?|"
    r"carbon|"
    r"co2|"
    r"footprint|"
    r"intensity"
    r")"
)

# Pattern for emission reduction language (verb plus emissions noun or a percentage).
REDUCTION = re.compile(
    rf"(?:"

    # Reduction verb followed by a climate term or percentage.
    rf"\b(?:reduc\w+|cut\w*|decreas\w+|lower\w*|halv\w+)\b"
    rf".{{0,80}}?(?:{CLIMATE_NOUN}|{PERCENT})"

    # Climate term followed by reduction wording.
    rf"|{CLIMATE_NOUN}.{{0,80}}?"
    rf"\b(?:reduc\w+|cut\w*|decreas\w+|lower\w*|halv\w+)\b"

    # Percentage before the word reduction:
    # '22.6% GHG reduction', '30-40% reduction'.
    rf"|{PERCENT}.{{0,50}}?\b(?:reduction|reductions)\b"

    # 'reduction in greenhouse gases'.
    rf"|\b(?:reduction|reductions)\b.{{0,50}}?{CLIMATE_NOUN}"

    # 'emissions reduction target/goal'.
    rf"|\b{CLIMATE_NOUN}\s+reduction\s+"
    rf"(?:target|targets|goal|goals)\b"

    # 'eliminate 60% of emissions', 'minimise GHG emissions by 55%'.
    rf"|\b(?:eliminat\w+|minimi[sz]\w+)\b"
    rf".{{0,60}}?(?:{PERCENT}.{{0,30}}?)?{CLIMATE_NOUN}"

    rf"|\b(?:eliminat\w+|minimi[sz]\w+)\b"
    rf".{{0,60}}?{CLIMATE_NOUN}.{{0,30}}?{PERCENT}"

    rf")",
    re.I | re.S,
)

# A target year must be connected to target language.  This avoids treating a base
# year, publication year, or other unrelated date as a robustness signal.
TARGET_YEAR = re.compile(
    r"\b(?:by|before|no later than|at the latest|by the end of|by end of)\s+"
    r"(?:the\s+year\s+)?(?:fy\s*)?(20[2-9]\d)\b"

    r"|\b(?:target|goal)s?\s+(?:for|of)?\s*"
    r"(?:the\s+year\s+)?(?:fy\s*)?(20[2-9]\d)\b"

    r"|\b(?:net\W{0,3}zero|"
    r"carbon[\s\-]?neutral(?:ity)?|"
    r"climate[\s\-]?neutral(?:ity)?|"
    r"emissions?[\s\-]?neutral)"
    r".{0,60}?\b(?:by|in|before)\s+"
    r"(?:the\s+year\s+)?(?:fy\s*)?(20[2-9]\d)\b",
    re.I | re.S,
)

# Scope 3 is positive only when the sentence explicitly covers Scope 3 or equivalent
# value/supply-chain emissions, matching the labelling guide.
SCOPE3 = re.compile(
    r"\bscope\s*(?:3|iii)\b"
    r"|\b(?:full|entire)?\s*value[\s\-]?chain(?:\s+emissions?)?\b"
    r"|\bsupply[\s\-]?chain\s+emissions?\b",
    re.I,
)

# The guide requires more than a generic mention of offsets.  A positive result needs
# language describing use, reliance, limitation, non-use, or treatment of residual
# emissions with offsets/credits/removals.
OFFSETS = re.compile(
    r"\b(?:use|using|used|rely|relying|reliance|purchase|purchasing|retire|retiring|"
    r"limit|limiting|limited|restrict|restricted)\b.{0,50}?"
    r"\b(?:offset\w*|carbon\s+credit\w*|carbon\s+removal\w*)\b"
    r"|\bwithout\b.{0,40}?\b(?:offset\w*|carbon\s+credit\w*|carbon\s+removal\w*)\b"
    r"|\b(?:residual|remaining|unabated)\s+emissions?\b.{0,60}?"
    r"\b(?:offset\w*|carbon\s+credit\w*|carbon\s+removal\w*|neutrali[sz]\w*|balance\w*)\b"
    r"|\b(?:offset\w*|carbon\s+credit\w*|carbon\s+removal\w*)\b.{0,60}?"
    r"\b(?:residual|remaining|unabated)\s+emissions?\b",
    re.I | re.S,
)

# Explicit wording can establish an interim milestone directly.
EXPLICIT_MILESTONE = re.compile(
    r"\b(?:interim|intermediate|near[\s\-]?term|short[\s\-]?term)\s+"
    r"(?:target|targets|milestone|milestones|goal|goals)\b"
    r"|\bmilestone\w*\b",
    re.I,
)

# Dated target expressions are used to detect a sequence of targets.  Two or more
# distinct target dates provide evidence of an interim step plus a later end target.
DATED_TARGET = re.compile(
    r"\b(?:by|before|no later than|at the latest|by the end of|by end of)\s+"
    r"(?:fy\s*)?(20[2-9]\d)\b"
    r"|\b(?:target|goal)s?\s+(?:for|of)?\s*(?:fy\s*)?(20[2-9]\d)\b",
    re.I,
)

NAMES = {
    "target_year": "a named target year",
    "scope3": "Scope 3 (value chain) coverage",
    "offsets": "an explanation of how offsets are used",
    "milestones": "interim milestones",
}
ADVICE = {
    "target_year": "state the year by which the target is to be met",
    "scope3": "say whether Scope 3 emissions are included",
    "offsets": "explain whether offsets or removals are relied on, and how much",
    "milestones": "set interim targets on the way to the final goal",
}


def classify_target(text: str) -> str:
    """Returns one of: net_zero, reduction, none."""
    if NET_ZERO.search(text):
        return "net_zero"
    if REDUCTION.search(text):
        return "reduction"
    return "none"


def _has_interim_milestone(text: str) -> bool:
    """Returns True when text contains an explicit or multi-date interim target."""
    if EXPLICIT_MILESTONE.search(text):
        return True

    # Capture each target-associated year and ignore duplicate mentions of the same
    # year.  A single final target is not an interim milestone.
    years = set()
    for match in DATED_TARGET.finditer(text):
        years.update(group for group in match.groups() if group)
    return len(years) >= 2


def detect_signals(text: str) -> dict:
    """Returns a boolean flag for each of the four robustness signals."""
    return {
        "target_year": bool(TARGET_YEAR.search(text)),
        "scope3": bool(SCOPE3.search(text)),
        "offsets": bool(OFFSETS.search(text)),
        "milestones": _has_interim_milestone(text),
    }


def score_and_explain(text: str, points: dict) -> dict:
    """Builds the full baseline output: label, signals, score out of 100 and explanation."""
    label = classify_target(text)
    signals = detect_signals(text)
    # A pledge with no detected target cannot earn robustness points.
    score = 0 if label == "none" else sum(points[k] for k, v in signals.items() if v)
    missing = [k for k, v in signals.items() if not v]
    if label == "none":
        reason = "No emissions target was detected, so the statement cannot be graded for robustness."
    elif not missing:
        reason = "All four robustness signals were detected. Expert verification is still advised."
    else:
        parts = [f"{NAMES[k]} (to improve, {ADVICE[k]})" for k in missing]
        reason = "The pledge is missing: " + "; ".join(parts) + "."
    return {
        "method": "rules",
        "label": label,
        "signals": signals,
        "score": score,
        "explanation": reason,
    }
