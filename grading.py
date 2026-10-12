"""Grading orchestration: runs every available method on one pledge and builds the web app response.

The response follows the agreed JSON contract (frontend/src/sampleResult.json):

* text, score, kind and summary describe the pledge as a whole.
* checks holds the four robustness checks with evidence positions for highlighting.
* methods holds one block per method that answered, with its own label.
* warnings lists any method that could not answer, so the page can explain a missing block.

Agreed rules:

* The overall kind is the majority label of the methods that answered. A tie is broken
  in the order ClimateBERT, then the LLM, then the rule based method.
* The four checks and the score come from the rule based signals, because they are fast,
  deterministic and carry exact character positions. When the overall kind is "none",
  no check is graded and the score is 0, which matches the scoring rule used in evaluation.
"""
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from src import evidence, rules

logger = logging.getLogger(__name__)

# Input limits shared with the web page.
MIN_CHARS = 20
MAX_CHARS = 2000

METHOD_ORDER = ["rules", "climatebert", "llm"]
METHOD_NAMES = {"rules": "Rule based", "climatebert": "ClimateBERT", "llm": "Zero shot LLM"}
# Order used to break a tie between methods.
TIE_BREAK_ORDER = ["climatebert", "llm", "rules"]
LABEL_NAMES = {"net_zero": "Net zero", "reduction": "Reduction target", "none": "No target"}
CHECK_ORDER = ["target_year", "scope3", "offsets", "milestones"]
CHECK_META = {
    "target_year": {"name": "Target year", "tag": "TARGET YEAR", "found": "names a year",
                    "missing": "names no target year",
                    "tip": "A stronger pledge names the year it will reach its target, for example by 2040."},
    "scope3": {"name": "Scope 3", "tag": "SCOPE 3", "found": "covers the whole supply chain",
               "missing": "leaves out the emissions of suppliers and customers",
               "tip": "A stronger pledge covers the emissions of suppliers and customers (Scope 3), "
                      "not just its own operations."},
    "offsets": {"name": "Offsets", "tag": "OFFSETS", "found": "explains how offsets are used",
                "missing": "does not say how offsets will be used",
                "tip": "A stronger pledge says how much it relies on offsets or carbon removal, and what kind."},
    "milestones": {"name": "Interim milestones", "tag": "MILESTONE", "found": "sets an interim milestone",
                   "missing": "sets no interim milestones",
                   "tip": "A stronger pledge sets a nearer target on the way, for example a 50% cut by 2030."},
}
# Opening words of the summary by the number of checks passed.
SUPPORT_LEVELS = {0: "Not yet supported", 1: "Weakly supported", 2: "Half supported",
                  3: "Mostly supported", 4: "Fully supported"}


def validate_text(text: str) -> None:
    """Raises ValueError with a readable message when the pledge is too short or too long."""
    if len(text.strip()) < MIN_CHARS:
        raise ValueError(f"Please enter at least {MIN_CHARS} characters.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Please keep the pledge under {MAX_CHARS:,} characters.")


def format_time(seconds: float) -> str:
    """Formats a duration for display, for example 4 ms, 0.12 s or 4.8 s."""
    if seconds < 0.1:
        return f"{max(1, round(seconds * 1000))} ms"
    if seconds < 1:
        return f"{seconds:.2f} s"
    return f"{seconds:.1f} s"


def join_phrases(items: list) -> str:
    """Joins phrases as a readable list, for example a, b and c."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def decide_label(results: dict) -> str:
    """Returns the majority label of the methods, breaking a tie by the agreed method order."""
    counts = Counter(r["label"] for r in results.values())
    top = max(counts.values())
    leaders = {label for label, n in counts.items() if n == top}
    for name in TIE_BREAK_ORDER:
        if name in results and results[name]["label"] in leaders:
            return results[name]["label"]
    return sorted(leaders)[0]


def build_checks(text: str, label: str, points: dict) -> tuple:
    """Returns the four check dictionaries and the score.

    A check is found when the rule based signal is present. When the overall label is
    none, no check is graded, so every check is reported as not found and the score is 0.
    """
    signals = rules.detect_signals(text)
    spans = evidence.find_evidence(text)
    graded = label != "none"
    checks, score = [], 0
    for signal in CHECK_ORDER:
        meta = CHECK_META[signal]
        span = spans[evidence.CHECK_IDS[signal]] if graded and signals[signal] else None
        found = span is not None
        score += points[signal] if found else 0
        checks.append({"id": evidence.CHECK_IDS[signal], "name": meta["name"], "tag": meta["tag"],
                       "found": found, "evidence": span["evidence"] if found else None,
                       "start": span["start"] if found else None, "end": span["end"] if found else None,
                       "tip": None if found else meta["tip"]})
    return checks, score


def build_summary(label: str, checks: list) -> str:
    """Returns a plain language summary of what the pledge does and does not contain."""
    if label == "none":
        return "No emissions target was detected, so the statement cannot be graded for robustness."
    by_id = {c["id"]: c for c in checks}
    found = [CHECK_META[s]["found"] for s in CHECK_ORDER if by_id[evidence.CHECK_IDS[s]]["found"]]
    missing = [CHECK_META[s]["missing"] for s in CHECK_ORDER if not by_id[evidence.CHECK_IDS[s]]["found"]]
    opening = SUPPORT_LEVELS[len(found)]
    if not missing:
        return f"{opening}. It {join_phrases(found)}. A high score still needs expert verification."
    if not found:
        return f"{opening}. It {join_phrases(missing)}."
    return f"{opening}. It {join_phrases(found)}, but it {join_phrases(missing)}."


def rules_detail(text: str, label: str) -> str:
    """Explains what the rule based method matched, since it has no confidence score."""
    phrase = evidence.find_target_phrase(text, label)
    lead = f"Matched “{phrase}”." if phrase else "No target wording matched."
    return f"{lead} No confidence score, because rules either match or do not."


def method_block(name: str, result: dict, seconds: float, text: str) -> dict:
    """Builds the response block for one method from its prediction."""
    confidence = result.get("confidence")
    return {
        "id": name,
        "name": METHOD_NAMES[name],
        "time": format_time(seconds),
        "label": LABEL_NAMES[result["label"]],
        "detail": rules_detail(text, result["label"]) if name == "rules" else None,
        # Confidence is expected as a percentage from 0 to 100 and is clamped to that range.
        "confidence": None if confidence is None else max(0, min(100, int(round(confidence)))),
        "quote": (result.get("explanation") or None) if name == "llm" else None,
    }


def _run_arm(arm, text: str) -> tuple:
    """Runs one method and returns (result, seconds, error). A failure is returned, not raised."""
    start = time.perf_counter()
    try:
        return arm.predict(text), time.perf_counter() - start, None
    except Exception as exc:  # A failing method must not stop the other methods.
        return None, time.perf_counter() - start, exc


def grade_pledge(text: str, arms: dict, cfg: dict) -> dict:
    """Grades one pledge with every available method and returns the web app response.

    Args:
        text: The pledge as entered by the user.
        arms: A dictionary from method name (rules, climatebert, llm) to an object with predict(text).
        cfg: The project configuration. The scoring section supplies the points per check.

    Raises:
        ValueError: The text is shorter or longer than the allowed range.
        RuntimeError: No method could grade the pledge.
    """
    names = [n for n in METHOD_ORDER if n in arms]
    # Methods run in parallel, so the wait is the slowest method and not the sum of all methods.
    with ThreadPoolExecutor(max_workers=max(1, len(names))) as pool:
        futures = {n: pool.submit(_run_arm, arms[n], text) for n in names}
        outcomes = {n: f.result() for n, f in futures.items()}

    results, blocks, warnings = {}, [], []
    for name in names:
        result, seconds, error = outcomes[name]
        if error is not None:
            logger.error("Method %s failed: %r", name, error)
            warnings.append(f"{METHOD_NAMES[name]} is unavailable right now, so the result uses the other methods.")
            continue
        results[name] = result
        blocks.append(method_block(name, result, seconds, text))
    if not results:
        raise RuntimeError("No method could grade this pledge.")

    label = decide_label(results)
    checks, score = build_checks(text, label, cfg["scoring"])
    return {"text": text, "score": score, "kind": LABEL_NAMES[label], "summary": build_summary(label, checks),
            "checks": checks, "methods": blocks, "warnings": warnings}
