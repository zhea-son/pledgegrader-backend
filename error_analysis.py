"""Error analysis export: where the methods are wrong, where they disagree and what to review by hand.

Reads the stored prediction files (outputs/<method>_predictions.csv with columns text, gold, pred)
and writes three files to the outputs directory:

* error_analysis.csv: every test sentence on which at least one method is wrong.
* error_summary.json: counts of agreement and error rates by sentence property.
* error_review_sample.csv: a seeded sample for manual review with blank reviewer columns.

Usage:
    python -m src.error_analysis
    python -m src.error_analysis --methods rules climatebert llm --n-review 30
"""
import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

from src import utils

logger = logging.getLogger(__name__)

# Sentences longer than this are likely to be cut at 128 tokens (about four characters per token).
LONG_CHARS = 500
# Common English words used by the rough language check.
ENGLISH_WORDS = set(
    "the of and to in a is for that on with as by are be this we our will from at or an have has its it which not all".split()
)
# Suggested categories for the manual review columns.
REVIEW_CATEGORIES = ["ambiguous_gold_label", "paraphrase_missed", "wrong_keyword_match",
                     "truncated_long_text", "non_english", "other"]
DEFAULT_METHODS = ["rules", "climatebert", "llm"]


def looks_non_english(text: str) -> bool:
    """Returns True when a sentence has very few common English words or many accented letters.

    This is a rough heuristic for reporting only. It can misjudge short or technical sentences.
    """
    words = re.findall(r"[a-zA-ZÀ-ɏ]+", text.lower())
    if len(words) < 4:
        return False
    english_share = sum(w in ENGLISH_WORDS for w in words) / len(words)
    accented = sum(1 for c in text if "À" <= c <= "ɏ") / max(len(text), 1)
    return english_share < 0.12 or accented > 0.02


def load_predictions(methods: list, out_dir: Path) -> pd.DataFrame:
    """Merges the stored prediction files into one table with a pred_<method> column per method.

    Raises an error when the files are not aligned on the same sentences and gold labels.
    """
    merged = None
    for method in methods:
        frame = utils.read_predictions(out_dir / f"{method}_predictions.csv")[["text", "gold", "pred"]]
        frame = frame.rename(columns={"pred": f"pred_{method}"})
        if merged is None:
            merged = frame
            continue
        if len(frame) != len(merged) or not (frame["text"] == merged["text"]).all() \
                or not (frame["gold"] == merged["gold"]).all():
            raise ValueError(f"The prediction file for '{method}' is not aligned with the others")
        merged[f"pred_{method}"] = frame[f"pred_{method}"]
    merged.insert(0, "row", range(len(merged)))
    return merged


def add_flags(frame: pd.DataFrame, methods: list) -> pd.DataFrame:
    """Adds correctness columns, a disagreement flag and sentence property flags."""
    out = frame.copy()
    for m in methods:
        out[f"correct_{m}"] = out[f"pred_{m}"] == out["gold"]
    pred_cols = [f"pred_{m}" for m in methods]
    out["methods_disagree"] = out[pred_cols].nunique(axis=1) > 1
    out["n_chars"] = out["text"].str.len()
    out["long_text"] = out["n_chars"] > LONG_CHARS
    out["has_line_break"] = out["text"].str.contains("\n")
    out["non_english"] = out["text"].map(looks_non_english)
    return out


def summarise(frame: pd.DataFrame, methods: list) -> dict:
    """Returns counts of errors, agreement and error rates by sentence property."""
    correct = frame[[f"correct_{m}" for m in methods]]
    summary = {
        "n_rows": int(len(frame)),
        "methods": methods,
        "all_correct": int(correct.all(axis=1).sum()),
        "all_wrong": int((~correct).all(axis=1).sum()),
        "methods_disagree": int(frame["methods_disagree"].sum()),
        "per_method": {},
    }
    for m in methods:
        wrong = ~frame[f"correct_{m}"]
        others = [o for o in methods if o != m]
        # An error is unique when every other method is right on the same sentence.
        unique = wrong & frame[[f"correct_{o}" for o in others]].all(axis=1) if others else wrong
        by_flag = {}
        for flag in ("long_text", "has_line_break", "non_english"):
            on, off = frame[flag], ~frame[flag]
            by_flag[flag] = {
                "rows_with_flag": int(on.sum()),
                "error_rate_with_flag": round(float(wrong[on].mean()), 4) if on.any() else None,
                "error_rate_without_flag": round(float(wrong[off].mean()), 4) if off.any() else None,
            }
        summary["per_method"][m] = {"errors": int(wrong.sum()), "error_rate": round(float(wrong.mean()), 4),
                                    "unique_errors": int(unique.sum()), "by_flag": by_flag}
    return summary


def review_sample(frame: pd.DataFrame, methods: list, n: int, seed: int) -> pd.DataFrame:
    """Returns a seeded sample of sentences for manual review with blank reviewer columns.

    With two or more methods the pool is the sentences on which the methods disagree.
    With one method the pool is the sentences that method gets wrong.
    """
    if len(methods) >= 2:
        pool = frame[frame["methods_disagree"]]
    else:
        pool = frame[~frame[f"correct_{methods[0]}"]]
    sample = pool.sample(n=min(n, len(pool)), random_state=seed).sort_values("row")
    keep = ["row", "text", "gold", *[f"pred_{m}" for m in methods], "long_text", "has_line_break", "non_english"]
    sample = sample[keep].copy()
    sample["reviewer_category"] = ""
    sample["reviewer_notes"] = ""
    return sample


def main() -> None:
    """Builds the error table, summary and review sample from the stored predictions."""
    parser = argparse.ArgumentParser(description="Error analysis export")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to compare")
    parser.add_argument("--n-review", type=int, default=30, help="Size of the manual review sample")
    args = parser.parse_args()
    utils.setup_logging()
    cfg = utils.load_config()
    out_dir = Path(cfg["paths"]["predictions_dir"])
    methods = args.methods or [m for m in DEFAULT_METHODS if (out_dir / f"{m}_predictions.csv").exists()]
    if not methods:
        raise SystemExit("No prediction files found in " + str(out_dir))
    logger.info("Comparing methods: %s", methods)

    frame = add_flags(load_predictions(methods, out_dir), methods)
    any_wrong = ~frame[[f"correct_{m}" for m in methods]].all(axis=1)
    frame[any_wrong].to_csv(out_dir / "error_analysis.csv", index=False)
    summary = summarise(frame, methods)
    (out_dir / "error_summary.json").write_text(json.dumps(summary, indent=2))
    review_sample(frame, methods, args.n_review, cfg["seed"]).to_csv(out_dir / "error_review_sample.csv", index=False)
    logger.info("Rows with at least one error: %d of %d", int(any_wrong.sum()), len(frame))
    logger.info("Summary: %s", {k: v for k, v in summary.items() if k != "per_method"})
    for m, stats in summary["per_method"].items():
        logger.info("%s: %d errors (rate %.4f), %d unique", m, stats["errors"], stats["error_rate"], stats["unique_errors"])


if __name__ == "__main__":
    main()
