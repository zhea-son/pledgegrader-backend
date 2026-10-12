"""Signal answer key: merges completed labelling sheets and scores each method on the four signals.

The answer key is used only at marking time. The live application never reads it.
The 100 labelled sentences are drawn from the shared test set, so the key must not be
used to tune the rules or the prompts.

Usage:
    python -m src.signal_key --methods rules
    python -m src.signal_key --methods rules llm
"""
import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import rules, utils

logger = logging.getLogger(__name__)

# The four robustness signals, in the order used by the labelling sheet.
SIGNALS = ["target_year", "scope3", "offsets", "milestones"]
# Name of the rows shared by every labeller, used to measure agreement.
SHARED_MARKER = "ALL"
# Methods whose signals come from the shared pattern checks in src/rules.py.
RULE_SIGNAL_METHODS = {"rules", "climatebert"}


def load_labelled_sheets(paths: list, sheet: str = "Labelling") -> pd.DataFrame:
    """Reads completed labelling workbooks and returns one long table of complete rows.

    Two layouts are supported. A shared workbook has one tab per labeller, and the
    tab name identifies the labeller. A single labeller workbook has one tab called
    by the sheet argument, and the file name identifies the labeller. Tabs without
    the four signal columns, such as the Guide tab, are ignored. Y becomes True and
    N becomes False. Rows with any empty or invalid signal cell are dropped and counted.
    """
    frames = []
    for path in paths:
        for tab, raw in pd.read_excel(path, sheet_name=None).items():
            if not set(["id", "assigned_to", "text", *SIGNALS]) <= set(raw.columns):
                continue
            rater = Path(path).stem if tab == sheet else tab
            marks = raw[SIGNALS].apply(lambda col: col.astype(str).str.strip().str.upper())
            complete = marks.isin(["Y", "N"]).all(axis=1)
            dropped = int((~complete).sum())
            if dropped:
                logger.warning("%s / %s: %d of %d rows are incomplete and were skipped",
                               Path(path).name, tab, dropped, len(raw))
            frame = raw.loc[complete, ["id", "assigned_to", "text"]].copy()
            frame[SIGNALS] = marks.loc[complete].eq("Y")
            frame["rater"] = rater
            frame["file"] = Path(path).name
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["id", "assigned_to", "text", *SIGNALS, "rater", "file"])
    long = pd.concat(frames, ignore_index=True)
    # The same labeller appearing in two workbooks (for example a partial and a final export) would count twice.
    duplicated = long[long.duplicated(["rater", "id"], keep=False)]
    if not duplicated.empty:
        raise ValueError(
            f"Duplicate labels for {sorted(duplicated['rater'].unique())} found in {sorted(duplicated['file'].unique())}. "
            "Keep only the final export in the completed folder.")
    return long


def build_key(long: pd.DataFrame) -> pd.DataFrame:
    """Returns one answer key row per sentence using a majority vote across labellers.

    A signal is True when more than half of the labellers marked Y. A tie is
    resolved to False, which follows the labelling rule to choose N when unsure.
    """
    rows = []
    for sentence_id, group in long.groupby("id", sort=True):
        row = {"id": int(sentence_id), "text": group["text"].iloc[0], "n_raters": int(len(group))}
        for signal in SIGNALS:
            row[signal] = bool(group[signal].mean() > 0.5)
        rows.append(row)
    return pd.DataFrame(rows, columns=["id", "text", "n_raters", *SIGNALS])


def fleiss_kappa(counts: np.ndarray):
    """Returns Fleiss kappa for an items by categories matrix of rating counts.

    Every row must sum to the same number of raters. The value is None when
    chance agreement equals one, because kappa is then undefined.
    """
    counts = np.asarray(counts, dtype=float)
    n_items, _ = counts.shape
    n_raters = counts[0].sum()
    # Share of agreeing rater pairs within each item.
    p_item = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_item.mean()
    # Expected agreement by chance from the overall category proportions.
    p_cat = counts.sum(axis=0) / (n_items * n_raters)
    p_chance = (p_cat ** 2).sum()
    if np.isclose(p_chance, 1.0):
        return None
    return float((p_bar - p_chance) / (1 - p_chance))


def agreement(long: pd.DataFrame) -> dict:
    """Measures labeller agreement on the shared rows for each signal.

    Only shared rows labelled by the full set of labellers are used, so that every
    item has the same number of ratings, as Fleiss kappa requires.
    """
    shared = long[long["assigned_to"] == SHARED_MARKER]
    if shared.empty:
        return {}
    per_item = shared.groupby("id").size()
    n_raters = int(per_item.max())
    complete_ids = per_item[per_item == n_raters].index
    shared = shared[shared["id"].isin(complete_ids)]
    if n_raters < 2 or shared.empty:
        return {}
    result = {}
    for signal in SIGNALS:
        yes = shared.groupby("id")[signal].sum().to_numpy()
        counts = np.column_stack([n_raters - yes, yes])
        majority_share = np.maximum(counts[:, 0], counts[:, 1]) / n_raters
        kappa = fleiss_kappa(counts)
        result[signal] = {
            "fleiss_kappa": None if kappa is None else round(kappa, 4),
            "mean_majority_share": round(float(majority_share.mean()), 4),
            "n_items": int(len(counts)),
            "n_raters": n_raters,
        }
    return result


def signals_for_method(method: str, texts: list, predictions_dir: Path) -> pd.DataFrame:
    """Returns the predicted signals of one method for the given sentences.

    The rule based and ClimateBERT arms share the pattern checks in src/rules.py, so
    those signals are computed directly. Any other method (for example the LLM arm)
    must provide outputs/<method>_signals.csv with a text column and the four signals.
    """
    if method in RULE_SIGNAL_METHODS:
        return pd.DataFrame([rules.detect_signals(t) for t in texts], columns=SIGNALS)
    path = predictions_dir / f"{method}_signals.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is required for method '{method}' (columns: text, {', '.join(SIGNALS)})")
    table = utils.read_predictions(path).drop_duplicates("text").set_index("text")
    texts = [utils.normalise_text(t) for t in texts]
    missing = [t for t in texts if t not in table.index]
    if missing:
        raise ValueError(f"{path} lacks {len(missing)} of the {len(texts)} answer key sentences")
    return table.loc[texts, SIGNALS].astype(bool).reset_index(drop=True)


def score_signals(key: pd.DataFrame, predicted: pd.DataFrame) -> dict:
    """Returns precision, recall and F1 per signal for one method against the answer key.

    Positive means the signal is present. The macro average gives each signal equal weight.
    """
    out = {}
    for signal in SIGNALS:
        gold = key[signal].to_numpy(dtype=bool)
        pred = predicted[signal].to_numpy(dtype=bool)
        tp, fp = int((gold & pred).sum()), int((~gold & pred).sum())
        fn, tn = int((gold & ~pred).sum()), int((~gold & ~pred).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[signal] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                       "accuracy": round((tp + tn) / len(gold), 4), "support_present": tp + fn,
                       "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    out["macro_f1"] = round(float(np.mean([out[s]["f1"] for s in SIGNALS])), 4)
    return out


def main() -> None:
    """Builds the answer key, reports agreement and scores the requested methods."""
    parser = argparse.ArgumentParser(description="Build the signal answer key and score methods against it")
    parser.add_argument("--methods", nargs="+", default=["rules"], help="Methods to score")
    parser.add_argument("--glob", default=None, help="Pattern for completed labelling workbooks")
    args = parser.parse_args()
    utils.setup_logging()
    cfg = utils.load_config()
    out_dir = Path(cfg["paths"]["predictions_dir"])
    pattern = args.glob or cfg["labelling"]["completed_glob"]
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No completed labelling workbooks found for pattern: {pattern}")
    logger.info("Reading %d workbooks: %s", len(paths), [Path(p).name for p in paths])

    long = load_labelled_sheets(paths, cfg["labelling"]["sheet"])
    key = build_key(long)
    logger.info("Answer key has %d sentences", len(key))

    # Confirms that every key sentence belongs to the shared test split when the cached split exists.
    test_path = Path(cfg["dataset"]["raw_dir"]) / "test.csv"
    if test_path.exists():
        outside = ~key["text"].isin(pd.read_csv(test_path)["text"])
        if outside.any():
            logger.warning("%d answer key sentences are not in the test split", int(outside.sum()))

    key.to_csv(out_dir / "signal_key.csv", index=False)
    agree = agreement(long)
    (out_dir / "signal_agreement.json").write_text(json.dumps(agree, indent=2))
    logger.info("Agreement: %s", agree)

    for method in args.methods:
        predicted = signals_for_method(method, key["text"].tolist(), out_dir)
        metrics = score_signals(key, predicted)
        (out_dir / f"signal_metrics_{method}.json").write_text(json.dumps(metrics, indent=2))
        logger.info("%s signal macro F1: %.4f", method, metrics["macro_f1"])
        for signal in SIGNALS:
            m = metrics[signal]
            logger.info("  %-12s precision %.4f recall %.4f f1 %.4f (present in %d)", signal,
                        m["precision"], m["recall"], m["f1"], m["support_present"])


if __name__ == "__main__":
    main()
