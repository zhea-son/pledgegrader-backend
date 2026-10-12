"""Paired bootstrap significance test for the macro F1 difference between two methods.

Usage: python -m src.significance --a rules --b climatebert
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src import utils

logger = logging.getLogger(__name__)


def paired_bootstrap(gold, pred_a, pred_b, n_resamples: int, seed: int) -> dict:
    """Resamples the test rows with replacement and returns the macro F1 difference (B minus A).

    The same resampled rows are used for both methods, which makes the test
    paired. The p value is the two sided share of resamples where the sign of
    the difference flips relative to the observed difference.
    """
    gold, pred_a, pred_b = map(np.asarray, (gold, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(gold)
    observed = f1_score(gold, pred_b, average="macro") - f1_score(gold, pred_a, average="macro")
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        diffs[i] = f1_score(gold[idx], pred_b[idx], average="macro") - f1_score(gold[idx], pred_a[idx], average="macro")
    flipped = np.mean(diffs <= 0) if observed > 0 else np.mean(diffs >= 0)
    return {"observed_diff": float(observed), "ci95_low": float(np.percentile(diffs, 2.5)),
            "ci95_high": float(np.percentile(diffs, 97.5)), "p_value": float(min(1.0, 2 * flipped))}


def main() -> None:
    """Loads two stored prediction files, runs the test and stores the result as JSON."""
    parser = argparse.ArgumentParser(description="Paired bootstrap on macro F1")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    args = parser.parse_args()
    utils.setup_logging()
    cfg = utils.load_config()
    out = Path(cfg["paths"]["predictions_dir"])
    a, b = (utils.read_predictions(out / f"{m}_predictions.csv") for m in (args.a, args.b))
    if not (a["text"] == b["text"]).all():
        raise SystemExit("Prediction files are not aligned on the same test rows.")
    result = paired_bootstrap(a["gold"], a["pred"], b["pred"], cfg["bootstrap"]["n_resamples"], cfg["seed"])
    result.update({"method_a": args.a, "method_b": args.b})
    (out / f"bootstrap_{args.a}_vs_{args.b}.json").write_text(json.dumps(result, indent=2))
    logger.info("Bootstrap result: %s", result)


if __name__ == "__main__":
    main()
