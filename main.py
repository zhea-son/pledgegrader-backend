"""Command line entry point: evaluates one method on the shared test set.

Usage: python -m src.main --method rules   (or climatebert, llm)
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src import data_processing, evaluation, model, utils

logger = logging.getLogger(__name__)


def evaluate(method: str, cfg: dict) -> dict:
    """Runs one method over the test set, stores predictions, and returns metrics."""
    test = data_processing.get_test_set(cfg)
    arm = model.get_arms(cfg, (method,))[method]
    # Labels are already standardised to string names by data_processing.
    gold = test["label"]
    outputs = [arm.predict(t) for t in test["text"]]
    preds = [o["label"] for o in outputs]
    # The class order comes from the config schema so every method uses one layout.
    labels = list(cfg["labels"].values())
    metrics = {"method": method, "n": len(test), **evaluation.compute_metrics(gold, preds, labels)}
    out = Path(cfg["paths"]["predictions_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # Stores row level predictions so the error review and bootstrap test can reuse them.
    pd.DataFrame({"text": test["text"], "gold": gold, "pred": preds}).to_csv(out / f"{method}_predictions.csv", index=False)
    # Stores the signals, score and explanation per sentence. src/signal_key.py reads this file for the LLM arm.
    signal_cols = ["target_year", "scope3", "offsets", "milestones"]
    pd.DataFrame([{"text": t, **{k: o["signals"][k] for k in signal_cols}, "score": o["score"],
                   "explanation": o["explanation"]} for t, o in zip(test["text"], outputs)]
                 ).to_csv(out / f"{method}_signals.csv", index=False)
    (out / f"{method}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf8")
    # Stores the confusion matrix as CSV so it can be opened directly or plotted for the report.
    evaluation.confusion_frame(metrics).to_csv(out / f"{method}_confusion_matrix.csv")
    return metrics


def main() -> None:
    """Parses arguments, evaluates the chosen method and logs the metrics."""
    parser = argparse.ArgumentParser(description="PledgeGrader evaluation")
    parser.add_argument("--method", default="rules", choices=["rules", "climatebert", "llm"])
    args = parser.parse_args()
    utils.setup_logging()
    cfg = utils.load_config()
    utils.set_seed(cfg["seed"])
    try:
        metrics = evaluate(args.method, cfg)
        logger.info("Overall: accuracy %.4f, macro F1 %.4f (n=%d)", metrics["accuracy"], metrics["macro_f1"], metrics["n"])
        for label, m in metrics["per_class"].items():
            logger.info("Class %-10s precision %.4f recall %.4f f1 %.4f support %d",
                        label, m["precision"], m["recall"], m["f1"], m["support"])
    except model.LLMUnavailableError as exc:  # Quota or outage: answers so far are cached.
        logger.error("%s. Answers so far are cached, so the next run resumes where this one stopped.", exc)
        raise SystemExit(1)
    except Exception:  # Logs the full traceback, then exits with a readable message.
        logger.exception("Evaluation failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
