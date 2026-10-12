"""Development runs for the zero shot LLM arm. These never touch the test split.

Commands:
    python -m src.llm_experiments validate [--limit 50]
        Runs the arm on the validation split (or a seeded sample of it) to check
        the prompt. Stores predictions and metrics in outputs/.
    python -m src.llm_experiments consistency [--n 50 --runs 3]
        Sends the same validation sentences several times, ignoring the cache,
        and measures how often the answers agree. Stores outputs/llm_consistency.json.

The final test set run uses the shared entry point: python -m src.main --method llm
If a run stops on the free tier daily limit, run the same command again later:
cached answers are reused, so it resumes where it stopped.
"""
import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd

from src import data_processing, evaluation, llm_arm, utils

logger = logging.getLogger(__name__)


def _sample(frame: pd.DataFrame, n, seed: int) -> pd.DataFrame:
    """Returns a seeded sample of n rows, or every row when n is None."""
    if n is None or n >= len(frame):
        return frame.reset_index(drop=True)
    return frame.sample(n=n, random_state=seed).reset_index(drop=True)


def _run_info(arm: llm_arm.ZeroShotLLMArm) -> dict:
    """Records the settings and usage that the report needs for reproducibility and cost."""
    calls = arm.stats["api_calls"]
    return {"model": arm.model_name, "prompt_version": llm_arm.PROMPT_VERSION,
            "temperature": arm.temperature, "run_date": dt.date.today().isoformat(), **arm.stats,
            "avg_input_tokens_per_call": round(arm.stats["input_tokens"] / calls, 1) if calls else None,
            "avg_output_tokens_per_call": round(arm.stats["output_tokens"] / calls, 1) if calls else None}


def validate(cfg: dict, limit) -> dict:
    """Runs the arm on validation sentences and stores predictions and metrics."""
    val = _sample(data_processing.load_dataset(cfg)["validation"], limit, cfg["seed"])
    arm = llm_arm.ZeroShotLLMArm(cfg)
    outputs = [arm.predict(t) for t in val["text"]]
    preds = [o["label"] for o in outputs]
    labels = list(cfg["labels"].values())
    metrics = {"split": "validation", "n": len(val), **evaluation.compute_metrics(val["label"], preds, labels),
               "run": _run_info(arm)}
    out = Path(cfg["paths"]["predictions_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # Wrong answers are listed first so prompt problems are easy to review.
    table = pd.DataFrame([{"text": t, "gold": g, "pred": o["label"], **o["signals"], "score": o["score"],
                           "explanation": o["explanation"]} for t, g, o in zip(val["text"], val["label"], outputs)])
    table["correct"] = table["gold"] == table["pred"]
    table.sort_values("correct").to_csv(out / "llm_validation_predictions.csv", index=False)
    (out / "llm_validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf8")
    return metrics


def consistency(cfg: dict, n: int, runs: int) -> dict:
    """Repeats the same validation sentences and measures agreement across runs."""
    val = _sample(data_processing.load_dataset(cfg)["validation"], n, cfg["seed"])
    arm = llm_arm.ZeroShotLLMArm(cfg)
    # Each run is a fresh call per sentence; the cache is bypassed on purpose.
    answers = [[arm.predict(t, use_cache=False) for t in val["text"]] for _ in range(runs)]
    per_sentence = list(zip(*answers))
    result = {"n_sentences": len(val), "runs": runs,
              "label_agreement": round(sum(len({a["label"] for a in s}) == 1 for s in per_sentence) / len(val), 4),
              "score_agreement": round(sum(len({a["score"] for a in s}) == 1 for s in per_sentence) / len(val), 4),
              "signal_agreement": {k: round(sum(len({a["signals"][k] for a in s}) == 1 for s in per_sentence)
                                            / len(val), 4) for k in llm_arm.SIGNALS},
              "run": _run_info(arm)}
    out = Path(cfg["paths"]["predictions_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "llm_consistency.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    return result


def main() -> None:
    """Parses the command and runs it, with a readable message if Gemini is unavailable."""
    parser = argparse.ArgumentParser(description="Zero shot LLM development runs (validation split only)")
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="Run the arm on the validation split")
    v.add_argument("--limit", type=int, default=None, help="Seeded sample size; default is all 516 sentences")
    c = sub.add_parser("consistency", help="Repeat sentences and measure agreement")
    c.add_argument("--n", type=int, default=50)
    c.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    utils.setup_logging()
    cfg = utils.load_config()
    utils.set_seed(cfg["seed"])
    try:
        if args.command == "validate":
            m = validate(cfg, args.limit)
            logger.info("Validation (n=%d): accuracy %.4f, macro F1 %.4f", m["n"], m["accuracy"], m["macro_f1"])
            logger.info("Run details: %s", m["run"])
        else:
            r = consistency(cfg, args.n, args.runs)
            logger.info("Consistency: %s", json.dumps(r, indent=2))
    except llm_arm.LLMUnavailableError as exc:
        logger.error("%s. Answers so far are cached; run the same command again to resume.", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
