"""Summarises the shared train, validation and test splits and checks their integrity.

The summary is stored as JSON so that the report data section can quote exact figures.
Usage: python -m src.split_summary
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src import utils

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")


def summarise_split(frame: pd.DataFrame) -> dict:
    """Returns size, class counts, class shares and text length statistics for one split."""
    counts = frame["label"].value_counts()
    lengths = frame["text"].astype(str).str.len()
    return {
        "n": int(len(frame)),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "class_share": {k: round(float(v) / len(frame), 4) for k, v in counts.items()},
        "duplicate_texts": int(frame["text"].duplicated().sum()),
        "chars_median": int(lengths.median()),
        "chars_p95": int(lengths.quantile(0.95)),
        "chars_max": int(lengths.max()),
        "rows_with_line_breaks": int(frame["text"].astype(str).str.contains("\n").sum()),
    }


def overlap_between(frames: dict) -> dict:
    """Counts identical sentences that appear in more than one split (should be zero)."""
    texts = {name: set(frame["text"]) for name, frame in frames.items()}
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    return {f"{a}_{b}": len(texts[a] & texts[b]) for a, b in pairs}


def main() -> None:
    """Reads the cached splits, builds the summary and writes it to the outputs folder."""
    parser = argparse.ArgumentParser(description="Summarise the shared dataset splits")
    parser.add_argument("--output", default=None, help="Optional output file path")
    args = parser.parse_args()
    utils.setup_logging()
    cfg = utils.load_config()
    data_dir = Path(cfg["dataset"]["raw_dir"])
    try:
        frames = {name: pd.read_csv(data_dir / f"{name}.csv") for name in SPLITS}
    except FileNotFoundError:
        logger.error("Split files not found in %s. Run: python -m src.main --method rules", data_dir)
        raise SystemExit(1)
    summary = {
        "seed": cfg["seed"],
        "total_rows": int(sum(len(f) for f in frames.values())),
        "splits": {name: summarise_split(frame) for name, frame in frames.items()},
        "overlap_between_splits": overlap_between(frames),
    }
    out = Path(args.output) if args.output else Path(cfg["paths"]["predictions_dir"]) / "split_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
