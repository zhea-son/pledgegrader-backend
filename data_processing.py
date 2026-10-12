"""Dataset loading and preparation with one shared held out test split."""
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Candidate column names, checked in order when the configured name is absent.
TEXT_CANDIDATES = ["text", "sentence", "sentences"]
LABEL_CANDIDATES = ["label", "labels", "target", "class", "category"]


def _pick_column(frame: pd.DataFrame, configured: str, candidates: list) -> str:
    """Returns the name of the matching column, or raises an error listing the real columns."""
    for name in [configured] + candidates:
        if name in frame.columns:
            return name
    raise KeyError(f"None of {[configured] + candidates} found. Actual columns: {list(frame.columns)}")


def _normalise_labels(series: pd.Series, id2label: dict) -> pd.Series:
    """Converts labels to the internal string names (none, net_zero, reduction).

    Integer labels are mapped through the config schema. String labels are
    lower cased with spaces and hyphens replaced by underscores.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).map(id2label)
    cleaned = series.astype(str).str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)
    return cleaned.replace({"no_target": "none", "no": "none", "netzero": "net_zero"})


def _standardise(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Returns a frame with exactly two columns: text and label (string names)."""
    ds = cfg["dataset"]
    text_col = _pick_column(frame, ds["text_column"], TEXT_CANDIDATES)
    label_col = _pick_column(frame, ds["label_column"], LABEL_CANDIDATES)
    id2label = {int(k): v for k, v in cfg["labels"].items()}
    out = pd.DataFrame({"text": frame[text_col], "label": _normalise_labels(frame[label_col], id2label)})
    logger.info("Columns used: text=%s, label=%s. Label counts: %s", text_col, label_col,
                out["label"].value_counts(dropna=False).to_dict())
    return out


def make_splits(full: pd.DataFrame, seed: int) -> dict:
    """Returns a stratified 70/15/15 split as a dict of train, validation and test frames.

    The first call keeps 70 percent for training. The second call halves the
    remainder into validation and test. Stratification keeps the class mix the
    same in every split, and the seed makes the result reproducible.
    """
    train, rest = train_test_split(full, test_size=0.30, stratify=full["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    return {"train": train, "validation": val, "test": test}


def load_dataset(cfg: dict) -> dict:
    """Loads the climatebert dataset and returns a dict of train, validation and test DataFrames.

    The Hugging Face `datasets` package is imported lazily so that unit tests
    run without it. The published dataset ships a single train split, so a
    stratified 70/15/15 split is created with the configured seed. Splits are
    cached as CSV so that all three methods use identical rows.
    """
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    cached = {s: raw_dir / f"{s}.csv" for s in ("train", "validation", "test")}
    if all(p.exists() for p in cached.values()):
        logger.info("Loading cached splits from %s", raw_dir)
        return {s: pd.read_csv(p) for s, p in cached.items()}

    from datasets import load_dataset as hf_load  # Lazy import, see docstring

    logger.info("Downloading %s", cfg["dataset"]["hf_name"])
    ds = hf_load(cfg["dataset"]["hf_name"])
    if {"train", "validation", "test"} <= set(ds.keys()):
        splits = {name: _standardise(ds[name].to_pandas(), cfg) for name in cached}
    else:
        # Only one split is published, so the shared split is created here.
        full = _standardise(ds["train"].to_pandas(), cfg).dropna().drop_duplicates("text")
        splits = make_splits(full, cfg["seed"])
    for name, frame in splits.items():
        frame.reset_index(drop=True).to_csv(cached[name], index=False)
        logger.info("Saved %s split: %d rows", name, len(frame))
    return {s: pd.read_csv(p) for s, p in cached.items()}


def get_test_set(cfg: dict) -> pd.DataFrame:
    """Returns the single held out test split shared by every method."""
    return load_dataset(cfg)["test"].reset_index(drop=True)
