"""Shared helpers: configuration loading, logging setup and seeding."""
import logging
import random
from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Reads the YAML configuration file and returns it as a dictionary."""
    with open(path, "r", encoding="utf8") as handle:
        return yaml.safe_load(handle)


def setup_logging(level: int = logging.INFO) -> None:
    """Configures one consistent log format for every module."""
    logging.basicConfig(
        level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )


def set_seed(seed: int) -> None:
    """Fixes the Python random seed so that runs are reproducible."""
    random.seed(seed)


def normalise_text(text: str) -> str:
    """Returns the text with Windows and old Mac line endings replaced by a single newline.

    Git on Windows can convert line endings inside quoted CSV fields, so the same
    sentence may differ between files only by the line ending characters.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_predictions(path: Path) -> pd.DataFrame:
    """Reads a prediction or signal CSV file and normalises line endings in the text column."""
    frame = pd.read_csv(path)
    frame["text"] = frame["text"].map(normalise_text)
    return frame
