"""Checks that the ClimateBERT weights on the Hugging Face Hub reproduce the stored results.

The script ignores any local model folder, loads the weights from the Hub repository named in
the config, predicts the shared test set and compares every label with the stored file
outputs/climatebert_predictions.csv. A full match proves that the deployed model is the
model behind the reported ClimateBERT results.

Usage:
    python -m src.verify_hub_model
"""
import copy
import logging
from pathlib import Path

from sklearn.metrics import f1_score

from src import data_processing, model, utils

logger = logging.getLogger(__name__)


def main() -> None:
    """Loads the Hub weights, predicts the test set and compares with the stored predictions."""
    utils.setup_logging()
    cfg = utils.load_config()
    if not cfg["climatebert"].get("hub_repo"):
        raise SystemExit("No climatebert.hub_repo is set in config/config.yaml.")
    # A copy of the config points the local folder at a path that cannot exist, which forces the Hub route.
    hub_cfg = copy.deepcopy(cfg)
    hub_cfg["climatebert"]["output_dir"] = "models/__force_hub_download__"
    arm = model.ClimateBertArm(hub_cfg)

    test = data_processing.get_test_set(cfg)
    preds = [arm.predict(text)["label"] for text in test["text"]]
    stored = utils.read_predictions(Path(cfg["paths"]["predictions_dir"]) / "climatebert_predictions.csv")
    if len(stored) != len(preds):
        raise SystemExit(f"Row count differs: stored {len(stored)}, Hub model {len(preds)}.")

    mismatches = [i for i, (new, old) in enumerate(zip(preds, stored["pred"])) if new != old]
    macro_f1 = f1_score(test["label"], preds, average="macro")
    logger.info("Hub model macro F1 on the test set: %.4f (n=%d)", macro_f1, len(preds))
    logger.info("Labels that differ from outputs/climatebert_predictions.csv: %d of %d", len(mismatches), len(preds))
    if mismatches:
        logger.error("Mismatch at test rows: %s", mismatches[:20])
        raise SystemExit(1)
    logger.info("PASS: the Hub weights reproduce the stored ClimateBERT predictions exactly.")


if __name__ == "__main__":
    main()
