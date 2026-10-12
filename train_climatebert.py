"""Fine tunes ClimateBERT on the shared train split and stores the model locally.

Usage: python -m src.train_climatebert
"""
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src import data_processing, utils

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred) -> dict:
    """Computes accuracy and macro F1 from the trainer logits, used to select the best epoch."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average="macro")}


def to_hf_dataset(frame, tokenizer, label2id: dict, max_length: int):
    """Tokenises one split and returns a Hugging Face Dataset with a labels column."""
    from datasets import Dataset

    ds = Dataset.from_dict({"text": frame["text"].astype(str).tolist(),
                            "labels": [label2id[x] for x in frame["label"]]})
    return ds.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length), batched=True)


def main() -> None:
    """Loads the splits, fine tunes the model, and writes the model plus validation metrics."""
    # Heavy imports are placed here so that the rest of the package works without torch.
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    utils.setup_logging()
    cfg = utils.load_config()
    utils.set_seed(cfg["seed"])
    c = cfg["climatebert"]
    splits = data_processing.load_dataset(cfg)
    # Label ids follow the order of the config schema so every arm shares one mapping.
    id2label = {int(k): v for k, v in cfg["labels"].items()}
    label2id = {v: k for k, v in id2label.items()}
    tokenizer = AutoTokenizer.from_pretrained(c["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        c["base_model"], num_labels=len(id2label), id2label=id2label, label2id=label2id)
    train = to_hf_dataset(splits["train"], tokenizer, label2id, c["max_length"])
    val = to_hf_dataset(splits["validation"], tokenizer, label2id, c["max_length"])
    args = TrainingArguments(
        output_dir=c["output_dir"] + "_checkpoints", num_train_epochs=c["epochs"],
        per_device_train_batch_size=c["batch_size"], per_device_eval_batch_size=c["batch_size"],
        learning_rate=c["learning_rate"], eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", save_total_limit=1,
        seed=cfg["seed"], report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    trainer.train()
    # Stores the best model and tokenizer so the evaluation script and the app load identical weights.
    out = Path(c["output_dir"])
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    metrics = trainer.evaluate()
    Path(cfg["paths"]["predictions_dir"]).mkdir(exist_ok=True)
    (Path(cfg["paths"]["predictions_dir"]) / "climatebert_validation_metrics.json").write_text(
        json.dumps(metrics, indent=2))
    logger.info("Saved model to %s. Validation metrics: %s", out, metrics)


if __name__ == "__main__":
    main()
