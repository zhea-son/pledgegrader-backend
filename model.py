"""Model arms. Each arm exposes predict(text) returning a dict with a label key.

All three arms share the same interface. The zero shot LLM arm is defined in
src/llm_arm.py and re-exported here.
"""
import logging
from pathlib import Path

from src import rules
# The zero shot LLM arm lives in its own module because it carries the prompt and API handling.
from src.llm_arm import LLMUnavailableError, ZeroShotLLMArm

logger = logging.getLogger(__name__)


class RuleBasedArm:
    """Traditional baseline built on regular expressions."""

    name = "rules"

    def __init__(self, cfg: dict):
        self.points = cfg["scoring"]

    def predict(self, text: str) -> dict:
        """Classifies and scores one pledge."""
        return rules.score_and_explain(text, self.points)


def resolve_model_source(cfg: dict):
    """Returns (source, revision) for the fine tuned ClimateBERT weights, or None when unavailable.

    A local folder wins because it is the fastest and works offline. Otherwise the weights are
    read from the Hugging Face Hub repository named in the config, pinned to one commit so
    that the deployed app always loads exactly the weights behind the reported results.
    """
    settings = cfg["climatebert"]
    local = Path(settings["output_dir"])
    if local.exists():
        return str(local), None
    if settings.get("hub_repo"):
        return settings["hub_repo"], settings.get("hub_revision")
    return None


class ClimateBertArm:
    """Fine tuned ClimateBERT classifier for the target type.

    The transformer predicts the target type only. The four robustness signals
    still come from the shared pattern checks, so the score stays comparable
    across arms.
    """

    name = "climatebert"

    def __init__(self, cfg: dict):
        # Heavy imports are deferred so that the rule based arm works without torch.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        resolved = resolve_model_source(cfg)
        if resolved is None:
            raise FileNotFoundError(
                "No fine tuned model found. Run: python -m src.train_climatebert, or set climatebert.hub_repo in the config.")
        source, revision = resolved
        logger.info("Loading ClimateBERT from %s%s", source, f" at revision {revision[:7]}" if revision else "")
        self.torch = torch
        self.max_length = cfg["climatebert"]["max_length"]
        self.points = cfg["scoring"]
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(source, revision=revision).eval()

    def predict(self, text: str) -> dict:
        """Classifies one pledge with the transformer and scores it with the shared signal checks."""
        inputs = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
        # The softmax probability of the chosen class is reported as a percentage for the app.
        probabilities = self.torch.softmax(logits, dim=-1)[0]
        best = int(probabilities.argmax())
        label = self.model.config.id2label[best]
        signals = rules.detect_signals(text)
        score = 0 if label == "none" else sum(self.points[k] for k, v in signals.items() if v)
        return {"method": self.name, "label": label, "signals": signals, "score": score,
                "confidence": round(float(probabilities[best]) * 100, 1),
                "explanation": rules.score_and_explain(text, self.points)["explanation"]}


def get_arms(cfg: dict, names=("rules", "climatebert"), live: bool = False) -> dict:
    """Builds the requested arms. ClimateBERT is skipped, with a logged warning, when its weights cannot be loaded.

    live=True is for the web app: the LLM arm then answers or fails within seconds
    instead of waiting out rate limits like the evaluation runs do.
    """
    arms = {}
    if "rules" in names:
        arms["rules"] = RuleBasedArm(cfg)
    if "climatebert" in names and resolve_model_source(cfg) is not None:
        try:
            arms["climatebert"] = ClimateBertArm(cfg)
        except (OSError, ImportError) as exc:  # Missing network, bad repository name or torch not installed.
            logger.warning("ClimateBERT unavailable, continuing without it: %s", exc)
    if "llm" in names:
        arms["llm"] = ZeroShotLLMArm(cfg, live=live)
    return arms
