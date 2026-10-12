"""Zero shot LLM arm: Google Gemini with written instructions only.

The model receives no labelled examples and no training. It returns the target
type, the four robustness signals with a supporting quote for each, and a plain
language explanation. The output follows the shared predict(text) format used
by the rule based and ClimateBERT arms.

Design choices:
- Gemini must return JSON that matches a fixed schema, so parsing is reliable.
- Temperature is 0 to keep answers as repeatable as possible.
- Every signal marked present must quote the input. A quote that is not found
  in the input is treated as a hallucination and the signal is set to False.
- The score is calculated here with the shared points in config.yaml, never by
  the LLM, so all three arms are scored by the same rule.
- Every answer is cached on disk, so a sentence is never sent twice and an
  interrupted run resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

try:
    # Loads GEMINI_API_KEY from a local .env file. Hugging Face Spaces sets it as a secret instead.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency
    pass

logger = logging.getLogger(__name__)
# Hides one log line per HTTP request from the client libraries; errors are still logged here.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

SIGNALS = ("target_year", "scope3", "offsets", "milestones")

# Changes whenever the prompt changes, so answers cached under an older prompt are not reused.
PROMPT_VERSION = "v1"

SYSTEM_INSTRUCTION = """\
You are an analyst who reads sentences from corporate climate disclosures.
You classify the text and check it for four credibility signals.
Judge only from the text given. Do not use outside knowledge about the company.
The text may contain line breaks, be very long, or not be in English. Classify it anyway.

TASK 1. Classify the text into exactly one label:
- "net_zero": the text states a target or commitment to reach net zero,
  carbon neutrality, climate neutrality or zero emissions overall.
- "reduction": the text states a target to cut emissions by an amount or
  percentage, without a net zero goal.
- "none": the text contains no emission target of either kind. General
  statements about climate, strategy or past performance are "none".
If a text has both a net zero target and a reduction target, choose "net_zero".

TASK 2. Check four signals. Mark a signal present only if the text itself states it.
- target_year: a specific year by which the target is reached (for example 2040).
- scope3: the target explicitly covers Scope 3, or value chain, supply chain,
  supplier or customer emissions.
- offsets: the text explains how offsets, credits or carbon removals are used,
  limited or excluded.
- milestones: there is an interim target at a nearer date on the way to the
  final goal (for example a 2030 target in a 2050 net zero pledge).
For every signal you mark present, copy the exact words from the text that show it
into its evidence field, character for character. If you cannot quote the text,
mark the signal absent and leave the evidence empty.

TASK 3. Write a one or two sentence explanation in plain English for a
non-expert reader: what kind of promise this is and which signals are missing.
"""


class _Signal(BaseModel):
    """One signal answer with its supporting quote."""

    present: bool
    evidence: str = Field(default="", description="Exact quote from the text, or empty.")


class _LLMAnswer(BaseModel):
    """The JSON schema that Gemini must follow."""

    label: Literal["net_zero", "reduction", "none"]
    target_year: _Signal
    scope3: _Signal
    offsets: _Signal
    milestones: _Signal
    explanation: str


class LLMUnavailableError(RuntimeError):
    """Raised when Gemini cannot return a valid answer, so callers can show a clear message."""


def _normalise(text: str) -> str:
    """Lowercases and collapses whitespace so quote matching ignores line breaks."""
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_text(quote: str, text: str) -> bool:
    """Returns True when the quote really appears in the input text."""
    q = _normalise(quote).strip(" .,;:\"'")
    return bool(q) and q in _normalise(text)


def find_quote_span(quote: str, text: str):
    """Returns the quote as it appears in the text with its character positions, or None.

    Matching ignores case and differences in spaces or line breaks, so a quote copied
    across a line break is still found. The web page uses the positions to highlight it.
    """
    words = _normalise(quote).strip(" .,;:\"'").split()
    if not words:
        return None
    match = re.search(r"\s+".join(re.escape(w) for w in words), text, re.I)
    return {"evidence": match.group(0), "start": match.start(), "end": match.end()} if match else None


def retry_delay(exc: Exception, attempt: int) -> float:
    """Returns how long to wait before the next attempt.

    Uses the delay Google sends with a rate limit error (for example "retry in 55s").
    Otherwise waits 5, 10, 20, 40 then 60 seconds, which suits temporary overload errors.
    """
    match = re.search(r"retry in ([\d.]+)s|retryDelay'?:\s*'([\d.]+)s", str(exc))
    if match:
        return float(match.group(1) or match.group(2)) + 1.0
    return float(min(5 * 2 ** (attempt - 1), 60))


class ZeroShotLLMArm:
    """Zero shot Gemini arm exposing the shared predict(text) interface."""

    name = "llm"

    def __init__(self, cfg: dict, client: Any = None, live: bool = False):
        """Sets up the Gemini client, the answer cache and the request spacing.

        Args:
            cfg: The full project config. The llm and scoring sections are used.
            client: An existing genai.Client. Tests pass a fake client here.
            live: True for the web app. Requests are not spaced out and are retried
                only briefly, so a user waits seconds rather than minutes. When Gemini
                is busy or the quota is used up, LLMUnavailableError is raised quickly
                and the app shows the other two methods.
        """
        llm = cfg["llm"]
        self.points = cfg["scoring"]
        self.model_name = llm["model"]
        self.temperature = llm.get("temperature", 0.0)
        self.max_chars = llm.get("max_chars", 8000)
        self.live = live
        if live:
            self.max_retries = llm.get("live_max_retries", 2)
            self.min_interval = 0.0
            self.max_wait = llm.get("live_max_wait_seconds", 5)
        else:
            self.max_retries = llm.get("max_retries", 6)
            self.min_interval = llm.get("min_seconds_between_calls", 4.0)
            self.max_wait = llm.get("max_wait_seconds", 300)
        self.cache_path = Path(llm.get("cache_path", "outputs/llm_cache.json"))

        # Builds the real client only when none is injected.
        if client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise LLMUnavailableError("GEMINI_API_KEY is not set. Add it to the .env file.")
            from google import genai  # Imported lazily so other arms work without the package.

            from google.genai import types

            # A request that hangs is stopped, so a page never waits indefinitely.
            timeout_ms = int(llm.get("live_timeout_seconds" if live else "timeout_seconds", 20 if live else 60) * 1000)
            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))
        self.client = client

        self._lock = threading.Lock()
        self._last_call = 0.0
        # Running totals reported in the cost and hallucination notes.
        self.stats = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "hallucinated_quotes": 0}
        self._cache = self._read_cache()

    # ----- cache ---------------------------------------------------------------

    def _read_cache(self) -> dict:
        """Loads cached answers from disk, or starts with an empty cache."""
        try:
            return json.loads(self.cache_path.read_text(encoding="utf8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            logger.warning("Cache file %s is unreadable and is ignored", self.cache_path)
            return {}

    def _write_cache(self) -> None:
        """Saves the cache through a temporary file so an interrupted run cannot corrupt it."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf8")
        tmp.replace(self.cache_path)

    def _cache_key(self, text: str) -> str:
        """Builds a cache key from the model, the prompt version and the text."""
        return hashlib.sha256(f"{self.model_name}|{PROMPT_VERSION}|{text}".encode("utf8")).hexdigest()

    # ----- API call ------------------------------------------------------------

    def _wait_turn(self) -> None:
        """Spaces requests to stay under the free tier requests per minute."""
        gap = time.monotonic() - self._last_call
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last_call = time.monotonic()

    def _call_gemini(self, text: str) -> _LLMAnswer:
        """Sends the text to Gemini and returns a schema checked answer.

        Rate limits, server errors and malformed JSON are retried with growing
        waits. A rejected request such as a bad key fails at once.
        """
        from google.genai import errors, types

        gen_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=_LLMAnswer,
            # No tools are used, so automatic function calling is switched off.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        last_error = None
        wait = 0.0
        for attempt in range(1, self.max_retries + 1):
            self._wait_turn()
            try:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=f"TEXT:\n{text}", config=gen_config)
                self.stats["api_calls"] += 1
                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    self.stats["input_tokens"] += usage.prompt_token_count or 0
                    self.stats["output_tokens"] += usage.candidates_token_count or 0
                return _LLMAnswer.model_validate_json(response.text)
            except errors.APIError as exc:
                last_error = exc
                code = getattr(exc, "code", None)
                # Client errors other than 429 (for example a bad key) will not fix themselves.
                if code is not None and 400 <= code < 500 and code != 429:
                    raise LLMUnavailableError(f"Gemini rejected the request: {exc}") from exc
                wait = retry_delay(exc, attempt)
                # A wait of more than a few minutes means the daily quota is used up, so stop now.
                if wait > self.max_wait:
                    if self.live:
                        raise LLMUnavailableError("Gemini is busy or its free quota is used up") from exc
                    raise LLMUnavailableError(
                        f"Daily free tier quota reached (Gemini asked to wait {wait / 3600:.1f} h). "
                        "Run the same command again after the quota resets") from exc
                logger.warning("Gemini error %s on attempt %d of %d; waiting %.0f s", code, attempt,
                               self.max_retries, wait)
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = exc
                wait = retry_delay(exc, attempt)
                logger.warning("Malformed answer on attempt %d: %s", attempt, exc)
            if attempt < self.max_retries:
                time.sleep(wait)
        raise LLMUnavailableError(f"Gemini failed after {self.max_retries} attempts: {last_error}")

    # ----- shared interface ----------------------------------------------------

    def predict(self, text: str, use_cache: bool = True) -> dict:
        """Grades one pledge and returns the shared prediction format.

        Returns a dict with method, label, signals (four booleans), score and
        explanation, plus evidence (one quote per present signal) and
        evidence_spans (the quote's position in the text, for highlighting).
        Raises ValueError for empty text and LLMUnavailableError when Gemini
        cannot answer. use_cache=False forces a fresh call for consistency checks.
        """
        if not text or not text.strip():
            raise ValueError("Text is empty.")
        # Very long pasted texts are cut to keep requests small. No test sentence is long enough to be cut.
        if len(text) > self.max_chars:
            logger.info("Text of %d characters cut to %d", len(text), self.max_chars)
            text = text[: self.max_chars]

        key = self._cache_key(text)
        if use_cache:
            with self._lock:
                if key in self._cache:
                    return self._with_spans(self._cache[key], text)

        answer = self._call_gemini(text)
        # Shows progress on long runs, because each new sentence takes several seconds.
        if self.stats["api_calls"] % 10 == 0:
            logger.info("Progress: %d new sentences sent to Gemini so far", self.stats["api_calls"])

        # Keeps a signal only when its quote is found in the input.
        signals, evidence = {}, {}
        for name in SIGNALS:
            sig = getattr(answer, name)
            kept = sig.present and quote_in_text(sig.evidence, text)
            if sig.present and not kept:
                self.stats["hallucinated_quotes"] += 1
                logger.warning("Signal %s dropped: quote not found in text (%r)", name, sig.evidence)
            signals[name] = kept
            evidence[name] = sig.evidence if kept else ""

        # Same scoring rule as the other arms: no points when no target was found.
        score = 0 if answer.label == "none" else sum(self.points[k] for k, v in signals.items() if v)
        result = {"method": self.name, "label": answer.label, "signals": signals, "score": score,
                  "explanation": answer.explanation.strip(), "evidence": evidence}
        if use_cache:
            with self._lock:
                self._cache[key] = result
                self._write_cache()
        return self._with_spans(result, text)

    @staticmethod
    def _with_spans(result: dict, text: str) -> dict:
        """Adds the position of each evidence quote. Positions are not cached, so older cache entries work too."""
        out = dict(result)
        out["evidence_spans"] = {k: find_quote_span(q, text) if q else None
                                 for k, q in result.get("evidence", {}).items()}
        return out
