"""Shared Gemini client helpers: client construction, cost accounting, fence stripping."""

import json
import logging
import os
import re
import time

import db
from . import PipelineError, QuotaUnavailable

log = logging.getLogger("paperpod.gemini")

_client = None

# Transient by nature: rate limits, and the server-side failures Gemini returns
# intermittently. Anything else (bad request, auth, unknown model) will fail the
# same way on every attempt, so retrying only wastes wall-clock.
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


def client():
    global _client
    if _client is None:
        if not os.environ.get("GEMINI_API_KEY"):
            raise PipelineError("GEMINI_API_KEY is not set in the environment")
        from google import genai
        _client = genai.Client()  # reads GEMINI_API_KEY
    return _client


def _payload(exc) -> dict:
    """The parsed error body, if the exception carries one."""
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        inner = details.get("error")
        return inner if isinstance(inner, dict) else details
    return {}


def retry_delay(exc) -> float | None:
    """Seconds the server asked us to wait, from its RetryInfo.

    This matters: Gemini answers a 429 with a specific delay (often 10-30s),
    and blind exponential backoff of 2s/4s gives up long before that window
    has passed, turning a recoverable rate limit into a failed episode.
    """
    for detail in _payload(exc).get("details") or []:
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
            m = re.match(r"([\d.]+)s?$", str(detail.get("retryDelay", "")).strip())
            if m:
                return float(m.group(1))
    m = re.search(r"retry in ([\d.]+)\s*s", str(exc))
    return float(m.group(1)) if m else None


def terminal_quota_reason(exc) -> str | None:
    """Why a 429 cannot clear by waiting inside this run, or None if it can.

    Two shapes qualify. `limit: 0` means the plan grants no allowance for this
    model at all. A *per-day* quota violation means the allowance is spent until
    the quota window rolls over — hours away, not seconds. Both are billing
    states rather than congestion, and retrying either one only burns wall-clock
    before failing anyway.
    """
    if getattr(exc, "code", None) != 429:
        return None

    payload = _payload(exc)
    try:
        blob = json.dumps(payload)
    except (TypeError, ValueError):
        blob = str(exc)
    if re.search(r"limit:\s*0\b", blob + " " + str(exc)):
        return "the plan grants no quota for this model (limit: 0)"

    for detail in payload.get("details") or []:
        if not str(detail.get("@type", "")).endswith("QuotaFailure"):
            continue
        for violation in detail.get("violations") or []:
            if "PerDay" in str(violation.get("quotaId", "")):
                cap = violation.get("quotaValue")
                return (
                    "the daily quota is exhausted"
                    + (f" ({cap} requests/day on this plan)" if cap else "")
                )
    return None


def quota_is_unavailable(exc) -> bool:
    return terminal_quota_reason(exc) is not None


def is_retryable(exc) -> bool:
    code = getattr(exc, "code", None)
    return isinstance(code, int) and code in RETRYABLE_CODES


def call_with_retry(fn, cfg: dict, model: str, label: str = "request",
                    extra_retryable: tuple = ()):
    """Run a Gemini call, retrying transient failures on the server's schedule.

    `extra_retryable` covers failures that arrive as a valid response rather
    than an HTTP error — notably TTS returning text tokens instead of audio.
    """
    rcfg = cfg.get("retry", {})
    attempts = max(1, int(rcfg.get("attempts", 4)))
    base = float(rcfg.get("base_delay_s", 2))
    max_delay = float(rcfg.get("max_delay_s", 60))

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except extra_retryable as e:
            last = e
            delay = base * (2 ** attempt)
        except Exception as e:
            last = e
            if reason := terminal_quota_reason(e):
                raise QuotaUnavailable(
                    f"{model}: {reason}. Retrying cannot help — enable billing on "
                    "the project, wait for the quota to reset, or point [models] "
                    "at a model you have allowance for."
                ) from e
            if not is_retryable(e):
                raise
            # Prefer the server's own delay; it knows when the window reopens.
            # A stated 0s is not a usable instruction — an immediate retry just
            # burns an attempt — so fall back to backoff for that too.
            delay = retry_delay(e) or base * (2 ** attempt)

        if attempt == attempts - 1:
            break
        delay = min(delay + 0.5, max_delay)  # small cushion past the stated window
        log.warning(
            "%s attempt %d/%d failed (%s); retrying in %.1fs",
            label, attempt + 1, attempts, type(last).__name__, delay,
        )
        time.sleep(delay)

    raise last if last else PipelineError(f"{label} failed for an unknown reason")


def record_cost(episode_id: str, model: str, response, cfg: dict,
                stage: str = "other") -> float:
    """Compute USD cost from usage metadata and accumulate it on the episode."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0.0
    prices = cfg.get("costs", {}).get(model)
    if not prices:
        return 0.0
    tokens_in = getattr(usage, "prompt_token_count", 0) or 0
    tokens_out = (getattr(usage, "candidates_token_count", 0) or 0) + (
        getattr(usage, "thoughts_token_count", 0) or 0
    )
    usd = (tokens_in / 1e6) * prices.get("input_per_1m", 0) + (
        tokens_out / 1e6
    ) * prices.get("output_per_1m", 0)
    if usd:
        db.add_cost(episode_id, usd, stage)
    return usd


def strip_fences(text: str) -> str:
    """Remove a surrounding markdown code fence if the model added one."""
    text = text.strip()
    m = re.match(r"^```[a-zA-Z0-9_-]*\n(.*)\n```$", text, re.DOTALL)
    return m.group(1).strip() if m else text


def pdf_part(pdf_path):
    from google.genai import types
    return types.Part.from_bytes(
        data=open(pdf_path, "rb").read(), mime_type="application/pdf"
    )
