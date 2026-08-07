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

# Ceiling on a single API call. Without one, a stalled connection blocks the
# worker thread forever: the episode sits in "synthesizing" with no error and
# nothing behind it in the queue ever starts. A TTS chunk is ~250 words and
# normally answers in well under a minute, so ten minutes is generous.
DEFAULT_TIMEOUT_S = 600
_timeout_s = DEFAULT_TIMEOUT_S


def configure(cfg: dict) -> None:
    """Apply config before anything builds the client. Called once at startup."""
    global _timeout_s
    if _client is not None:
        return  # already built; the timeout is baked into the transport
    _timeout_s = float(cfg.get("retry", {}).get(
        "request_timeout_s", DEFAULT_TIMEOUT_S))

# Transient by nature: rate limits, and the server-side failures Gemini returns
# intermittently. Anything else (bad request, auth, unknown model) will fail the
# same way on every attempt, so retrying only wastes wall-clock.
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}

# Warn once per model rather than on every chunk.
_UNPRICED_WARNED: set[str] = set()


def client():
    global _client
    if _client is None:
        if not os.environ.get("GEMINI_API_KEY"):
            raise PipelineError("GEMINI_API_KEY is not set in the environment")
        from google import genai
        from google.genai import types
        _client = genai.Client(  # reads GEMINI_API_KEY
            http_options=types.HttpOptions(timeout=int(_timeout_s * 1000)),
        )
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


def quota_is_unavailable(exc) -> bool:
    """A 429 carrying `limit: 0` means the plan grants no allowance for this
    model at all. That is a billing state, not congestion, so no amount of
    waiting will clear it."""
    if getattr(exc, "code", None) != 429:
        return False
    try:
        blob = json.dumps(_payload(exc))
    except (TypeError, ValueError):
        blob = str(exc)
    return bool(re.search(r"limit:\s*0\b", blob + " " + str(exc)))


def is_timeout(exc) -> bool:
    """A request that ran past its deadline, by any name the transport uses."""
    return "timeout" in type(exc).__name__.casefold()


def is_retryable(exc) -> bool:
    # A timeout is the most transient failure there is -- one stalled
    # connection should cost a retry, not the whole chunk.
    if is_timeout(exc):
        return True
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
            if quota_is_unavailable(e):
                raise QuotaUnavailable(
                    f"{model} has no quota on this API key's plan (limit: 0). "
                    "Retrying cannot help — enable billing on the project, or "
                    "point [models] in config.toml at a model you have access to."
                ) from e
            if not is_retryable(e):
                raise
            # Prefer the server's own delay; it knows when the window reopens.
            delay = retry_delay(e)
            if delay is None:
                delay = base * (2 ** attempt)

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
        # Silently costing an unpriced model at zero is worse than useless when
        # the point of running two models is comparing what they cost.
        if model not in _UNPRICED_WARNED:
            _UNPRICED_WARNED.add(model)
            log.warning(
                "no [costs.%r] entry in config.toml — this model's spend will be "
                "reported as $0.00", model,
            )
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
