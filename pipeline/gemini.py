"""Shared Gemini client helpers: client construction, cost accounting, fence stripping."""

import json
import logging
import os
import re
import threading
import time

import db
from . import ModelRetired, PipelineError, QuotaUnavailable

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


class _Throttle:
    """A rate limit is a fact about the account, not about one request.

    Without this, each call discovers a closed quota window on its own: chunk 3
    exhausts its retries, chunk 4 starts with a fresh budget and no idea the
    API just said "slow down", and sprints straight back into the same wall.
    That is how one 429 turns into a contiguous tail of failed chunks.

    Process-global on purpose. Both workers draw on one account, so a limit hit
    by one episode applies to the other.
    """

    def __init__(self) -> None:
        self._until = 0.0
        self._lock = threading.Lock()
        # The thread that got the 429 backs off on its own schedule in the
        # retry loop below. Without this it would wait twice for one window:
        # once in its backoff and again here.
        self._own = threading.local()

    def hold(self, seconds: float) -> None:
        """Close the window for at least this long."""
        if seconds <= 0:
            return
        with self._lock:
            self._until = max(self._until, time.monotonic() + seconds)
            self._own.until = self._until

    def remaining(self) -> float:
        with self._lock:
            if getattr(self._own, "until", 0.0) >= self._until:
                return 0.0     # this thread is already serving the sentence
            return max(0.0, self._until - time.monotonic())

    def wait(self, label: str = "request") -> None:
        """Block until the window reopens. Waiting here is the point: the
        alternative is spending a retry to be told the same thing."""
        left = self.remaining()
        if left <= 0:
            return
        log.info("%s waiting %.1fs for the rate-limit window to reopen",
                 label, left)
        time.sleep(left)

    def reset(self) -> None:
        with self._lock:
            self._until = 0.0
            self._own.until = 0.0


THROTTLE = _Throttle()


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


def quota_violation(exc) -> dict | None:
    """Which quota a 429 actually hit, from its QuotaFailure detail.

    The prose in a 429 is the same sentence every time -- "You exceeded your
    current quota, please check your plan and billing details" -- and it is
    followed by two long documentation URLs. None of that distinguishes a
    per-minute limit you should wait out from a per-day one that will not move
    until tomorrow, or from a free-tier cap that means the key's Cloud project
    has no billing account. The structured detail does, and it is what gets
    truncated away when the message is stored.
    """
    if getattr(exc, "code", None) != 429:
        return None
    for detail in _payload(exc).get("details") or []:
        if not isinstance(detail, dict):
            continue
        if not str(detail.get("@type", "")).endswith("QuotaFailure"):
            continue
        for v in detail.get("violations") or []:
            if not isinstance(v, dict) or not (v.get("quotaId") or v.get("quotaMetric")):
                continue
            dims = v.get("quotaDimensions") if isinstance(v.get("quotaDimensions"), dict) else {}
            return {
                "id": str(v.get("quotaId") or ""),
                "metric": str(v.get("quotaMetric") or ""),
                "value": str(v.get("quotaValue") or ""),
                "model": str(dims.get("model") or ""),
            }
    return None


def quota_is_daily(v: dict | None) -> bool:
    """Google's quota ids spell the window out -- GenerateRequestsPerDayPer...
    versus ...PerMinute... -- so the distinction is readable rather than
    guessed at."""
    return bool(v) and "perday" in v["id"].casefold()


def quota_is_free_tier(v: dict | None) -> bool:
    return bool(v) and "freetier" in (v["id"] + v["metric"]).casefold()


def quota_summary(v: dict) -> str:
    """One line naming the limit, for a page rather than a log."""
    window = "per day" if quota_is_daily(v) else "per minute" if "perminute" in v["id"].casefold() else ""
    limit = f"limit {v['value']}" if v["value"] else "limit unknown"
    bits = [b for b in [v["id"] or v["metric"], limit, window] if b]
    return ", ".join(bits)


def describe(exc) -> str:
    """A short reason fit to store on an episode.

    Raw 429 text is ~400 characters of boilerplate and URLs, so it gets cut off
    exactly where the useful part lives. This keeps the part that says what to
    do about it.
    """
    v = quota_violation(exc)
    if v:
        return f"{type(exc).__name__}: 429 quota exhausted — {quota_summary(v)}"
    return f"{type(exc).__name__}: {exc}"


def model_is_gone(exc) -> bool:
    """A 404 for the model itself: retired, renamed, or never existed.

    Distinct from a 404 on anything else, and worth naming, because it is a
    fact about the config rather than about this request -- every retry and
    every remaining chunk will fail identically, and the only useful response
    is to try a different model.
    """
    if getattr(exc, "code", None) != 404:
        return False
    blob = (str(exc) + " " + json.dumps(_payload(exc), default=str)).casefold()
    return "model" in blob


def is_timeout(exc) -> bool:
    """A request that ran past its deadline, by any name the transport uses."""
    return "timeout" in type(exc).__name__.casefold()


# Substrings that mark a connection that broke rather than a request that was
# refused. Matched against the exception's type name and message because the
# SDK surfaces these from several transport layers (httpx, httpcore, ssl,
# socket) with no common base class and no `code`.
_TRANSPORT_MARKERS = (
    "connection", "connect", "disconnected", "protocol", "ssl", "handshake",
    "reset by peer", "broken pipe", "incomplete read", "socket", "eof occurred",
)


def is_transport_error(exc) -> bool:
    """The connection failed before the server had its say.

    These carry no HTTP status, so a check that keys on `code` calls them
    permanent and burns the chunk on the first attempt -- the exact opposite of
    the truth, since a dropped connection is the most transient failure there
    is. Requires the absence of a code: once the server answered, its status is
    the authority and this must not second-guess it.

    Retrying does risk paying twice for a call whose response was lost in
    flight. That is already true of timeouts, and a chunk that has to be
    re-synthesized later costs the same either way.
    """
    if getattr(exc, "code", None) is not None:
        return False
    blob = (type(exc).__name__ + " " + str(exc)).casefold()
    return any(m in blob for m in _TRANSPORT_MARKERS)


def is_retryable(exc) -> bool:
    # A timeout is the most transient failure there is -- one stalled
    # connection should cost a retry, not the whole chunk.
    if is_timeout(exc) or is_transport_error(exc):
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
        # Somebody else's 429 is this call's problem too -- one account, one
        # quota. Waiting costs the same wall-clock as the failure would, and
        # keeps the attempt.
        THROTTLE.wait(label)
        try:
            return fn()
        except extra_retryable as e:
            last = e
            delay = base * (2 ** attempt)
        except Exception as e:
            last = e
            if model_is_gone(e):
                raise ModelRetired(
                    f"{model} no longer exists at the API. Providers retire "
                    "preview models on their own schedule, so this can start "
                    "failing without anything here changing. Check /admin/models "
                    "for the current IDs and update [models] in config.toml "
                    "(or the PAPERPOD_MODEL_* environment override)."
                ) from e
            if quota_is_unavailable(e):
                raise QuotaUnavailable(
                    f"{model} has no quota on this API key's plan (limit: 0). "
                    "Retrying cannot help — enable billing on the project, or "
                    "point [models] in config.toml at a model you have access to."
                ) from e
            violation = quota_violation(e)
            if quota_is_daily(violation):
                # A daily allowance resets on Google's clock, not on ours.
                # Backing off through it burns half an hour per episode to
                # arrive at the same 429, and the retry window the server
                # names is about the minute, not the day.
                raise QuotaUnavailable(
                    f"{model}: the daily quota is used up ({quota_summary(violation)}). "
                    "Requests-per-day quotas reset at midnight Pacific, so retrying "
                    "before then cannot help. Note that failed requests count "
                    "against the day's allowance too — retrying into an exhausted "
                    "quota spends more of it. "
                    + ("That quota is a free-tier one. Limits are per Cloud project, "
                       "not per API key, so this means the project this key belongs "
                       "to has no billing attached — having budget on a different "
                       "project does not raise it, and minting a new key in the "
                       "same project changes nothing. "
                       if quota_is_free_tier(violation) else
                       "A paid project still caps preview models well below the "
                       "headline numbers, and budget remaining is a separate thing "
                       "from rate limit. ")
                    + "Chunks already synthesized are kept, so a retry once it "
                      "resets only pays for what is missing."
                ) from e
            if not is_retryable(e):
                raise
            # Prefer the server's own delay; it knows when the window reopens.
            delay = retry_delay(e)
            if delay is None:
                delay = base * (2 ** attempt)
            elif getattr(e, "code", None) == 429:
                # The server named a window. Hold every other call off until it
                # reopens, so the rest of the script does not spend its retries
                # finding out the same thing one at a time.
                THROTTLE.hold(min(delay, max_delay))

        if attempt == attempts - 1:
            break
        delay = min(delay + 0.5, max_delay)  # small cushion past the stated window
        log.warning(
            "%s attempt %d/%d failed (%s); retrying in %.1fs",
            label, attempt + 1, attempts, type(last).__name__, delay,
        )
        time.sleep(delay)

    raise last if last else PipelineError(f"{label} failed for an unknown reason")


def resolved_model(response, requested: str) -> str:
    """Which model actually served the request.

    [models] names aliases like gemini-pro-latest, and Google repoints those
    without telling anyone. The response says what really ran, so pricing and
    the UI can both stop guessing.
    """
    return (getattr(response, "model_version", None) or "").strip() or requested


def record_cost(episode_id: str, model: str, response, cfg: dict,
                stage: str = "other") -> float:
    """Compute USD cost from usage metadata and accumulate it on the episode."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0.0
    costs = cfg.get("costs", {})
    # Price what ran, not what was asked for. An alias repointed at a
    # differently-priced model is exactly the case a static table cannot see.
    actual = resolved_model(response, model)
    prices = costs.get(actual) or costs.get(model)
    if not prices:
        # Silently costing an unpriced model at zero is worse than useless when
        # the point of running two models is comparing what they cost.
        key = f"{model}->{actual}" if actual != model else model
        if key not in _UNPRICED_WARNED:
            _UNPRICED_WARNED.add(key)
            if actual != model:
                log.warning(
                    "%s now resolves to %s, which has no [costs] entry — its "
                    "spend will read as $0.00. The alias was repointed; add a "
                    "price for the new model.", model, actual,
                )
            else:
                log.warning(
                    "no [costs.%r] entry in config.toml — this model's spend "
                    "will be reported as $0.00", model,
                )
        return 0.0
    if actual != model and costs.get(actual) is None:
        # Priced off the alias because the resolved name is not in the table.
        # Works, but the number is only right while the alias has not moved.
        if model not in _UNPRICED_WARNED:
            _UNPRICED_WARNED.add(model)
            log.info("pricing %s as %s; add a [costs.%r] entry to price it "
                     "directly", actual, model, actual)
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
