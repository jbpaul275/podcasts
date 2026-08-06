"""Shared Gemini client helpers: client construction, cost accounting, fence stripping."""

import os
import re

import db
from . import PipelineError

_client = None


def client():
    global _client
    if _client is None:
        if not os.environ.get("GEMINI_API_KEY"):
            raise PipelineError("GEMINI_API_KEY is not set in the environment")
        from google import genai
        _client = genai.Client()  # reads GEMINI_API_KEY
    return _client


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
