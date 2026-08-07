"""Ingest: PDF in, validated episode row + extracted metadata out.

Two halves:
- ingest_pdf(): local, fast. Hash, dedupe, page/text validation, copy into
  data/papers/, create the episode row. Called at enqueue time.
- extract_metadata(): the 'extracting' stage. Sends the PDF natively to Gemini
  and stores strict-JSON metadata on the episode.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path

import fitz  # pymupdf

import db
from config import PAPERS_DIR, categories, load_prompt
from . import DuplicateEpisode, PipelineError
from .gemini import call_with_retry, client, pdf_part, record_cost, strip_fences

log = logging.getLogger("paperpod.ingest")


def ingest_pdf(path: str | Path, cfg: dict) -> str:
    """Validate and register a PDF, returning the new episode id.

    Raises DuplicateEpisode (carrying the existing id) when this content has
    already been ingested. Validation failures create a `failed` episode row so
    they stay visible in the library, and raise PipelineError."""
    path = Path(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    existing = db.find_by_sha(sha)
    if existing:
        log.info("skipping %s: duplicate of episode %s", path.name, existing["id"])
        raise DuplicateEpisode(existing["id"])

    episode_id = db.new_ulid()
    dest = PAPERS_DIR / f"{episode_id}.pdf"
    shutil.copy2(path, dest)

    error = None
    try:
        doc = fitz.open(dest)
        pages = doc.page_count
        first_page_text = doc[0].get_text() if pages else ""
        doc.close()
        max_pages = cfg["script"]["max_pages"]
        if pages > max_pages:
            error = f"paper is {pages} pages; limit is {max_pages}. Rejected rather than truncated."
        elif len(first_page_text.strip()) < 500:
            error = (
                "first-page text extraction yielded under 500 characters -- likely a "
                "scanned PDF with no text layer, which is out of scope for v1."
            )
    except Exception as e:
        error = f"could not open PDF: {e}"

    db.create_episode(
        episode_id,
        source_path=str(path),
        sha256=sha,
        status="failed" if error else "queued",
        error=error,
    )
    if error:
        raise PipelineError(error)
    return episode_id


def _metadata_prompt(cfg: dict) -> str:
    """The metadata prompt with the tag vocabulary spliced in, so the model can
    only choose from slugs the site actually has filters for."""
    vocab = "\n".join(f'  - "{c["slug"]}" — {c["label"]}' for c in categories(cfg))
    return load_prompt("metadata.md").replace(
        "$CATEGORIES", vocab or "  (no categories configured; return [])")


def clean_categories(values, cfg: dict) -> list[str]:
    """Keep only slugs the config knows, in config order.

    The model is asked for slugs from a list and mostly complies, but an
    invented tag would show up nowhere and silently drop the episode out of
    every filter -- so anything unrecognised is dropped here rather than stored.
    """
    if not isinstance(values, list):
        return []
    wanted = {str(v).strip().casefold() for v in values}
    return [c["slug"] for c in categories(cfg) if c["slug"].casefold() in wanted]


def extract_metadata(episode_id: str, cfg: dict) -> None:
    """Stage 'extracting': native-PDF metadata extraction with strict JSON output."""
    pdf_path = PAPERS_DIR / f"{episode_id}.pdf"
    if not pdf_path.exists():
        raise PipelineError(f"missing source PDF {pdf_path}")

    from google.genai import types

    model = cfg["models"]["metadata"]
    resp = call_with_retry(
        lambda: client().models.generate_content(
            model=model,
            contents=[pdf_part(pdf_path), _metadata_prompt(cfg)],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        ),
        cfg, model, label="metadata extraction",
    )
    record_cost(episode_id, model, resp, cfg, stage="metadata")

    raw = strip_fences(resp.text or "")
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError as e:
        raise PipelineError(f"metadata response was not valid JSON: {e}\n{raw[:500]}")

    authors = meta.get("authors") or []
    if not isinstance(authors, list):
        authors = [str(authors)]
    year = meta.get("year")
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None

    db.update_episode(
        episode_id,
        title=(meta.get("title") or "").strip() or None,
        authors=json.dumps([str(a) for a in authors]),
        year=year,
        abstract=(meta.get("abstract") or "").strip() or None,
        summary=(meta.get("summary") or "").strip() or None,
        categories=json.dumps(clean_categories(meta.get("categories"), cfg)),
        venue=(meta.get("venue_or_series") or "") or None,
    )
