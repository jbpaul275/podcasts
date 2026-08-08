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
from config import categories, load_prompt
from . import citations
from . import DuplicateEpisode, PipelineError
from . import arc as arc_mod
from .gemini import call_with_retry, client, pdf_part, record_cost, strip_fences

log = logging.getLogger("paperpod.ingest")


def ingest_pdf(path: str | Path, cfg: dict, status: str = "queued") -> str:
    """Validate and register a PDF, returning the new episode id.

    Raises DuplicateEpisode (carrying the existing id) when this content has
    already been ingested. Validation failures create a `failed` episode row so
    they stay visible in the library, and raise PipelineError.

    `status` is what an accepted paper starts as. The web upload uses "draft",
    which means validated and stored but not yet queued -- it is waiting on the
    creation wizard. The inbox watcher uses the default, because a file dropped
    in a folder has nobody standing by to answer questions about it."""
    path = Path(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    paper = db.find_paper_by_sha(sha)
    if paper:
        existing = db.episodes_for_paper(paper["id"])
        for episode in existing:
            reaccepted = _recheck_rejected(episode, paper, cfg, status)
            if reaccepted:
                return reaccepted
        if existing:
            log.info("skipping %s: duplicate of episode %s", path.name,
                     existing[0]["id"])
            raise DuplicateEpisode(existing[0]["id"])
        # A known paper with no episode of its own -- it arrived as a reference
        # in someone else's comparison. Nothing to duplicate, and no reason to
        # store the bytes twice: give it an episode on the paper already here.
        log.info("%s is already stored as a paper; giving it an episode",
                 path.name)
        return _new_episode(paper["id"], str(path), status)

    paper_id = db.create_paper(source_path=str(path), sha256=sha)
    shutil.copy2(path, db.paper_pdf(paper_id))

    error = None
    try:
        error = _validate(db.paper_pdf(paper_id), cfg)
    except Exception as e:
        error = f"could not open PDF: {e}"

    episode_id = _new_episode(paper_id, str(path), status, error=error)
    if error:
        raise PipelineError(error)
    return episode_id


def _new_episode(paper_id: str, source_path: str, status: str,
                 error: str | None = None) -> str:
    episode_id = db.new_ulid()
    paper = db.get_paper(paper_id)
    db.create_episode(
        episode_id,
        source_path=source_path,
        sha256=paper["sha256"] if paper else None,
        status="failed" if error else status,
        error=error,
        failed_at=db.now_iso() if error else None,
        papers=[paper_id],
    )
    return episode_id


def _recheck_rejected(existing, paper, cfg: dict, status: str) -> str | None:
    """Re-validate a PDF that was turned away at ingest, and accept it if the
    rules have since changed. Returns its episode id, or None to leave it be.

    A rejection is a verdict under the limits in force at the time, but it is
    stored as flat text on the episode. Raise [script] max_pages and every
    paper refused under the old ceiling keeps quoting that ceiling forever,
    because re-uploading matches on SHA and never reaches the validator again.
    The message then reads as a live decision by code that no longer exists,
    which is a genuinely misleading place to end up.

    Only episodes that never entered the pipeline qualify: an empty stage log
    is exactly what an ingest-time rejection looks like. A failure at scripting
    or TTS is a different thing entirely, and quietly restarting one of those
    from the top would re-spend real money.
    """
    if existing["status"] != "failed" or not existing["error"]:
        return None
    if db.get_stage_log(existing["id"]):
        return None
    pdf = db.paper_pdf(paper["id"])
    if not pdf.exists():
        return None

    try:
        error = _validate(pdf, cfg)
    except Exception as e:
        error = f"could not open PDF: {e}"
    if error:
        # Still refused -- but refresh the stored reason, so the page states
        # today's limit rather than one nobody can act on.
        if error != existing["error"]:
            db.update_episode(existing["id"], error=error)
        return None

    log.info("re-accepting episode %s: it passes validation under the current "
             "limits", existing["id"])
    # Re-dated, because re-uploading is a new submission. The row is reused, so
    # without this the episode keeps the timestamp from when it first bounced --
    # and created_at is what the library sorts on and what the feed sends as
    # pubDate. A paper you just added would appear wherever it sat days ago
    # instead of at the top, which reads exactly like it never arrived.
    db.update_episode(existing["id"], status=status, error=None,
                      created_at=db.now_iso())
    return existing["id"]


# How many pages to sample when deciding whether a PDF has a text layer.
# Working papers open on a cover page carrying almost nothing -- BLS, NBER and
# IZA all do this -- so looking only at page one calls a perfectly good paper a
# scan. A genuine scan has no text on any page, so a handful is plenty.
TEXT_SAMPLE_PAGES = 10
MIN_TEXT_CHARS = 500

# The API's own ceilings: 1000 pages and 50 MB per PDF. Worth checking here so
# an oversized file is refused with a sentence instead of failing later with
# whatever the API says about it.
API_MAX_PAGES = 1000
API_MAX_MB = 50


def _validate(pdf: Path, cfg: dict) -> str | None:
    """Why this PDF cannot be used, or None. Every message says what to do."""
    size_mb = pdf.stat().st_size / (1024 * 1024)
    doc = fitz.open(pdf)
    try:
        pages = doc.page_count
        sampled = [doc[i].get_text().strip() for i in range(min(pages, TEXT_SAMPLE_PAGES))]
    finally:
        doc.close()

    max_pages = min(int(cfg["script"]["max_pages"]), API_MAX_PAGES)
    if pages == 0:
        return "the PDF has no pages."
    if pages > max_pages:
        return (f"paper is {pages} pages; the limit is {max_pages}. Rejected rather "
                f"than truncated. Raise [script] max_pages if you want it — the API "
                f"itself stops at {API_MAX_PAGES}.")
    if size_mb > API_MAX_MB:
        return (f"the file is {size_mb:.0f} MB; the API accepts {API_MAX_MB} MB per "
                "PDF. Re-save it with compressed images and try again.")

    found = sum(len(t) for t in sampled)
    looked = len(sampled)
    where = (f"the first {looked} pages" if looked > 1 else "its only page")
    if found == 0:
        return (f"no text at all on {where} — this is a scan with no text layer. "
                "Run it through OCR and upload the result.")
    if found < MIN_TEXT_CHARS:
        # Some text, just not much. Saying "scan" here would send someone off to
        # OCR a document that is simply thin.
        return (f"only {found} characters of text on {where}. Too little to work "
                "from — if the body is images of text, OCR it first.")
    return None


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
    paper = db.principal_paper(episode_id)
    pdf_path = db.paper_pdf(paper["id"]) if paper else None
    if pdf_path is None or not pdf_path.exists():
        raise PipelineError(f"missing source PDF for episode {episode_id}")

    from google.genai import types

    ep = db.get_episode(episode_id)
    model = (ep["metadata_model"] if ep and ep["metadata_model"]
             else cfg["models"]["metadata"])
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

    db.update_paper(
        paper["id"],
        title=(meta.get("title") or "").strip() or None,
        authors=json.dumps([str(a) for a in authors]),
        year=year,
        abstract=(meta.get("abstract") or "").strip() or None,
        summary=(meta.get("summary") or "").strip() or None,
        categories=json.dumps(clean_categories(meta.get("categories"), cfg)),
        doi=citations.normalize_doi(meta.get("doi")),
        venue=(meta.get("venue_or_series") or "") or None,
        # Which arc this work needs. Anything unrecognised falls to empirical:
        # most uploads are papers, and that is the arc with mileage on it.
        work_kind=arc_mod.clean_kind(meta.get("kind")),
    )
    refresh_citations(episode_id, cfg)


HAND_ENTERED = "entered by hand"


def refresh_citations(episode_id: str, cfg: dict, force: bool = False) -> int | None:
    """Look up the citation count and store it, or leave it alone.

    Wrapped so a lookup can never fail an episode: an unreachable third party
    is not a reason to lose a finished podcast.

    A number entered by hand survives automatic lookups. Somebody typed it
    because the automatic route did not work, or because they preferred a
    different source, and having re-running a stage quietly undo that is how
    manual entry stops being worth doing. `force` is the explicit button.
    """
    if not (cfg.get("citations", {}) or {}).get("enabled", True):
        return None
    row = db.get_episode(episode_id)
    if not row:
        return None
    if not force and row["cited_by_source"] == HAND_ENTERED:
        log.info("episode %s has a hand-entered citation count; leaving it",
                 episode_id)
        return None
    try:
        found = citations.lookup(row["doi"], row["title"], cfg)
    except Exception:
        log.exception("citation lookup crashed for %s", episode_id)
        return None
    if not found:
        return None
    count, source = found
    db.update_principal(episode_id, cited_by=count, cited_by_source=source,
                        cited_by_at=db.now_iso())
    log.info("episode %s cited %d times per %s", episode_id, count, source)
    return count
