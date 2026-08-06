"""Paperpod: FastAPI app, background worker, inbox watcher.

Concurrency is deliberately trivial: one worker thread draining an in-process
queue. No Celery, no Redis, no websockets. The UI polls.
"""

import logging
import queue
import re
import shutil
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape as xml_escape

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import auth
import db
from config import (
    CHUNKS_DIR,
    FINAL_DIR,
    INBOX_DIR,
    PAPERS_DIR,
    PROCESSED_DIR,
    ROOT,
    load_config,
)
from pipeline import PipelineError, ingest, run, script as script_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("paperpod")

CFG = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    _requeue_interrupted()
    threading.Thread(target=_worker, daemon=True, name="paperpod-worker").start()
    threading.Thread(target=_watch_inbox, daemon=True, name="paperpod-watcher").start()
    log.info("paperpod up; base_url=%s", CFG["server"]["base_url"])
    if auth.admin_password() is None:
        log.warning(
            "PAPERPOD_ADMIN_PASSWORD is not set — the admin surface is reachable "
            "from localhost only. Set it before exposing this to a network."
        )
    yield


app = FastAPI(title="Paperpod", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))


def static_url(name: str) -> str:
    """Cache-bust CSS and JS by mtime. Without this the browser happily serves
    a stale app.js after an edit, which looks exactly like a broken feature."""
    try:
        stamp = int((ROOT / "static" / name).stat().st_mtime)
    except OSError:
        stamp = 0
    return f"/static/{name}?v={stamp}"


templates.env.globals["static_url"] = static_url

# Bump when the terms text changes materially.
TERMS_UPDATED = "6 August 2026"

WORK_Q: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
WORKER_STATE = {"alive": False, "current": None, "last_beat": None}


# --------------------------------------------------------------------------
# worker + watcher
# --------------------------------------------------------------------------

def _worker() -> None:
    db.init_db()
    WORKER_STATE["alive"] = True
    while True:
        WORKER_STATE["last_beat"] = db.now_iso()
        try:
            episode_id, from_stage = WORK_Q.get(timeout=5)
        except queue.Empty:
            continue
        WORKER_STATE["current"] = episode_id
        try:
            run.run_episode(episode_id, CFG, from_stage=from_stage)
        except Exception:
            log.exception("worker crashed on episode %s", episode_id)
        finally:
            WORKER_STATE["current"] = None
            WORK_Q.task_done()


def _watch_inbox() -> None:
    """Poll data/inbox/ for new PDFs. Debounce 2s on size stability so
    partially-written files are not picked up. Originals are moved to
    inbox/processed/, never mutated or deleted."""
    db.init_db()
    pending: dict[Path, tuple[int, float]] = {}
    while True:
        time.sleep(1.0)
        try:
            candidates = [
                p for p in INBOX_DIR.glob("*.pdf")
                if p.is_file() and PROCESSED_DIR not in p.parents
            ]
        except OSError:
            continue
        seen = set()
        for path in candidates:
            seen.add(path)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            prev = pending.get(path)
            if prev is None or prev[0] != size:
                pending[path] = (size, time.monotonic())
                continue
            if time.monotonic() - prev[1] < 2.0:
                continue
            pending.pop(path, None)
            try:
                _enqueue_path(path, move_to_processed=True)
            except PipelineError as e:
                log.error("inbox rejected %s: %s", path.name, e)
                _move_to_processed(path)
            except Exception:
                log.exception("inbox failed on %s", path.name)
        for gone in [p for p in pending if p not in seen]:
            pending.pop(gone, None)


def _move_to_processed(path: Path) -> None:
    dest = PROCESSED_DIR / path.name
    if dest.exists():
        dest = PROCESSED_DIR / f"{path.stem}-{int(time.time())}{path.suffix}"
    try:
        shutil.move(str(path), str(dest))
    except OSError:
        log.warning("could not move %s out of inbox", path.name)


def _enqueue_path(path: Path, move_to_processed: bool = False) -> str | None:
    episode_id = ingest.ingest_pdf(path, CFG)
    if move_to_processed:
        _move_to_processed(path)
    if episode_id:
        log.info("queued episode %s from %s", episode_id, path.name)
        WORK_Q.put((episode_id, None))
    return episode_id


def _requeue_interrupted() -> None:
    """A process killed mid-run leaves episodes in a working status. Re-enqueue
    them; run.py restarts at the stage that was in flight, and TTS skips chunks
    already on disk."""
    for row in db.list_episodes():
        if row["status"] in ("queued", *run.STAGE_NAMES):
            log.info("resuming interrupted episode %s from %s", row["id"], row["status"])
            WORK_Q.put((row["id"], None))


# --------------------------------------------------------------------------
# view helpers
# --------------------------------------------------------------------------

def _fmt_duration(seconds) -> str:
    if not seconds:
        return "--:--"
    total = int(round(float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


_SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in",
                "nor", "of", "on", "or", "the", "to", "via", "with"}

# Acronyms that must survive title-casing an all-caps string. Without these,
# "NBER WORKING PAPER SERIES" reads back as "Nber Working Paper Series".
_ACRONYMS = {
    "NBER", "IZA", "CEPR", "SSRN", "IMF", "OECD", "ECB", "BLS", "BEA", "IRS",
    "FDA", "EPA", "CDC", "WHO", "UN", "EU", "US", "USA", "UK", "MIT", "UCLA",
    "NYU", "LSE", "AER", "QJE", "JPE", "JEL", "PNAS", "GDP", "GNP", "CPI",
    "RCT", "RCTS", "OLS", "IV", "GMM", "AI", "ML", "LLM", "COVID", "HIV",
    "CEO", "CFO", "GPS", "STEM", "PISA", "NAEP", "SAT", "GED", "K-12",
}


def _decaps(text: str) -> str:
    """NBER and many journals print titles in all caps. Left alone they shout
    on the page, so title-case them — but only when they really are all caps,
    so deliberate capitalization (RCT, GDP) in mixed-case titles survives."""
    if not text or not text.isupper():
        return text
    out = []
    for i, raw in enumerate(text.split()):
        if raw.strip(".,:;()").upper() in _ACRONYMS:
            out.append(raw)          # already correctly capitalized
            continue
        w = raw.lower()
        out.append(w if (w in _SMALL_WORDS and i) else w[:1].upper() + w[1:])
    return " ".join(out)


def _author_credit(authors: list[str], max_named: int = 3) -> str:
    if not authors:
        return "an uncredited author"
    if len(authors) > max_named:
        return f"{authors[0]} et al."
    if len(authors) == 1:
        return authors[0]
    return ", ".join(authors[:-1]) + " and " + authors[-1]


def _attribution(paper_title: str | None, authors: list[str]) -> str:
    title = _decaps((paper_title or "").strip()) or "an untitled paper"
    credit = _author_credit(authors)
    stop = "" if credit.endswith(".") else "."   # "et al." already ends a sentence
    return f"This is an AI generated podcast drawing from “{title}” by {credit}{stop}"


_COST_STAGE_LABELS = {
    "metadata": "Metadata extraction",
    "script": "Script generation",
    "title": "Episode title",
    "tts": "Speech synthesis",
    "other": "Other",
}


def _cost_rows(row) -> list[dict]:
    """Per-stage spend, largest first, with each stage's share of the total.
    Speech synthesis normally dominates by an order of magnitude."""
    breakdown = db.cost_breakdown(row)
    total = sum(breakdown.values())
    if not total:
        return []
    return [
        {
            "stage": stage,
            "label": _COST_STAGE_LABELS.get(stage, stage),
            "usd": usd,
            "pct": round(100 * usd / total),
        }
        for stage, usd in sorted(breakdown.items(), key=lambda kv: -kv[1])
    ]


_paper_text_cache: dict[str, str] = {}


def _paper_text(episode_id: str) -> str | None:
    """Full text of the source PDF, cached. Used to tell a real fabricated
    citation from a name the paper genuinely uses."""
    if episode_id in _paper_text_cache:
        return _paper_text_cache[episode_id] or None
    path = PAPERS_DIR / f"{episode_id}.pdf"
    if not path.exists():
        return None
    try:
        import fitz

        with fitz.open(path) as doc:
            text = "\n".join(page.get_text() for page in doc)
    except Exception:
        log.warning("could not extract text from %s for flag checking", path)
        text = ""
    _paper_text_cache[episode_id] = text
    return text or None


def _blurb(row, limit: int = 260) -> str:
    """The listing summary. Prefers the model's written teaser; falls back to
    the first sentences of the abstract for episodes made before summaries
    existed."""
    summary = (row["summary"] or "").strip()
    if summary:
        return summary

    abstract = " ".join((row["abstract"] or "").split())
    if not abstract:
        return ""
    sentences = re.findall(r"[^.!?]+[.!?]+(?:\s|$)", abstract) or [abstract]
    out = ""
    for sentence in sentences[:2]:
        if out and len(out) + len(sentence) > limit:
            break
        out += sentence
    out = out.strip() or abstract
    return out if len(out) <= limit else out[:limit].rsplit(" ", 1)[0] + "…"


def _episode_view(row) -> dict:
    flags = (
        script_mod.citation_flags(row["script_md"], _paper_text(row["id"]))
        if row["script_md"]
        else []
    )
    # The count that matters is the one needing a human: strings that do not
    # trace back to the paper. Everything else is dimmed, not hidden.
    unverified = [f for f in flags if not f.get("in_paper")]
    authors = db.episode_authors(row)
    paper_title = _decaps((row["title"] or "").strip())
    return {
        "id": row["id"],
        # The episode's own title, with the paper's as fallback until the
        # scripting stage has written one.
        "title": row["episode_title"] or paper_title or "(untitled)",
        "paper_title": paper_title,
        "attribution": _attribution(row["title"], authors),
        "summary": _blurb(row),
        "authors": authors,
        "year": row["year"],
        "abstract": row["abstract"],
        "venue": _decaps((row["venue"] or "").strip()) or None,
        "status": row["status"],
        "error": row["error"],
        "created_at": row["created_at"],
        "duration_s": row["duration_s"],
        "duration": _fmt_duration(row["duration_s"]),
        "cost_usd": row["cost_usd"] or 0.0,
        "cost_breakdown": _cost_rows(row),
        "published": bool(row["published"]),
        "flags_reviewed": bool(row["flags_reviewed"]),
        "flag_count": len(unverified),
        "flags": flags,
        "flags_unverified": unverified,
        "flags_in_paper": [f for f in flags if f.get("in_paper")],
        "has_audio": bool(row["audio_path"] and Path(row["audio_path"]).exists()),
    }


def _script_lines(script_md: str, flags: list[dict]) -> list[dict]:
    by_line: dict[int, list[dict]] = {}
    for f in flags:
        by_line.setdefault(f["line"], []).append(f)
    out = []
    for i, raw in enumerate(script_md.splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        speaker, _, text = raw.partition(":")
        out.append({
            "speaker": speaker.strip(),
            "text": text.strip(),
            "flags": by_line.get(i, []),
        })
    return out


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

def require_admin(request: Request) -> None:
    if not auth.is_admin(request):
        raise HTTPException(401, "admin only")


def unverified_flag_count(row) -> int:
    if not row["script_md"]:
        return 0
    flags = script_mod.citation_flags(row["script_md"], _paper_text(row["id"]))
    return sum(1 for f in flags if not f.get("in_paper"))


def publish_blocker(row) -> str | None:
    """Why this episode cannot go public yet, or None if it can."""
    if row["status"] != "done":
        return f"episode is {row['status']}, not done"
    if not row["audio_path"] or not Path(row["audio_path"]).exists():
        return "no audio file on disk"
    if unverified_flag_count(row) and not row["flags_reviewed"]:
        return "citation flags not reviewed"
    return None


def _short_error(text: str | None, limit: int = 150) -> str:
    """Errors can carry a whole ffmpeg stderr dump. The library gets a
    readable first line; the episode page keeps the full text."""
    if not text:
        return ""
    first = " ".join(text.split())
    first = first.split(". ")[0]
    return first if len(first) <= limit else first[:limit].rsplit(" ", 1)[0] + "…"


@app.get("/admin/login", response_class=HTMLResponse)
def login_form(request: Request, error: str = ""):
    return templates.TemplateResponse(
        request, "login.html",
        {"error": error, "no_password": auth.admin_password() is None,
         "admin": False, "signed_in": False},
    )


@app.post("/admin/login")
def login(request: Request, password: str = Form("")):
    if not auth.password_matches(password):
        return RedirectResponse("/admin/login?error=1", status_code=303)
    resp = RedirectResponse("/admin", status_code=303)
    resp.set_cookie(
        auth.COOKIE, auth.make_token(),
        max_age=auth.SESSION_TTL, httponly=True, samesite="lax",
        secure=request.url.scheme == "https",
    )
    return resp


@app.post("/admin/logout")
def logout():
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie(auth.COOKIE)
    return resp


@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request):
    if not auth.is_admin(request):
        return RedirectResponse("/admin/login", status_code=303)
    return _render_library(request, admin_mode=True)


@app.post("/episode/{episode_id}/publish")
def set_published(request: Request, episode_id: str,
                  published: str = Form("1"), reviewed: str = Form("")):
    require_admin(request)
    row = db.get_episode(episode_id)
    if not row:
        raise HTTPException(404, "no such episode")

    if reviewed:
        db.update_episode(episode_id, flags_reviewed=1)
        row = db.get_episode(episode_id)

    want = published == "1"
    if want:
        blocker = publish_blocker(row)
        if blocker:
            raise HTTPException(400, f"cannot publish: {blocker}")
    db.update_episode(episode_id, published=1 if want else 0)
    return RedirectResponse(f"/episode/{episode_id}", status_code=303)


@app.get("/terms", response_class=HTMLResponse)
def terms(request: Request):
    site = CFG.get("site", {})
    return templates.TemplateResponse(
        request,
        "terms.html",
        {
            "admin": False,
            "signed_in": auth.is_admin(request),
            "feed_title": CFG.get("feed", {}).get("title", "Paperpod"),
            "owner_name": site.get("owner_name", "the site operator"),
            "contact_email": site.get("contact_email", ""),
            "updated": TERMS_UPDATED,
        },
    )


@app.get("/", response_class=HTMLResponse)
def library(request: Request):
    return _render_library(request, admin_mode=False)


def _render_library(request: Request, admin_mode: bool):
    all_episodes = [
        _episode_view(r) for r in db.list_episodes(published_only=not admin_mode)
    ]
    # Failures are moved out of the reading list: they are maintenance, not
    # something to browse. They stay reachable, collapsed, below the fold.
    episodes = [e for e in all_episodes if e["status"] != "failed"]
    failed = ([{**e, "short_error": _short_error(e["error"])}
               for e in all_episodes if e["status"] == "failed"]
              if admin_mode else [])
    return templates.TemplateResponse(
        request,
        "library.html",
        {
            "episodes": episodes,
            "failed": failed,
            "admin": admin_mode,
            "signed_in": auth.is_admin(request),
            "total_cost": sum(e["cost_usd"] for e in all_episodes),
            "feed_url": CFG["server"]["base_url"].rstrip("/") + "/feed.xml",
            "feed_title": CFG.get("feed", {}).get("title", "Paperpod"),
            "feed_description": CFG.get("feed", {}).get("description", ""),
        },
    )


@app.post("/upload")
async def upload(request: Request, file: UploadFile):
    require_admin(request)
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf files are accepted")
    staged = INBOX_DIR / f"upload-{db.new_ulid()}.pdf"
    with open(staged, "wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        episode_id = _enqueue_path(staged, move_to_processed=True)
    except PipelineError as e:
        _move_to_processed(staged)
        raise HTTPException(400, str(e))
    if episode_id is None:
        return RedirectResponse("/?dup=1", status_code=303)
    return RedirectResponse(f"/episode/{episode_id}", status_code=303)


@app.get("/episode/{episode_id}", response_class=HTMLResponse)
def episode_page(request: Request, episode_id: str):
    row = db.get_episode(episode_id)
    if not row:
        raise HTTPException(404, "no such episode")
    admin_mode = auth.is_admin(request)
    # An unpublished episode does not exist as far as the public is concerned.
    if not admin_mode and not (row["published"] and row["status"] == "done"):
        raise HTTPException(404, "no such episode")

    view = _episode_view(row)
    return templates.TemplateResponse(
        request,
        "episode.html",
        {
            "ep": view,
            "admin": admin_mode,
            "signed_in": admin_mode,
            "lines": _script_lines(row["script_md"], view["flags"]) if row["script_md"] else [],
            "stages": db.get_stage_log(episode_id) if admin_mode else [],
            "retry_stages": run.STAGE_NAMES,
            "publish_blocker": publish_blocker(row) if admin_mode else None,
        },
    )


@app.get("/episode/{episode_id}/audio")
def episode_audio(episode_id: str, request: Request):
    row = db.get_episode(episode_id)
    if not row or not row["audio_path"]:
        raise HTTPException(404, "no audio for this episode")
    if not auth.is_admin(request) and not row["published"]:
        raise HTTPException(404, "no audio for this episode")
    path = Path(row["audio_path"])
    if not path.exists():
        raise HTTPException(404, "audio file is missing from disk")
    return _ranged_file(path, request, "audio/mpeg")


def _ranged_file(path: Path, request: Request, media_type: str) -> Response:
    """HTTP range support, required for seeking in browsers and podcast apps."""
    size = path.stat().st_size
    range_header = request.headers.get("range")
    base_headers = {
        "accept-ranges": "bytes",
        "content-type": media_type,
    }
    if not range_header:
        return FileResponse(path, media_type=media_type, headers=base_headers)

    m = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not m:
        return FileResponse(path, media_type=media_type, headers=base_headers)
    start_s, end_s = m.groups()
    if start_s:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    else:  # suffix range: last N bytes
        if not end_s:
            return FileResponse(path, media_type=media_type, headers=base_headers)
        start = max(0, size - int(end_s))
        end = size - 1
    end = min(end, size - 1)
    if start >= size or start > end:
        return Response(
            status_code=416, headers={"content-range": f"bytes */{size}", **base_headers}
        )

    length = end - start + 1

    def stream():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                block = f.read(min(64 * 1024, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block

    return StreamingResponse(
        stream(),
        status_code=206,
        headers={
            **base_headers,
            "content-range": f"bytes {start}-{end}/{size}",
            "content-length": str(length),
        },
    )


@app.post("/episode/{episode_id}/retry")
def retry(request: Request, episode_id: str, stage: str = Form("extracting")):
    require_admin(request)
    row = db.get_episode(episode_id)
    if not row:
        raise HTTPException(404, "no such episode")
    if stage not in run.STAGE_NAMES:
        raise HTTPException(400, f"unknown stage {stage!r}")
    db.update_episode(episode_id, status="queued", error=None)
    WORK_Q.put((episode_id, stage))
    return RedirectResponse(f"/episode/{episode_id}", status_code=303)


@app.delete("/episode/{episode_id}")
def delete_episode(request: Request, episode_id: str):
    require_admin(request)
    row = db.get_episode(episode_id)
    if not row:
        raise HTTPException(404, "no such episode")
    (PAPERS_DIR / f"{episode_id}.pdf").unlink(missing_ok=True)
    (FINAL_DIR / f"{episode_id}.mp3").unlink(missing_ok=True)
    shutil.rmtree(CHUNKS_DIR / episode_id, ignore_errors=True)
    db.delete_episode(episode_id)
    return JSONResponse({"deleted": episode_id})


@app.get("/health")
def health(request: Request):
    require_admin(request)
    return {
        "queue_depth": WORK_Q.qsize(),
        "worker_alive": WORKER_STATE["alive"],
        "worker_current": WORKER_STATE["current"],
        "worker_last_beat": WORKER_STATE["last_beat"],
        "episodes": {
            "total": len(db.list_episodes()),
            "done": sum(1 for r in db.list_episodes() if r["status"] == "done"),
            "failed": sum(1 for r in db.list_episodes() if r["status"] == "failed"),
        },
    }


@app.get("/feed.xml")
def feed():
    base = CFG["server"]["base_url"].rstrip("/")
    fcfg = CFG.get("feed", {})
    scfg = CFG.get("site", {})
    items = []
    for row in db.list_episodes(published_only=True):
        if not row["audio_path"]:
            continue
        path = Path(row["audio_path"])
        if not path.exists():
            continue
        authors = db.episode_authors(row)
        title = row["episode_title"] or _decaps((row["title"] or "").strip()) or "(untitled)"
        # The disclosure leads the description so it is visible in every
        # podcast client, not just on the web page.
        summary = _attribution(row["title"], authors)
        blurb = _blurb(row)
        if blurb:
            summary += "\n\n" + blurb
        try:
            pub = datetime.fromisoformat(row["created_at"])
        except ValueError:
            pub = datetime.now(timezone.utc)
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        items.append(f"""    <item>
      <title>{xml_escape(title)}</title>
      <description>{xml_escape(summary)}</description>
      <itunes:summary>{xml_escape(summary)}</itunes:summary>
      <itunes:author>{xml_escape(', '.join(authors) or fcfg.get('author', 'Paperpod'))}</itunes:author>
      <itunes:duration>{_fmt_duration(row['duration_s'])}</itunes:duration>
      <guid isPermaLink="false">{row['id']}</guid>
      <pubDate>{format_datetime(pub)}</pubDate>
      <link>{base}/episode/{row['id']}</link>
      <enclosure url="{base}/episode/{quote(row['id'])}/audio" length="{path.stat().st_size}" type="audio/mpeg"/>
    </item>""")

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>{xml_escape(fcfg.get('title', 'Paperpod'))}</title>
    <link>{base}/</link>
    <atom:link href="{base}/feed.xml" rel="self" type="application/rss+xml"/>
    <description>{xml_escape(fcfg.get('description', ''))}</description>
    <language>en-us</language>
    <itunes:author>{xml_escape(fcfg.get('author', 'Paperpod'))}</itunes:author>
    <itunes:summary>{xml_escape(fcfg.get('description', ''))}</itunes:summary>
    <itunes:owner>
      <itunes:name>{xml_escape(scfg.get('owner_name', ''))}</itunes:name>
      <itunes:email>{xml_escape(scfg.get('contact_email', ''))}</itunes:email>
    </itunes:owner>
    <itunes:explicit>false</itunes:explicit>
    <itunes:image href="{base}/static/cover.png"/>
    <itunes:category text="Science"/>
{chr(10).join(items)}
  </channel>
</rss>
"""
    return Response(content=xml, media_type="application/rss+xml")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=CFG["server"]["port"])
