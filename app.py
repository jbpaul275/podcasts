"""Paperpod: FastAPI app, background worker, inbox watcher.

Concurrency is deliberately trivial: a small pool of worker threads draining
one in-process queue. No Celery, no Redis, no websockets. The UI polls.
"""

import json
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
from urllib.parse import quote, urlparse
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
from starlette.exceptions import HTTPException as StarletteHTTPException

import auth
import db
from config import (
    CHUNKS_DIR,
    FINAL_DIR,
    INBOX_DIR,
    PAPERS_DIR,
    PROCESSED_DIR,
    ROOT,
    categories,
    category_labels,
    is_prompt_name,
    load_config,
    prompt_default,
    prompt_names,
    prompt_override,
    prompt_warnings,
    reset_prompt,
    save_prompt,
)
from pipeline import DuplicateEpisode, PipelineError, gemini, ingest, run, script as script_mod

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
    gemini.configure(CFG)
    _requeue_interrupted()
    for i in range(worker_count()):
        name = f"paperpod-worker-{i + 1}"
        threading.Thread(target=_worker, args=(name,), daemon=True,
                         name=name).start()
    threading.Thread(target=_watch_inbox, daemon=True, name="paperpod-watcher").start()
    log.info("paperpod up; base_url=%s workers=%d",
             CFG["server"]["base_url"], worker_count())
    unpriced = [m for m in tts_choices() if m not in CFG.get("costs", {})]
    if unpriced:
        log.warning(
            "TTS models offered with no [costs] entry, so their spend will read "
            "as $0.00: %s", ", ".join(unpriced),
        )
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


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    """Render 404s as a page for anyone who arrived by clicking a link.

    A deleted episode leaves stale links behind -- bookmarks, the browser's
    back button, a URL someone shared -- and answering those with a raw JSON
    body reads as a crash rather than as "that is gone".
    """
    wants_html = "text/html" in request.headers.get("accept", "")
    if exc.status_code == 404 and wants_html:
        return templates.TemplateResponse(
            request, "notfound.html",
            {"detail": exc.detail, "admin": False,
             "signed_in": auth.is_admin(request),
             "feed_title": CFG.get("feed", {}).get("title", "Paperpod")},
            status_code=404,
        )
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code,
                        headers=getattr(exc, "headers", None))

# Bump when the terms text changes materially.
TERMS_UPDATED = "6 August 2026"

# Payload: {"id", "from_stage", "stop_after"}. None is the poison pill.
WORK_Q: "queue.Queue[dict | None]" = queue.Queue()

# One entry per worker thread, so /health can show what each is doing.
WORKER_STATE: dict[str, dict] = {}

# Episodes currently being processed. With more than one worker, the same id
# can be queued twice -- retry while it is already running, or the inbox
# watcher racing an upload -- and two threads on one episode would write the
# same chunk WAVs and DB rows over each other. The claim makes the second one
# a no-op instead.
_INFLIGHT: set[str] = set()
_INFLIGHT_LOCK = threading.Lock()


def worker_count() -> int:
    """How many episodes to process at once.

    Not an API limit: the pipeline is almost entirely waiting on Gemini, so
    concurrency is nearly free in CPU terms. The ceiling is the account's
    requests-per-minute and the volume's disk headroom during assembly, which
    is why this is config rather than a large fixed number.
    """
    return max(1, int(CFG.get("server", {}).get("workers", 1)))


def enqueue(episode_id: str, from_stage: str | None = None,
            stop_after: str | None = None) -> None:
    WORK_Q.put({"id": episode_id, "from_stage": from_stage,
                "stop_after": stop_after})


def _claim(episode_id: str) -> bool:
    with _INFLIGHT_LOCK:
        if episode_id in _INFLIGHT:
            return False
        _INFLIGHT.add(episode_id)
        return True


def _release(episode_id: str) -> None:
    with _INFLIGHT_LOCK:
        _INFLIGHT.discard(episode_id)


# --------------------------------------------------------------------------
# worker + watcher
# --------------------------------------------------------------------------

def _worker(name: str = "worker") -> None:
    db.init_db()
    state = WORKER_STATE.setdefault(name, {})
    state.update(alive=True, current=None, last_beat=None)
    try:
        _work_loop(state)
    finally:
        state["alive"] = False


def _work_loop(state: dict) -> None:
    while True:
        state["last_beat"] = db.now_iso()
        try:
            job = WORK_Q.get(timeout=5)
        except queue.Empty:
            continue
        try:
            if job is None:
                return  # poison pill: one per worker, used to stop them cleanly
            episode_id = job["id"]
            if not _claim(episode_id):
                log.info("episode %s already in flight; skipping duplicate",
                         episode_id)
                continue
            state["current"] = episode_id
            try:
                run.run_episode(episode_id, CFG,
                                from_stage=job.get("from_stage"),
                                stop_after=job.get("stop_after"))
            except Exception:
                log.exception("worker crashed on episode %s", episode_id)
            finally:
                state["current"] = None
                _release(episode_id)
        finally:
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
            except DuplicateEpisode as e:
                log.info("inbox skipped %s: %s", path.name, e)
                _move_to_processed(path)
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
        enqueue(episode_id)
    return episode_id


def _requeue_interrupted() -> None:
    """A process killed mid-run leaves episodes in a working status. Re-enqueue
    them; run.py restarts at the stage that was in flight, and TTS skips chunks
    already on disk."""
    for row in db.list_episodes():
        if row["status"] in ("queued", *run.STAGE_NAMES):
            log.info("resuming interrupted episode %s from %s", row["id"], row["status"])
            enqueue(row["id"])


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


# A running episode is only worrying if it has stopped moving. Chunks land
# every minute or two, so silence past this means something is wedged.
STALL_AFTER_S = 15 * 60


def _audio_is_stale(row) -> bool:
    """Audio built before the current script -- it is speaking older words.

    Only meaningful once both timestamps exist; an episode whose script
    predates this feature has no script_updated_at and is left alone rather
    than warned about.
    """
    built, written = row["audio_built_at"], row["script_updated_at"]
    if not built or not written:
        return False
    return written > built


def _progress(row) -> dict | None:
    """What a running episode is doing, and whether it is still moving."""
    if row["status"] not in run.STAGE_NAMES:
        return None
    note = row["progress"] or row["status"]
    since = None
    if row["progress_at"]:
        try:
            since = (datetime.now(timezone.utc)
                     - datetime.fromisoformat(row["progress_at"])).total_seconds()
        except ValueError:
            since = None
    return {
        "note": note,
        "idle_s": since,
        "idle": _fmt_duration(since) if since else None,
        "stalled": since is not None and since > STALL_AFTER_S,
    }


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


def safe_url(raw: str | None) -> str | None:
    """Only http(s) links survive. This value is rendered into an href, so a
    javascript: or data: URL here would be script injection on a public page."""
    raw = (raw or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return raw
    return None


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


def _grounding_corpus(row) -> str | None:
    """Titles and domains the script model consulted. A citation absent from the
    PDF but present here was looked up, not invented."""
    g = db.grounding(row)
    sources = g.get("sources") or []
    if not sources:
        return None
    # Queries count too: searching for a name is evidence of looking it up
    # rather than inventing it, and source titles are usually the paper's title
    # rather than its authors.
    parts = [f"{s.get('title', '')} {s.get('domain', '')}" for s in sources]
    parts.extend(g.get("queries") or [])
    return " ".join(parts)


def _episode_view(row) -> dict:
    cats = db.episode_categories(row)
    labels = category_labels(CFG)
    flags = (
        script_mod.citation_flags(row["script_md"], _paper_text(row["id"]),
                                  _grounding_corpus(row))
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
        # The stored value, not the display fallback, so an edit form does not
        # silently promote the paper's title into the episode's.
        "episode_title": row["episode_title"] or "",
        "paper_title": paper_title,
        "attribution": _attribution(row["title"], authors),
        "attrib_title": paper_title or "an untitled paper",
        "attrib_credit": _author_credit(authors),
        "attrib_stop": "" if _author_credit(authors).endswith(".") else ".",
        "source_url": safe_url(row["source_url"]),
        "tts_model": row["tts_model"] or CFG["models"]["tts"],
        "tts_model_pinned": bool(row["tts_model"]),
        "audio_built_at": row["audio_built_at"],
        "script_updated_at": row["script_updated_at"],
        "script_model": row["script_model"] or CFG["models"]["script"],
        "grounding": db.grounding(row),
        "script_md_present": bool(row["script_md"]),
        "summary": _blurb(row),
        "categories": cats,
        "doi": row["doi"],
        "cited_by": row["cited_by"],
        "cited_by_at": row["cited_by_at"],
        "cited_by_source": row["cited_by_source"],
        "category_labels": [labels.get(c, c) for c in cats],
        "authors": authors,
        "year": row["year"],
        "abstract": row["abstract"],
        "venue": _decaps((row["venue"] or "").strip()) or None,
        "status": row["status"],
        "status_label": (row["status"] or "").replace("_", " "),
        "progress": _progress(row),
        "script_note": row["script_note"],
        "can_revert_script": bool((row["script_prev"] or "").strip()),
        "audio_is_stale": _audio_is_stale(row),
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


def _default_retry_stage(row) -> str:
    """Which stage the retry picker should land on.

    A rewritten script needs audio, not a new script -- defaulting to
    `extracting` here would regenerate the script and throw the rewrite away,
    which is the opposite of what the button is for at that moment.
    """
    if row["status"] == run.NEEDS_REVIEW or _audio_is_stale(row):
        return "synthesizing"
    if row["status"] in run.STAGE_NAMES:
        return row["status"]
    return run.STAGE_NAMES[0]


def publish_blocker(row) -> str | None:
    """Why this episode cannot go public yet, or None if it can."""
    if row["status"] != "done":
        status = (row["status"] or "").replace("_", " ")
        if row["status"] == run.NEEDS_REVIEW:
            return "the script was rewritten and the audio has not been rebuilt"
        return f"episode is {status}, not done"
    if not row["audio_path"] or not Path(row["audio_path"]).exists():
        return "no audio file on disk"
    if unverified_flag_count(row) and not row["flags_reviewed"]:
        return "citation flags not reviewed"
    return None


def _current_gaps(stages) -> list[str]:
    """Gap warnings from the latest run of each stage, not every run ever.

    The stage log is append-only, so an episode that failed, was retried and
    then succeeded still carries the original ":gaps" rows. Reading all of them
    told you the audio was full of holes long after a retry had filled them --
    which is worse than saying nothing, because it trains you to ignore the one
    warning that is real.

    A fresh run of a stage supersedes whatever the previous run said about it:
    only a gap recorded *after* the last start of that stage still applies.
    """
    latest: dict[str, str | None] = {}
    for s in stages:
        base, _, suffix = s["stage"].partition(":")
        if not suffix:
            latest[base] = None          # this run replaces the old verdict
        elif suffix == "gaps" and s["detail"]:
            latest[base] = s["detail"]
    return [d for d in latest.values() if d]


# What each admin action did, said back to you. Every one of these routes
# redirects to the top of a long page, so without a message the only evidence
# anything happened is a field further down that you have to go and find --
# which reads exactly like a dead button.
DONE_MESSAGES = {
    "published": ("Published. It is now on the public site and in the feed.",
                  "visibility"),
    "unpublished": ("Unpublished. It is private again — the public site and the "
                    "feed no longer list it.", "visibility"),
    "reviewed": ("Citation flags marked as reviewed.", "flags"),
    "edited": ("Saved.", None),
    "reverted": ("Restored the previous script.", "scriptmodel"),
    "rewriting": ("Rewriting the script. This page refreshes itself; the new "
                  "script appears below when it lands.", "scriptmodel"),
    "retrying": ("Re-running. This page refreshes itself while it works.", None),
    "cited": ("Citation count updated.", "citations"),
    "nocite": ("No citation count found. OpenAlex has no record matching this "
               "paper's DOI or title — type the number in by hand below.",
               "citations"),
}


def _done(episode_id: str, key: str) -> RedirectResponse:
    anchor = DONE_MESSAGES.get(key, (None, None))[1]
    url = f"/episode/{episode_id}?done={key}"
    if anchor:
        url += f"#{anchor}"
    return RedirectResponse(url, status_code=303)


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
def admin(request: Request, error: str = "", deleted: str = "",
          category: str = "", sort: str = ""):
    if not auth.is_admin(request):
        return RedirectResponse("/admin/login", status_code=303)
    return _render_library(request, admin_mode=True, error=error,
                           deleted=bool(deleted), category=category, sort=sort)


@app.post("/episode/{episode_id}/edit")
async def edit_episode(request: Request, episode_id: str):
    """Hand-edit the public-facing strings. Submitting a field empty restores
    its generated fallback; omitting it entirely leaves it alone, so a partial
    form cannot silently wipe values it never showed.

    The form is read raw rather than through Form() parameters, which coerce an
    empty submitted value to None and so cannot tell "cleared" from "absent".

    Paper title and authors feed the attribution line, where an extraction
    error misnames a real person on a public page.
    """
    require_admin(request)
    if not db.get_episode(episode_id):
        raise HTTPException(404, "no such episode")

    form = await request.form()

    def sent(name: str) -> str | None:
        return " ".join(str(form[name]).split()) if name in form else None

    fields: dict = {}
    if "episode_title" in form:
        fields["episode_title"] = sent("episode_title") or None
    if "summary" in form:
        fields["summary"] = sent("summary") or None
    if "paper_title" in form:
        fields["title"] = sent("paper_title") or None
    if "authors" in form:
        names = [" ".join(a.split()) for a in str(form["authors"]).split(",")]
        fields["authors"] = json.dumps([a for a in names if a])
    if "categories_submitted" in form:
        # Checkboxes send nothing when all are cleared, so a hidden marker is
        # what separates "untag everything" from "this form has no tag fields".
        chosen = set(form.getlist("categories"))
        fields["categories"] = json.dumps(
            [c["slug"] for c in categories(CFG) if c["slug"] in chosen])
    if "doi" in form:
        fields["doi"] = ingest.citations.normalize_doi(sent("doi"))
    if "cited_by" in form:
        raw = (sent("cited_by") or "").replace(",", "").replace(" ", "")
        if not raw:
            # Cleared means "not known" -- so the provenance goes with it,
            # rather than leaving the card claiming a source for no number.
            fields.update(cited_by=None, cited_by_source=None, cited_by_at=None)
        else:
            try:
                count = int(raw)
                if count < 0:
                    raise ValueError(raw)
            except ValueError:
                # Silently discarding a typo would wipe a number that took
                # effort to find, and say nothing about why.
                raise HTTPException(
                    400, f"citations must be a whole number, not {raw!r}. "
                         "Leave it empty for “not known”.")
            fields.update(cited_by=count, cited_by_source=ingest.HAND_ENTERED,
                          cited_by_at=db.now_iso())
    if "source_url" in form:
        fields["source_url"] = safe_url(str(form["source_url"]))
    if "year" in form:
        raw = sent("year") or ""
        try:
            fields["year"] = int(raw) if raw else None
        except ValueError:
            fields["year"] = None

    db.update_episode(episode_id, **fields)
    return _done(episode_id, "edited")


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
    if want:
        demoted = db.demote_siblings(row["sha256"], episode_id)
        if demoted:
            log.info("episode %s is now canonical; unpublished %s", episode_id, demoted)
    if not want and reviewed:
        return _done(episode_id, "reviewed")
    return _done(episode_id, "published" if want else "unpublished")


def tts_choices() -> list[str]:
    """Models offered in the picker. Always includes the configured default so
    the list can never be empty."""
    listed = CFG.get("tts", {}).get("models") or []
    default = CFG["models"]["tts"]
    return list(dict.fromkeys([*listed, default]))


PROMPT_USED_BY = {
    "metadata.md": "Reading title, authors, year and abstract off the PDF.",
    "script_system.md": "The system instruction for every script. The segment "
                        "arc, the hard constraints and the house style all live "
                        "here — this is the main quality lever.",
    "script_user.md": "The per-episode request that carries the length budget.",
    "script_grounding.md": "Appended to the system instruction only when "
                           "[script] grounding is on.",
    "script_revise.md": "Rewriting an existing script to an editor's notes.",
    "episode_title.md": "Naming the finished episode.",
}


def _prompt_view(name: str) -> dict:
    override = prompt_override(name)
    body = override if override is not None else prompt_default(name)
    return {
        "name": name,
        "body": body,
        "default": prompt_default(name),
        "edited": override is not None,
        "used_by": PROMPT_USED_BY.get(name, ""),
        "warnings": prompt_warnings(name, body) if override is not None else [],
    }


@app.get("/admin/prompts", response_class=HTMLResponse)
def admin_prompts(request: Request, saved: str = "", reset: str = ""):
    """Edit the prompts without a redeploy.

    They are read at call time, so a save takes effect on the next episode.
    Edits land on the data volume rather than over the shipped files, which
    keeps the default available as something an edit can always be reverted to.
    """
    if not auth.is_admin(request):
        return RedirectResponse("/admin/login", status_code=303)
    return templates.TemplateResponse(
        request, "prompts.html",
        {"admin": True, "signed_in": True,
         "prompts": [_prompt_view(n) for n in prompt_names()],
         "saved": saved, "reset": reset,
         "feed_title": CFG.get("feed", {}).get("title", "Paperpod")},
    )


@app.post("/admin/prompts/{name}")
async def save_prompt_route(request: Request, name: str):
    require_admin(request)
    # `name` arrives from the URL. Anything not on the shipped allowlist must
    # never reach a filesystem path.
    if not is_prompt_name(name):
        raise HTTPException(404, "no such prompt")
    form = await request.form()
    if form.get("action") == "reset":
        reset_prompt(name)
        log.info("prompt %s reset to the shipped default", name)
        return RedirectResponse(f"/admin/prompts?reset={quote(name)}#{name}",
                                status_code=303)

    body = (form.get("body") or "")
    if body.replace("\r\n", "\n").strip() == prompt_default(name).strip():
        # Saving the default verbatim is a reset; keeping it as an override
        # would just freeze this copy against future upstream edits.
        reset_prompt(name)
    else:
        save_prompt(name, body.replace("\r\n", "\n"))
    log.info("prompt %s updated", name)
    return RedirectResponse(f"/admin/prompts?saved={quote(name)}#{name}",
                            status_code=303)


@app.get("/admin/models", response_class=HTMLResponse)
def admin_models(request: Request):
    """What this API key can actually call. Hardcoded model IDs go stale and a
    wrong one 404s every chunk of an episode, so ask the API instead."""
    if not auth.is_admin(request):
        return RedirectResponse("/admin/login", status_code=303)
    try:
        from pipeline.gemini import client

        models = [
            {"name": m.name.removeprefix("models/"),
             "display": m.display_name or "",
             "actions": ", ".join(m.supported_actions or [])}
            for m in client().models.list()
        ]
        models.sort(key=lambda m: m["name"])
        error = ""
    except Exception as e:
        models, error = [], str(e)

    return templates.TemplateResponse(
        request, "models.html",
        {"admin": True, "signed_in": True, "models": models, "error": error,
         "configured": CFG["models"], "choices": tts_choices()},
    )


@app.post("/episode/{episode_id}/clone")
def clone_episode(request: Request, episode_id: str, tts_model: str = Form("")):
    """Copy an episode's script onto a new episode with a different TTS model.

    Comparing two voice models needs the same words in both, so this reuses the
    script rather than regenerating it — the audio is the only variable, and the
    only thing paid for again."""
    require_admin(request)
    src = db.get_episode(episode_id)
    if not src:
        raise HTTPException(404, "no such episode")
    if not src["script_md"]:
        raise HTTPException(400, "nothing to clone: this episode has no script yet")
    if tts_model not in tts_choices():
        raise HTTPException(400, f"unknown TTS model {tts_model!r}")

    new_id = db.new_ulid()
    db.create_episode(new_id, src["source_path"], src["sha256"], status="queued")
    shutil.copy2(PAPERS_DIR / f"{episode_id}.pdf", PAPERS_DIR / f"{new_id}.pdf")
    db.update_episode(
        new_id,
        title=src["title"], authors=src["authors"], year=src["year"],
        abstract=src["abstract"], venue=src["venue"], summary=src["summary"],
        episode_title=src["episode_title"], source_url=src["source_url"],
        script_md=src["script_md"], flags_reviewed=src["flags_reviewed"],
        tts_model=tts_model, status="queued",
    )
    enqueue(new_id, "synthesizing")
    log.info("cloned %s to %s for TTS model %s", episode_id, new_id, tts_model)
    return RedirectResponse(f"/episode/{new_id}?queued=1", status_code=303)


def script_choices() -> list[str]:
    return script_mod.script_choices(CFG)


@app.post("/episode/{episode_id}/rewrite")
async def rewrite_script(request: Request, episode_id: str):
    """Rewrite the script -- to notes, to a different model, or both.

    Runs the scripting stage alone and stops. Audio is ~97% of an episode, so
    re-synthesizing on every wording change would make iteration unaffordable;
    the existing audio is left in place and marked stale instead.
    """
    require_admin(request)
    ep = db.get_episode(episode_id)
    if not ep:
        raise HTTPException(404, "no such episode")

    form = await request.form()
    mode = (form.get("mode") or "revise").strip()
    instructions = (form.get("instructions") or "").strip()
    model = (form.get("script_model") or "").strip()

    if mode not in ("revise", "regenerate"):
        raise HTTPException(400, f"unknown mode {mode!r}")
    if model and model not in script_choices():
        raise HTTPException(400, f"unknown script model {model!r}")
    if mode == "revise" and not instructions:
        raise HTTPException(400, "revising needs notes saying what to change")
    if mode == "revise" and not (ep["script_md"] or "").strip():
        raise HTTPException(400, "nothing to revise: this episode has no script yet")
    if ep["status"] in run.STAGE_NAMES:
        raise HTTPException(409, "this episode is already running; wait for it to finish")

    db.update_episode(episode_id, rewrite_json=json.dumps({
        "mode": mode, "instructions": instructions, "model": model or None,
    }), status="queued", error=None)
    enqueue(episode_id, from_stage="scripting", stop_after="scripting")
    log.info("queued %s rewrite of %s (model=%s)", mode, episode_id, model or "default")
    return _done(episode_id, "rewriting")


@app.post("/episode/{episode_id}/citations")
def refresh_citations_route(request: Request, episode_id: str):
    """Look the count up again. Counts go up over time, and a paper that had no
    OpenAlex record when it was ingested may have one now."""
    require_admin(request)
    if not db.get_episode(episode_id):
        raise HTTPException(404, "no such episode")
    # Explicit button, so it overrides a hand-entered number -- unlike the
    # automatic lookup during extraction, which leaves one alone.
    found = ingest.refresh_citations(episode_id, CFG, force=True)
    return RedirectResponse(
        f"/episode/{episode_id}?done={'cited' if found is not None else 'nocite'}"
        "#citations", status_code=303)


@app.post("/episode/{episode_id}/script/revert")
def revert_script(request: Request, episode_id: str):
    """Undo the last rewrite. One step back is enough to escape a bad edit."""
    require_admin(request)
    if not db.get_episode(episode_id):
        raise HTTPException(404, "no such episode")
    if not db.restore_script(episode_id):
        raise HTTPException(400, "no previous script to restore")
    return _done(episode_id, "reverted")


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
def library(request: Request, category: str = "", sort: str = ""):
    return _render_library(request, admin_mode=False, category=category, sort=sort)


# key -> (label, sort key, reverse). Order is the order the control offers them.
SORTS: dict[str, tuple] = {
    "created": ("Newest episodes", lambda e: (e["created_at"] or "", e["id"]), True),
    "published": ("Paper date", lambda e: (e["year"] or 0, e["created_at"] or ""), True),
    "cited": ("Most cited", lambda e: (e["cited_by"] if e["cited_by"] is not None else -1,
                                       e["year"] or 0), True),
}
DEFAULT_SORT = "created"


def _sorted_episodes(episodes: list[dict], sort: str) -> list[dict]:
    """Sorting is stable on a secondary key, so equal values keep a sensible
    order instead of shuffling between requests. Papers with no citation count
    sort last under "most cited" rather than mixing in among the zeroes:
    unknown and zero are different facts."""
    _, key, rev = SORTS.get(sort, SORTS[DEFAULT_SORT])
    return sorted(episodes, key=key, reverse=rev)


def _category_chips(episodes: list[dict], selected: str) -> list[dict]:
    """The filter row: only tags that actually have episodes behind them.

    An empty category is a dead end -- showing it invites a click that lands on
    "nothing here". Counts come from the same list being rendered, so the
    numbers can never disagree with the page.
    """
    counts: dict[str, int] = {}
    for ep in episodes:
        for slug in ep["categories"]:
            counts[slug] = counts.get(slug, 0) + 1
    chips = [{"slug": "", "label": "All", "count": len(episodes),
              "selected": not selected}]
    for c in categories(CFG):
        if counts.get(c["slug"]):
            chips.append({**c, "count": counts[c["slug"]],
                          "selected": c["slug"] == selected})
    # A row with only "All" in it is chrome, not navigation.
    return chips if len(chips) > 1 else []


def _render_library(request: Request, admin_mode: bool, error: str = "",
                    deleted: bool = False, category: str = "", sort: str = ""):
    all_episodes = [
        _episode_view(r) for r in db.list_episodes(published_only=not admin_mode)
    ]
    known = {c["slug"] for c in categories(CFG)}
    category = category if category in known else ""
    # Failures are moved out of the reading list: they are maintenance, not
    # something to browse. They stay reachable, collapsed, below the fold.
    episodes = [e for e in all_episodes if e["status"] != "failed"]
    failed = ([{**e, "short_error": _short_error(e["error"])}
               for e in all_episodes if e["status"] == "failed"]
              if admin_mode else [])
    chips = _category_chips(episodes, category)
    if category:
        episodes = [e for e in episodes if category in e["categories"]]
    sort = sort if sort in SORTS else DEFAULT_SORT
    episodes = _sorted_episodes(episodes, sort)
    return templates.TemplateResponse(
        request,
        "library.html",
        {
            "episodes": episodes,
            "failed": failed,
            "admin": admin_mode,
            "signed_in": auth.is_admin(request),
            "tts_choices": tts_choices(),
            "queue": [e for e in all_episodes if e["status"] in run.STAGE_NAMES
                      or e["status"] == "queued"],
            "workers": worker_count(),
            "deleted": deleted,
            "chips": chips,
            "category": category,
            "category_label": category_labels(CFG).get(category, ""),
            "sort": sort,
            "sorts": [{"key": k, "label": v[0]} for k, v in SORTS.items()],
            "error": error,
            "total_cost": sum(e["cost_usd"] for e in all_episodes),
            "feed_url": CFG["server"]["base_url"].rstrip("/") + "/feed.xml",
            "feed_title": CFG.get("feed", {}).get("title", "Paperpod"),
            "feed_description": CFG.get("feed", {}).get("description", ""),
        },
    )


@app.post("/upload")
async def upload(request: Request, file: UploadFile, tts_model: str = Form("")):
    require_admin(request)
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf files are accepted")
    if tts_model and tts_model not in tts_choices():
        raise HTTPException(400, f"unknown TTS model {tts_model!r}")
    staged = INBOX_DIR / f"upload-{db.new_ulid()}.pdf"
    with open(staged, "wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        episode_id = _enqueue_path(staged, move_to_processed=True)
    except DuplicateEpisode as e:
        # Not a failure: this paper is already in the library. Show the episode
        # it landed as, rather than bouncing to a page that looks unchanged.
        _move_to_processed(staged)
        return RedirectResponse(f"/episode/{e.episode_id}?dup=1", status_code=303)
    except PipelineError as e:
        _move_to_processed(staged)
        return RedirectResponse(f"/admin?error={quote(str(e))}", status_code=303)
    if tts_model:
        db.update_episode(episode_id, tts_model=tts_model)
    return RedirectResponse(f"/episode/{episode_id}?queued=1", status_code=303)


@app.get("/episode/{episode_id}", response_class=HTMLResponse)
def episode_page(request: Request, episode_id: str, done: str = ""):
    row = db.get_episode(episode_id)
    if not row:
        raise HTTPException(404, "no such episode")
    admin_mode = auth.is_admin(request)
    # An unpublished episode does not exist as far as the public is concerned.
    if not admin_mode and not (row["published"] and row["status"] == "done"):
        raise HTTPException(404, "no such episode")

    view = _episode_view(row)
    versions = []
    if admin_mode:
        for sib in db.siblings(row["sha256"], episode_id):
            sv = _episode_view(sib)
            versions.append(sv)
        if versions:
            versions.insert(0, view)
    stages = db.get_stage_log(episode_id) if admin_mode else []
    gaps = _current_gaps(stages)
    return templates.TemplateResponse(
        request,
        "episode.html",
        {
            "ep": view,
            "admin": admin_mode,
            "done": DONE_MESSAGES.get(done, (None, None))[0] if admin_mode else None,
            "gaps": gaps,
            "signed_in": admin_mode,
            "lines": _script_lines(row["script_md"], view["flags"]) if row["script_md"] else [],
            "stages": stages,
            "retry_stages": run.STAGE_NAMES,
            "default_retry_stage": _default_retry_stage(row),
            "publish_blocker": publish_blocker(row) if admin_mode else None,
            "tts_choices": tts_choices(),
            "script_choices": script_choices(),
            "all_categories": categories(CFG),
            "versions": versions,
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
    enqueue(episode_id, stage)
    return _done(episode_id, "retrying")


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
        "workers": worker_count(),
        # Kept for the client's health line, which only asks "is anything
        # running": true while at least one thread is beating.
        "worker_alive": any(s.get("alive") for s in WORKER_STATE.values()),
        "worker_current": [
            s["current"] for s in WORKER_STATE.values() if s.get("current")
        ],
        "worker_last_beat": max(
            (s["last_beat"] for s in WORKER_STATE.values() if s.get("last_beat")),
            default=None,
        ),
        "in_flight": sorted(_INFLIGHT),
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
        if source := safe_url(row["source_url"]):
            summary += f"\n\nSource paper: {source}"
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
