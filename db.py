"""SQLite schema and queries. sqlite3 stdlib, no ORM, one connection per thread.

Two things live here that used to be one. A `paper` is a work: a PDF, its
title, its authors, how often it has been cited. An `episode` is a discussion:
a script, some audio, a number in the feed. For most of this project's life
they were the same row, which was true right up until an episode needed to talk
about two papers at once.

`episode_paper` joins them, carrying the two facts that only make sense at the
join: what order the papers come in, and whether each one is a principal (the
work the episode is about) or a reference (a work it draws on). A solo episode
is one join row, and everything below reads the same as it did before.
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR, PAPERS_DIR

DB_PATH = DATA_DIR / "paperpod.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS episode (
  id            TEXT PRIMARY KEY,   -- ulid
  created_at    TEXT NOT NULL,
  source_path   TEXT NOT NULL,
  sha256        TEXT,               -- content hash, dedupe key
  title         TEXT,
  authors       TEXT,               -- JSON array
  year          INTEGER,
  abstract      TEXT,
  summary       TEXT,               -- 1-2 sentence blurb for the library
  episode_title TEXT,               -- LLM-written title for the episode itself
  venue         TEXT,
  source_url    TEXT,               -- link to the paper, when it is public
  tts_model     TEXT,               -- overrides config for this episode
  audio_built_at TEXT,              -- when this audio was assembled
  script_model  TEXT,               -- model that actually wrote the script
  grounding_json TEXT,              -- {queries: [], sources: [{title,uri,domain}]}
  status        TEXT NOT NULL,      -- queued|extracting|scripting|synthesizing|assembling|done|failed
  script_md     TEXT,
  audio_path    TEXT,
  duration_s    REAL,
  error         TEXT,
  cost_usd      REAL DEFAULT 0,
  cost_json     TEXT,               -- {stage: usd} breakdown
  published     INTEGER DEFAULT 0,  -- visible on the public site and feed
  flags_reviewed INTEGER DEFAULT 0  -- citation flags checked by a human
);
CREATE INDEX IF NOT EXISTS idx_episode_sha ON episode(sha256);

CREATE TABLE IF NOT EXISTS stage_log (
  episode_id TEXT NOT NULL,
  stage      TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at   TEXT,
  ok         INTEGER,
  detail     TEXT
);
CREATE INDEX IF NOT EXISTS idx_stage_log_episode ON stage_log(episode_id);

-- A work, as opposed to a discussion of one. Everything here is a fact about
-- the PDF and would be the same fact if the episode were rebuilt, re-voiced or
-- never made at all -- which is exactly why it cannot live on the episode once
-- one episode can cover several papers.
CREATE TABLE IF NOT EXISTS paper (
  id              TEXT PRIMARY KEY,   -- ulid; also the stored PDF's filename
  created_at      TEXT NOT NULL,
  sha256          TEXT,               -- content hash, dedupe key
  source_path     TEXT,               -- where it was uploaded from
  title           TEXT,
  authors         TEXT,               -- JSON array
  year            INTEGER,
  abstract        TEXT,
  summary         TEXT,               -- 1-2 sentence blurb for the library
  venue           TEXT,
  source_url      TEXT,               -- link to the paper, when it is public
  doi             TEXT,
  categories      TEXT,               -- JSON array of slugs
  work_kind       TEXT,               -- empirical|theoretical; picks the arc
  cited_by        INTEGER,
  cited_by_at     TEXT,
  cited_by_source TEXT,
  dossier_json    TEXT                -- research on how the work landed
);
CREATE INDEX IF NOT EXISTS idx_paper_sha ON paper(sha256);

-- Which works an episode is built from. `position` is running order, so the
-- first principal is "the paper" wherever something still needs just one.
CREATE TABLE IF NOT EXISTS episode_paper (
  episode_id TEXT NOT NULL,
  paper_id   TEXT NOT NULL,
  position   INTEGER NOT NULL DEFAULT 0,
  role       TEXT NOT NULL DEFAULT 'principal',  -- principal|reference
  PRIMARY KEY (episode_id, paper_id)
);
CREATE INDEX IF NOT EXISTS idx_episode_paper_paper ON episode_paper(paper_id);

-- Sticky choices from the creation wizard. A key/value table rather than
-- columns because these are preferences, not facts about an episode: the set
-- of them changes when the wizard changes, and an unset key must mean "use the
-- configured default" rather than NULL-meaning-something.
CREATE TABLE IF NOT EXISTS setting (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""

_local = threading.local()

_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32


def new_ulid() -> str:
    val = (int(time.time() * 1000) << 80) | int.from_bytes(os.urandom(10), "big")
    return "".join(_ULID_ALPHABET[(val >> (5 * i)) & 31] for i in range(25, -1, -1))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


PRINCIPAL = "principal"
REFERENCE = "reference"
ROLES = (PRINCIPAL, REFERENCE)

# Facts that belong to the work rather than to the discussion of it. These are
# read from `paper` and written with update_paper(); the identically-named
# columns still on `episode` are the pre-split originals, kept as a backstop and
# no longer read once a paper row exists.
PAPER_FIELDS = (
    "sha256", "source_path", "title", "authors", "year", "abstract", "summary",
    "venue", "source_url", "doi", "categories", "work_kind",
    "cited_by", "cited_by_at", "cited_by_source", "dossier_json",
)


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    _migrate(conn)
    _split_papers(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive column migrations, so an existing library survives an upgrade."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(episode)")}
    for name, decl in (("summary", "TEXT"), ("episode_title", "TEXT"),
                       ("cost_json", "TEXT"), ("source_url", "TEXT"), ("tts_model", "TEXT"), ("audio_built_at", "TEXT"), ("script_model", "TEXT"),
                       ("grounding_json", "TEXT"),
                       ("published", "INTEGER DEFAULT 0"),
                       ("progress", "TEXT"),
                       ("categories", "TEXT"),
                       ("doi", "TEXT"),
                       ("cited_by", "INTEGER"),
                       ("cited_by_at", "TEXT"),
                       ("cited_by_source", "TEXT"),
                       ("script_prev", "TEXT"),
                       ("script_note", "TEXT"),
                       ("script_updated_at", "TEXT"),
                       ("rewrite_json", "TEXT"),
                       ("progress_at", "TEXT"),
                       ("episode_number", "INTEGER"),
                       ("voice_a", "TEXT"),
                       ("voice_b", "TEXT"),
                       ("metadata_model", "TEXT"),
                       # Distinct from script_model, which records what
                       # actually ran. This is what to ask for next time.
                       ("script_model_wanted", "TEXT"),
                       ("outline_json", "TEXT"),
                       ("target_words", "INTEGER"),
                       ("length_policy", "TEXT"),
                       ("dossier_json", "TEXT"),
                       ("research", "TEXT"),
                       ("work_kind", "TEXT"),
                       ("failed_at", "TEXT"),
                       ("flags_reviewed", "INTEGER DEFAULT 0")):
        if name not in cols:
            conn.execute(f"ALTER TABLE episode ADD COLUMN {name} {decl}")
    conn.commit()


def _split_papers(conn: sqlite3.Connection) -> None:
    """Give every pre-split episode the paper row it always implied.

    The new paper takes the episode's own id. That is not tidiness -- it is
    what keeps the migration off the filesystem. A stored PDF is named for its
    paper, and before the split it was named for its episode, so reusing the id
    means the files on a live volume do not move. Papers created from here on
    get their own ulid, since an episode may now have several.
    """
    rows = conn.execute(
        "SELECT e.* FROM episode e LEFT JOIN episode_paper ep"
        " ON ep.episode_id = e.id WHERE ep.episode_id IS NULL"
    ).fetchall()
    if not rows:
        return
    # Only the columns actually on this database. A library old enough to
    # predate one of them would otherwise take the whole migration down, and a
    # migration is the one thing that gets no second attempt.
    have = {r["name"] for r in conn.execute("PRAGMA table_info(episode)")}
    fields = [f for f in PAPER_FIELDS if f in have]
    cols = ", ".join(fields)
    holes = ", ".join("?" for _ in fields)
    for row in rows:
        conn.execute(
            f"INSERT INTO paper (id, created_at, {cols}) VALUES (?, ?, {holes})",
            (row["id"], row["created_at"], *(row[f] for f in fields)),
        )
        conn.execute(
            "INSERT INTO episode_paper (episode_id, paper_id, position, role)"
            " VALUES (?, ?, 0, ?)",
            (row["id"], row["id"], PRINCIPAL),
        )
    conn.commit()


# ---- settings ----

def get_settings() -> dict[str, str]:
    return {r["key"]: r["value"] for r in
            get_conn().execute("SELECT key, value FROM setting")}


def set_settings(values: dict[str, str]) -> None:
    conn = get_conn()
    conn.executemany(
        "INSERT INTO setting (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        list(values.items()),
    )
    conn.commit()


def clear_settings() -> None:
    """Back to the configured defaults. Deleting rather than writing the
    defaults in, so a later change to config.toml is picked up instead of
    frozen against a copy taken today."""
    conn = get_conn()
    conn.execute("DELETE FROM setting")
    conn.commit()


# ---- paper ----

def paper_pdf(paper_id: str) -> Path:
    """Where a paper's bytes live. The filename is the paper id, which is why
    this sits with identity rather than with the pipeline: nothing else needs
    to know, and nothing else gets to decide."""
    return PAPERS_DIR / f"{paper_id}.pdf"


def create_paper(source_path: str | None = None, sha256: str | None = None,
                 **fields) -> str:
    paper_id = new_ulid()
    cols = {"source_path": source_path, "sha256": sha256,
            **{k: v for k, v in fields.items() if k in PAPER_FIELDS}}
    names = ", ".join(cols)
    holes = ", ".join("?" for _ in cols)
    conn = get_conn()
    conn.execute(
        f"INSERT INTO paper (id, created_at, {names}) VALUES (?, ?, {holes})",
        (paper_id, now_iso(), *cols.values()),
    )
    conn.commit()
    return paper_id


def get_paper(paper_id: str) -> sqlite3.Row | None:
    return get_conn().execute("SELECT * FROM paper WHERE id = ?",
                              (paper_id,)).fetchone()


def find_paper_by_sha(sha256: str) -> sqlite3.Row | None:
    return get_conn().execute("SELECT * FROM paper WHERE sha256 = ?",
                              (sha256,)).fetchone()


def update_paper(paper_id: str, **fields) -> None:
    fields = {k: v for k, v in fields.items() if k in PAPER_FIELDS}
    if not fields:
        return
    conn = get_conn()
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE paper SET {cols} WHERE id = ?",
                 (*fields.values(), paper_id))
    conn.commit()


def attach_paper(episode_id: str, paper_id: str, position: int = 0,
                 role: str = PRINCIPAL) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO episode_paper (episode_id, paper_id, position, role)"
        " VALUES (?, ?, ?, ?) ON CONFLICT(episode_id, paper_id) DO UPDATE SET"
        " position = excluded.position, role = excluded.role",
        (episode_id, paper_id, position, role if role in ROLES else PRINCIPAL),
    )
    conn.commit()


def papers_for(episode_id: str) -> list[sqlite3.Row]:
    """Every paper this episode is built from, in running order. Each row is a
    paper plus the two things that only exist at the join, `role` and
    `position`."""
    return get_conn().execute(
        "SELECT p.*, ep.role AS role, ep.position AS position"
        " FROM episode_paper ep JOIN paper p ON p.id = ep.paper_id"
        " WHERE ep.episode_id = ? ORDER BY ep.position, p.created_at",
        (episode_id,),
    ).fetchall()


def principal_paper(episode_id: str) -> sqlite3.Row | None:
    """The work the episode is about. With two principals -- two papers that
    disagree, say -- this is the first of them, which is what anything wanting
    exactly one paper should get."""
    rows = papers_for(episode_id)
    return next((r for r in rows if r["role"] == PRINCIPAL), rows[0] if rows else None)


def update_principal(episode_id: str, **fields) -> None:
    """Write paper facts to the episode's principal. The metadata and citation
    stages know an episode id and nothing else; this is how they reach the
    work."""
    paper = principal_paper(episode_id)
    if paper is not None:
        update_paper(paper["id"], **fields)


def paper_paths(episode_id: str) -> list[Path]:
    """The PDFs to attach, in running order. Missing files are dropped: the
    caller's complaint is better than a stack trace from the API."""
    return [p for p in (paper_pdf(r["id"]) for r in papers_for(episode_id))
            if p.exists()]


def episodes_for_paper(paper_id: str) -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT e.* FROM episode_paper ep JOIN episode e ON e.id = ep.episode_id"
        " WHERE ep.paper_id = ? ORDER BY e.created_at",
        (paper_id,),
    ).fetchall()


# ---- episode ----

def _merge(row: sqlite3.Row, paper: sqlite3.Row | None) -> dict:
    """An episode and its principal paper as one mapping.

    Nearly everything that asks for an episode wants the work's title and
    authors alongside it, and did so through `row["title"]` for the whole of
    this project's life. Keeping that working is not a compatibility shim: "the
    episode and the paper it is about" is a real thing to want, and multi-paper
    callers have papers_for() to ask the other question with.

    With no paper row -- only reachable before the migration has run -- the
    episode's own pre-split columns answer instead.
    """
    out = dict(row)
    if paper is not None:
        out.update({f: paper[f] for f in PAPER_FIELDS})
        out["paper_id"] = paper["id"]
    else:
        out["paper_id"] = None
    return out


def create_episode(id: str, source_path: str, sha256: str | None,
                   status: str = "queued", error: str | None = None,
                   failed_at: str | None = None,
                   papers: list[str] | None = None) -> None:
    """Create an episode and give it its papers.

    `papers` names existing paper rows to attach. Left out, one is made from
    the source_path and hash given here -- the common case, and the reason an
    episode can never exist without a paper to be about. Pass an empty list to
    attach them yourself, which is what a caller with roles in mind wants.
    """
    conn = get_conn()
    conn.execute(
        "INSERT INTO episode (id, created_at, source_path, sha256, status, error,"
        " failed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (id, now_iso(), source_path, sha256, status, error, failed_at),
    )
    conn.commit()
    if papers is None:
        papers = [create_paper(source_path=source_path, sha256=sha256)]
    for position, paper_id in enumerate(papers):
        attach_paper(id, paper_id, position=position, role=PRINCIPAL)


def get_episode(id: str) -> dict | None:
    row = get_conn().execute("SELECT * FROM episode WHERE id = ?", (id,)).fetchone()
    return None if row is None else _merge(row, principal_paper(id))


def list_episodes(published_only: bool = False) -> list[dict]:
    where = "WHERE published = 1 AND status = 'done'" if published_only else ""
    conn = get_conn()
    rows = conn.execute(
        f"SELECT * FROM episode {where} ORDER BY created_at DESC, id DESC"
    ).fetchall()
    # One query for the papers rather than one per episode: this runs on every
    # library and feed request.
    principals: dict[str, sqlite3.Row] = {}
    for r in conn.execute(
        "SELECT p.*, ep.episode_id AS episode_id, ep.role AS role,"
        " ep.position AS position FROM episode_paper ep"
        " JOIN paper p ON p.id = ep.paper_id ORDER BY ep.position, p.created_at"
    ):
        seen = principals.get(r["episode_id"])
        if seen is None or (seen["role"] != PRINCIPAL and r["role"] == PRINCIPAL):
            principals[r["episode_id"]] = r
    return [_merge(r, principals.get(r["id"])) for r in rows]


def assign_episode_number(episode_id: str) -> int | None:
    """Give an episode its number, the first time it is published.

    Assigned at publish rather than at upload because the two orders are not
    the same. Papers that failed, that are still private, and the extra
    renderings created when comparing voice models would all consume numbers
    they never use, and the public feed would count 1, 2, 5, 9.

    Once given, a number is never taken back or reused -- unpublishing keeps
    it, so a listener's "episode 7" still means the same episode afterwards.

    A re-voiced rendering inherits the number of its sibling: it is the same
    paper and the same discussion in a different voice, and publishing it
    unpublishes the other. Calling that a new episode would advertise a
    duplicate.
    """
    conn = get_conn()
    row = conn.execute("SELECT episode_number FROM episode WHERE id = ?",
                       (episode_id,)).fetchone()
    if row is None or row["episode_number"] is not None:
        return row["episode_number"] if row else None

    number = next((s["episode_number"] for s in siblings(episode_id)
                   if s["episode_number"] is not None), None)
    if number is None:
        highest = conn.execute(
            "SELECT MAX(episode_number) AS n FROM episode").fetchone()["n"]
        number = (highest or 0) + 1

    conn.execute("UPDATE episode SET episode_number = ? WHERE id = ?",
                 (number, episode_id))
    conn.commit()
    return number


def _paper_ids(episode_id: str) -> frozenset[str]:
    return frozenset(r["paper_id"] for r in get_conn().execute(
        "SELECT paper_id FROM episode_paper WHERE episode_id = ?", (episode_id,)))


def siblings(episode_id: str) -> list[sqlite3.Row]:
    """Other episodes built from the same papers. Re-voicing a script produces
    one of these, so they are alternate renderings of one discussion rather
    than different episodes.

    The whole set has to match, not merely overlap. Two papers that disagree
    and a solo episode about one of them share a paper, but they are not two
    versions of the same thing, and publishing one must not unpublish the other.
    """
    mine = _paper_ids(episode_id)
    if not mine:
        return []
    conn = get_conn()
    candidates = conn.execute(
        "SELECT DISTINCT episode_id FROM episode_paper WHERE paper_id IN"
        f" ({', '.join('?' for _ in mine)}) AND episode_id != ?",
        (*mine, episode_id),
    ).fetchall()
    matched = [r["episode_id"] for r in candidates if _paper_ids(r["episode_id"]) == mine]
    if not matched:
        return []
    rows = conn.execute(
        f"SELECT * FROM episode WHERE id IN ({', '.join('?' for _ in matched)})"
        " ORDER BY created_at", matched,
    ).fetchall()
    return [_merge(r, principal_paper(r["id"])) for r in rows]


def demote_siblings(keep_id: str) -> list[str]:
    """Exactly one rendering of a discussion may be public: the canonical one.
    Returns the ids that were unpublished."""
    ids = [s["id"] for s in siblings(keep_id) if s["published"]]
    if ids:
        conn = get_conn()
        conn.execute(
            f"UPDATE episode SET published = 0 WHERE id IN ({', '.join('?' for _ in ids)})",
            ids,
        )
        conn.commit()
    return ids


def mark_failed(id: str, error: str) -> None:
    """Fail an episode and record when. The timestamp is what lets the library
    tell a failure that happened just now from one from last week: a failed
    episode leaves the main list for a collapsed box, so a fresh one needs to
    announce itself or it reads as having vanished."""
    update_episode(id, status="failed", error=error, failed_at=now_iso())


def update_episode(id: str, **fields) -> None:
    if not fields:
        return
    # The pre-split columns are still on the table, so a stray
    # update_episode(id, title=...) would write somewhere real, succeed, and be
    # invisible from that moment on -- get_episode reads the title from the
    # paper. Refusing is the only version of this that fails loudly.
    stray = [f for f in fields if f in PAPER_FIELDS]
    if stray:
        raise ValueError(
            f"{', '.join(sorted(stray))} belong(s) to the paper, not the episode; "
            "use update_paper() or update_principal()")
    conn = get_conn()
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE episode SET {cols} WHERE id = ?", (*fields.values(), id))
    conn.commit()


def set_script(id: str, text: str, note: str | None = None) -> None:
    """Store a new script, keeping the previous one for a one-step undo.

    script_updated_at is what makes stale audio detectable: an episode whose
    script is newer than its audio is playing words nobody approved.
    """
    row = get_episode(id)
    prev = row["script_md"] if row else None
    update_episode(id, script_md=text, script_note=note,
                   script_updated_at=now_iso(),
                   **({"script_prev": prev} if prev and prev != text else {}))


def restore_script(id: str) -> bool:
    """Swap the stored script back to the previous one. Returns False if there
    is nothing to go back to."""
    row = get_episode(id)
    if not row or not (row["script_prev"] or "").strip():
        return False
    update_episode(id, script_md=row["script_prev"], script_prev=row["script_md"],
                   script_note=None, script_updated_at=now_iso())
    return True


def episode_categories(row) -> list[str]:
    """Tag slugs on an episode. Stored as JSON so an episode can carry several
    -- a classic AI paper belongs under both filters, not one."""
    try:
        val = json.loads(row["categories"]) if row["categories"] else []
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(v) for v in val] if isinstance(val, list) else []


def set_progress(id: str, note: str | None) -> None:
    """A human-readable note on what a running stage is doing right now.

    Timestamped, because the useful question about a long-running episode is
    not "how far along" but "is it still moving".
    """
    update_episode(id, progress=note, progress_at=now_iso() if note else None)


def add_cost(id: str, usd: float, stage: str = "other") -> None:
    """Accumulate spend on the episode total and on the per-stage breakdown.
    TTS dominates by a wide margin, so the split is what makes the number
    actionable."""
    conn = get_conn()
    row = conn.execute("SELECT cost_json FROM episode WHERE id = ?", (id,)).fetchone()
    if row is None:
        return
    try:
        breakdown = json.loads(row["cost_json"]) if row["cost_json"] else {}
    except (json.JSONDecodeError, TypeError):
        breakdown = {}
    breakdown[stage] = round(breakdown.get(stage, 0.0) + usd, 8)
    conn.execute(
        "UPDATE episode SET cost_usd = COALESCE(cost_usd, 0) + ?, cost_json = ?"
        " WHERE id = ?",
        (usd, json.dumps(breakdown), id),
    )
    conn.commit()


def grounding(row: sqlite3.Row) -> dict:
    """Web sources the script model consulted, if grounding was on."""
    try:
        data = json.loads(row["grounding_json"]) if row["grounding_json"] else {}
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def cost_breakdown(row: sqlite3.Row) -> dict[str, float]:
    try:
        data = json.loads(row["cost_json"]) if row["cost_json"] else {}
    except (json.JSONDecodeError, TypeError):
        return {}
    return {k: float(v) for k, v in data.items()} if isinstance(data, dict) else {}


def delete_episode(id: str) -> list[str]:
    """Delete an episode and return the papers left with nothing referring to
    them, whose PDFs the caller should now remove.

    Reported rather than deleted here because a paper is a file on disk as much
    as a row, and unlinking is not this module's job. Papers still attached to
    another episode -- a re-voicing, or a comparison the paper also appears in
    -- are not in the list, which is the whole point of asking.
    """
    conn = get_conn()
    paper_ids = _paper_ids(id)
    conn.execute("DELETE FROM episode WHERE id = ?", (id,))
    conn.execute("DELETE FROM stage_log WHERE episode_id = ?", (id,))
    conn.execute("DELETE FROM episode_paper WHERE episode_id = ?", (id,))
    orphans = [p for p in paper_ids if not conn.execute(
        "SELECT 1 FROM episode_paper WHERE paper_id = ? LIMIT 1", (p,)).fetchone()]
    if orphans:
        conn.execute(
            f"DELETE FROM paper WHERE id IN ({', '.join('?' for _ in orphans)})",
            orphans)
    conn.commit()
    return orphans


def episode_authors(row: sqlite3.Row) -> list[str]:
    try:
        return json.loads(row["authors"]) if row["authors"] else []
    except (json.JSONDecodeError, TypeError):
        return []


# ---- stage log ----

def stage_start(episode_id: str, stage: str) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO stage_log (episode_id, stage, started_at) VALUES (?, ?, ?)",
        (episode_id, stage, now_iso()),
    )
    conn.commit()


def stage_end(episode_id: str, stage: str, ok: bool, detail: str = "") -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE stage_log SET ended_at = ?, ok = ?, detail = ?"
        " WHERE rowid = (SELECT MAX(rowid) FROM stage_log"
        "                WHERE episode_id = ? AND stage = ? AND ended_at IS NULL)",
        (now_iso(), 1 if ok else 0, detail, episode_id, stage),
    )
    conn.commit()


def get_stage_log(episode_id: str) -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT * FROM stage_log WHERE episode_id = ? ORDER BY rowid",
        (episode_id,),
    ).fetchall()
