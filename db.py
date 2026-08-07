"""SQLite schema and queries. sqlite3 stdlib, no ORM, one connection per thread."""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

from config import DATA_DIR

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


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive column migrations, so an existing library survives an upgrade."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(episode)")}
    for name, decl in (("summary", "TEXT"), ("episode_title", "TEXT"),
                       ("cost_json", "TEXT"), ("source_url", "TEXT"), ("tts_model", "TEXT"), ("audio_built_at", "TEXT"), ("script_model", "TEXT"),
                       ("grounding_json", "TEXT"),
                       ("published", "INTEGER DEFAULT 0"),
                       ("progress", "TEXT"),
                       ("script_prev", "TEXT"),
                       ("script_note", "TEXT"),
                       ("script_updated_at", "TEXT"),
                       ("rewrite_json", "TEXT"),
                       ("progress_at", "TEXT"),
                       ("flags_reviewed", "INTEGER DEFAULT 0")):
        if name not in cols:
            conn.execute(f"ALTER TABLE episode ADD COLUMN {name} {decl}")
    conn.commit()


# ---- episode ----

def create_episode(id: str, source_path: str, sha256: str | None,
                   status: str = "queued", error: str | None = None) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO episode (id, created_at, source_path, sha256, status, error)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (id, now_iso(), source_path, sha256, status, error),
    )
    conn.commit()


def get_episode(id: str) -> sqlite3.Row | None:
    return get_conn().execute("SELECT * FROM episode WHERE id = ?", (id,)).fetchone()


def list_episodes(published_only: bool = False) -> list[sqlite3.Row]:
    where = "WHERE published = 1 AND status = 'done'" if published_only else ""
    return get_conn().execute(
        f"SELECT * FROM episode {where} ORDER BY created_at DESC, id DESC"
    ).fetchall()


def siblings(sha256: str | None, exclude_id: str) -> list[sqlite3.Row]:
    """Other episodes built from the same PDF. Re-voicing a script produces one
    of these, so they are alternate renderings of one paper rather than
    different papers."""
    if not sha256:
        return []
    return get_conn().execute(
        "SELECT * FROM episode WHERE sha256 = ? AND id != ? ORDER BY created_at",
        (sha256, exclude_id),
    ).fetchall()


def demote_siblings(sha256: str | None, keep_id: str) -> list[str]:
    """Exactly one rendering of a paper may be public: the canonical one. Returns
    the ids that were unpublished."""
    if not sha256:
        return []
    conn = get_conn()
    rows = conn.execute(
        "SELECT id FROM episode WHERE sha256 = ? AND id != ? AND published = 1",
        (sha256, keep_id),
    ).fetchall()
    if rows:
        conn.execute(
            "UPDATE episode SET published = 0 WHERE sha256 = ? AND id != ?",
            (sha256, keep_id),
        )
        conn.commit()
    return [r["id"] for r in rows]


def find_by_sha(sha256: str) -> sqlite3.Row | None:
    return get_conn().execute(
        "SELECT * FROM episode WHERE sha256 = ?", (sha256,)
    ).fetchone()


def update_episode(id: str, **fields) -> None:
    if not fields:
        return
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


def delete_episode(id: str) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM episode WHERE id = ?", (id,))
    conn.execute("DELETE FROM stage_log WHERE episode_id = ?", (id,))
    conn.commit()


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
