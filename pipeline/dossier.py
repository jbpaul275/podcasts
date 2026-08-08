"""Research: what happened to this work after it was published.

The Context segment is the weakest part of the arc by construction. The script
prompt forbids naming outside work that is not in the paper's own literature
review, so Context comes out either hedged to the point of saying nothing or as
a rehash of the paper's framing. For a book like Kuhn's that is fatal — the
interesting thing is the reception, and none of the reception is in the book.

This stage goes and finds it, with Gemini's search grounding, and stores what
it finds as a readable artifact.

Two rules make it safe rather than dangerous.

Every entry carries a source URL, and an entry without one is dropped. A
dossier's whole value is that the script can lean on it; an unverifiable one is
the exact failure the citation flags exist to catch, moved a step upstream and
dressed up as research.

And no quotations. Putting invented words in the mouth of a named, often
living, academic is the highest-risk thing this system could do, and it would
be published as audio. Quotable passages come from the attached work, where
they can be checked against the text.

Off by default: it is a search-grounded call on top of an already-expensive
pipeline, and most papers do not need it.
"""

import json
import logging

import db
from config import load_prompt
from . import ModelUnusable, PipelineError
from .gemini import call_with_retry, client, pdf_part, record_cost, strip_fences
from .script import _script_models, collect_grounding

log = logging.getLogger("paperpod.dossier")

KINDS = ("critic", "extension", "held_up", "did_not_hold")
KIND_LABELS = {
    "critic": "Objected",
    "extension": "Built on it",
    "held_up": "Held up",
    "did_not_hold": "Did not hold up",
}


def wanted(row) -> bool:
    try:
        return str(row["research"] or "").strip() == "on" if row is not None else False
    except (IndexError, KeyError, TypeError):
        return False


def _source_ok(url: str) -> bool:
    """An http(s) URL and nothing else. This value is rendered as a link on an
    admin page, so a javascript: entry would be script injection by way of a
    research note."""
    return url.startswith("https://") or url.startswith("http://")


def _parse(text: str) -> dict:
    data = json.loads(strip_fences(text or ""))
    if not isinstance(data, dict):
        raise ValueError("dossier is not a JSON object")

    entries, dropped = [], 0
    for raw in data.get("entries") or []:
        if not isinstance(raw, dict):
            continue
        who = str(raw.get("who") or "").strip()
        what = str(raw.get("what") or "").strip()
        source = str(raw.get("source") or "").strip()
        kind = str(raw.get("kind") or "").strip().casefold()
        if not (who and what):
            continue
        if not _source_ok(source):
            # The rule the whole stage rests on. An unsourced entry reads
            # exactly like a sourced one once it is in the dossier, so it has
            # to be dropped here rather than flagged for later.
            dropped += 1
            continue
        entries.append({"who": who, "what": what, "source": source,
                        "kind": kind if kind in KINDS else "critic"})
    return {
        "reception": str(data.get("reception") or "").strip(),
        "entries": entries,
        "dropped": dropped,
    }


def stored(row) -> dict | None:
    try:
        raw = row["dossier_json"]
    except (IndexError, KeyError, TypeError):
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) and data.get("entries") else None


def as_brief(data: dict) -> str:
    """The dossier as the outline and script prompts see it."""
    lines = []
    if data.get("reception"):
        lines.append(f"How it landed: {data['reception']}")
    for kind in KINDS:
        group = [e for e in data.get("entries") or [] if e["kind"] == kind]
        if not group:
            continue
        lines.append(f"\n{KIND_LABELS[kind]}:")
        for entry in group:
            lines.append(f"  - {entry['who']}: {entry['what']}")
    return "\n".join(lines)


def corroboration(data: dict | None) -> str:
    """Text the citation check treats as evidence a name was looked up.

    Without this every person the dossier introduces would flag as a possible
    fabrication, and a flag list where most entries are fine is a flag list
    nobody reads.
    """
    if not data:
        return ""
    parts = [data.get("reception") or ""]
    for entry in data.get("entries") or []:
        parts += [entry["who"], entry["what"], entry["source"]]
    return "\n".join(p for p in parts if p)


def research(episode_id: str, cfg: dict) -> None:
    """Stage 'researching'. A no-op unless this episode asked for it."""
    ep = db.get_episode(episode_id)
    if not wanted(ep):
        log.info("episode %s did not ask for research; skipping", episode_id)
        return

    paths = db.paper_paths(episode_id)
    if not paths:
        raise PipelineError(f"no source PDF stored for episode {episode_id}")

    from google.genai import types

    parts = [pdf_part(p) for p in paths]
    user = load_prompt("dossier.md")
    wanted_model = (ep["script_model_wanted"] if ep and ep["script_model_wanted"]
                    else None)

    last = None
    for model in _script_models(cfg, prefer=wanted_model):
        try:
            resp = call_with_retry(
                lambda m=model: client().models.generate_content(
                    model=m, contents=[*parts, user],
                    config=types.GenerateContentConfig(
                        system_instruction="You research how published work was received.",
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                    ),
                ),
                cfg, model, label="dossier",
            )
            record_cost(episode_id, model, resp, cfg, stage="dossier")
            data = _parse(resp.text or "")
            data["grounding"] = collect_grounding(resp)
            break
        except ModelUnusable as e:
            log.warning("dossier model %s unusable (%s); trying the next", model, e)
            last = e
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("dossier from %s did not parse (%s); trying the next", model, e)
            last = e
    else:
        # Not fatal. The dossier makes an episode better; failing to get one
        # should not stop an episode that would otherwise be fine.
        log.error("no usable dossier for %s: %s", episode_id, last)
        db.stage_start(episode_id, "researching:none")
        db.stage_end(episode_id, "researching:none", ok=False,
                     detail=(f"no usable dossier ({last}). The episode continues "
                             "without one; Context will be hedged as it was before."))
        return

    db.update_principal(episode_id, dossier_json=json.dumps(data))
    log.info("episode %s researched: %d entries, %d dropped for want of a source",
             episode_id, len(data["entries"]), data["dropped"])
    if data["dropped"]:
        db.stage_start(episode_id, "researching:unsourced")
        db.stage_end(
            episode_id, "researching:unsourced", ok=False,
            detail=(f"{data['dropped']} finding(s) arrived without a source URL and "
                    "were dropped. An attribution nobody can check is worse than "
                    "a hedge."),
        )
