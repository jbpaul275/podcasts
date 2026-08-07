"""Orchestrator: a linear stage state machine over one episode.

Every stage writes its output (disk and/or DB) and its status to the DB before
the next stage starts, so a crash mid-run is resumable from the last completed
stage — in particular, TTS resumes at the chunk level without re-running
scripting.
"""

import json
import logging
import traceback

import db
from . import PipelineError, assemble, ingest, script, tts

log = logging.getLogger("paperpod.run")

# stage name -> (status while running, callable)
STAGES: list[tuple[str, object]] = [
    ("extracting", ingest.extract_metadata),
    ("scripting", None),  # bound below; needs a post-step
    ("synthesizing", tts.synthesize),
    ("assembling", assemble.assemble),
]
STAGE_NAMES = [s for s, _ in STAGES]

# Reached when a run is asked to stop after a stage -- a script rewrite, which
# deliberately does not go on to spend money on TTS. Not a stage: no worker
# picks it up, it waits for a person.
NEEDS_REVIEW = "needs_review"


def _run_scripting(episode_id: str, cfg: dict) -> None:
    """Write the script -- or rewrite it, if a rewrite is pending.

    The request lives on the episode rather than in the queue so that a process
    killed mid-rewrite resumes the rewrite instead of quietly reverting to a
    plain regeneration and discarding what was asked for.
    """
    ep = db.get_episode(episode_id)
    req = {}
    if ep and ep["rewrite_json"]:
        try:
            req = json.loads(ep["rewrite_json"]) or {}
        except (json.JSONDecodeError, TypeError):
            req = {}

    notes = (req.get("instructions") or "").strip()
    model = req.get("model") or None
    if req.get("mode") == "revise" and notes:
        text = script.revise_script(episode_id, cfg, notes, model=model)
    else:
        text = script.generate_script(episode_id, cfg, instructions=notes or None,
                                      model=model)

    flags = script.citation_flags(text)
    db.set_script(episode_id, text, note=notes or None)
    db.update_episode(episode_id, rewrite_json=None)
    # A rewrite keeps the title a human may have edited; a first pass needs one.
    if not (ep and ep["episode_title"]):
        title = script.generate_title(episode_id, text, cfg)
        if title:
            db.update_episode(episode_id, episode_title=title)
    if flags:
        db.stage_start(episode_id, "scripting:flags")
        db.stage_end(
            episode_id, "scripting:flags", ok=True,
            detail=f"{len(flags)} citation-shaped string(s) flagged for review",
        )


STAGES[1] = ("scripting", _run_scripting)


def resume_stage_for(status: str) -> str:
    """Which stage to (re)start for an episode found in a given status."""
    if status in STAGE_NAMES:
        return status  # it died mid-stage; redo that stage
    return STAGE_NAMES[0]


def run_episode(episode_id: str, cfg: dict, from_stage: str | None = None,
                stop_after: str | None = None) -> None:
    ep = db.get_episode(episode_id)
    if not ep:
        log.error("episode %s vanished before processing", episode_id)
        return

    start = from_stage or resume_stage_for(ep["status"])
    if start not in STAGE_NAMES:
        log.error("unknown stage %r for episode %s", start, episode_id)
        return
    start_idx = STAGE_NAMES.index(start)
    # A rewrite runs scripting alone: the point is to read the new script and
    # iterate before paying for TTS, which is ~97% of an episode.
    end_idx = (STAGE_NAMES.index(stop_after) + 1
               if stop_after in STAGE_NAMES else len(STAGE_NAMES))

    for stage_name, fn in STAGES[start_idx:end_idx]:
        db.update_episode(episode_id, status=stage_name, error=None)
        db.set_progress(episode_id, stage_name)
        db.stage_start(episode_id, stage_name)
        try:
            fn(episode_id, cfg)
        except PipelineError as e:
            db.stage_end(episode_id, stage_name, ok=False, detail=str(e))
            db.mark_failed(episode_id, str(e))
            db.set_progress(episode_id, None)
            log.error("episode %s failed at %s: %s", episode_id, stage_name, e)
            return
        except Exception as e:
            detail = f"{e}\n{traceback.format_exc()[-1500:]}"
            db.stage_end(episode_id, stage_name, ok=False, detail=detail)
            db.mark_failed(episode_id, str(e))
            db.set_progress(episode_id, None)
            log.exception("episode %s crashed at %s", episode_id, stage_name)
            return
        db.stage_end(episode_id, stage_name, ok=True)

    db.set_progress(episode_id, None)
    # Stopping early leaves the episode short of a finished MP3, so it is not
    # "done" -- it is waiting for a person to look at the script.
    if end_idx < len(STAGE_NAMES):
        db.update_episode(episode_id, status=NEEDS_REVIEW)
        log.info("episode %s stopped after %s", episode_id, stop_after)
        return
    db.update_episode(episode_id, status="done")
    log.info("episode %s done", episode_id)
