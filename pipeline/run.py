"""Orchestrator: a linear stage state machine over one episode.

Every stage writes its output (disk and/or DB) and its status to the DB before
the next stage starts, so a crash mid-run is resumable from the last completed
stage — in particular, TTS resumes at the chunk level without re-running
scripting.
"""

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


def _run_scripting(episode_id: str, cfg: dict) -> None:
    text = script.generate_script(episode_id, cfg)
    flags = script.citation_flags(text)
    db.update_episode(episode_id, script_md=text)
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


def run_episode(episode_id: str, cfg: dict, from_stage: str | None = None) -> None:
    ep = db.get_episode(episode_id)
    if not ep:
        log.error("episode %s vanished before processing", episode_id)
        return

    start = from_stage or resume_stage_for(ep["status"])
    if start not in STAGE_NAMES:
        log.error("unknown stage %r for episode %s", start, episode_id)
        return
    start_idx = STAGE_NAMES.index(start)

    for stage_name, fn in STAGES[start_idx:]:
        db.update_episode(episode_id, status=stage_name, error=None)
        db.stage_start(episode_id, stage_name)
        try:
            fn(episode_id, cfg)
        except PipelineError as e:
            db.stage_end(episode_id, stage_name, ok=False, detail=str(e))
            db.update_episode(episode_id, status="failed", error=str(e))
            log.error("episode %s failed at %s: %s", episode_id, stage_name, e)
            return
        except Exception as e:
            detail = f"{e}\n{traceback.format_exc()[-1500:]}"
            db.stage_end(episode_id, stage_name, ok=False, detail=detail)
            db.update_episode(episode_id, status="failed", error=str(e))
            log.exception("episode %s crashed at %s", episode_id, stage_name)
            return
        db.stage_end(episode_id, stage_name, ok=True)

    db.update_episode(episode_id, status="done")
    log.info("episode %s done", episode_id)
