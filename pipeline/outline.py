"""Outline: paper in, beat sheet out. The stage that decides how long an
episode is, and keeps the script from wandering once it starts writing.

Two problems, one artifact.

A fixed target length made every paper the same size, which meant padding a
methods note and compressing a paper with three experiments. Here the length is
a *consequence*: the beat sheet says what there is to say, and the episode is
as long as that comes to. Nobody picks a duration in advance.

And a long script generated in one pass drifts -- the model loses track of an
arc it is holding only in its head. The beat sheet is that arc written down,
handed to the writer as a map, which is the standard fix and gets more valuable
exactly as episodes get longer.

The beat sheet is stored on the episode and shown in the UI, so a length that
looks wrong is a thing you can read the reasoning for rather than guess at.
"""

import json
import logging

import db
from config import load_prompt
from . import ModelUnusable, PipelineError
from .gemini import call_with_retry, client, pdf_part, record_cost, strip_fences
from . import arc as arc_mod
from . import dossier as dossier_mod
from .script import _script_config, _script_models

log = logging.getLogger("paperpod.outline")

DEFAULT_WPM = 160
DEFAULT_RANGE = (5, 30)

# A beat this small is a fragment rather than a beat; the prompt says so, and
# this is what happens when it is ignored.
MIN_BEAT_WORDS = 40


def length_policies(cfg: dict) -> dict[str, list]:
    raw = (cfg.get("script", {}).get("lengths") or {})
    out = {}
    for name, span in raw.items():
        try:
            lo, hi = int(span[0]), int(span[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 < lo <= hi:
            out[name] = [lo, hi]
    return out or {"auto": list(DEFAULT_RANGE)}


def policy_range(cfg: dict, policy: str | None) -> tuple[int, int]:
    policies = length_policies(cfg)
    span = policies.get((policy or "").strip()) or policies.get("auto") \
        or next(iter(policies.values()))
    return span[0], span[1]


def words_per_minute(cfg: dict) -> int:
    try:
        return max(60, int(cfg.get("script", {}).get("words_per_minute", DEFAULT_WPM)))
    except (TypeError, ValueError):
        return DEFAULT_WPM


def _parse(text: str, segments: list[str]) -> dict:
    data = json.loads(strip_fences(text or ""))
    if not isinstance(data, dict):
        raise ValueError("outline is not a JSON object")
    beats = data.get("beats")
    if not isinstance(beats, list) or not beats:
        raise ValueError("outline has no beats")

    clean = []
    for raw in beats:
        if not isinstance(raw, dict):
            continue
        segment = str(raw.get("segment") or "").strip()
        # Segment names are matched case-insensitively but stored canonically:
        # the script prompt groups by them, and "Cold Open" vs "Cold open"
        # would silently make two segments out of one.
        match = next((s for s in segments if s.casefold() == segment.casefold()), None)
        covers = str(raw.get("covers") or "").strip()
        if not match or not covers:
            continue
        try:
            words = int(raw.get("words") or 0)
        except (TypeError, ValueError):
            words = 0
        facts = [str(f).strip() for f in (raw.get("facts") or []) if str(f).strip()]
        clean.append({"segment": match, "covers": covers, "facts": facts,
                      "words": max(MIN_BEAT_WORDS, words)})
    if not clean:
        raise ValueError("outline had beats but none were usable")
    return {"why": str(data.get("why") or "").strip(), "beats": clean}


def planned_words(outline: dict) -> int:
    """What the beats add up to. Trusted over the model's own `minutes`, which
    is a summary of this rather than a separate judgement."""
    return sum(int(b.get("words") or 0) for b in outline.get("beats") or [])


def resolve_length(outline: dict, cfg: dict, policy: str | None) -> tuple[int, str]:
    """Target words for the script, and a note if the plan had to be clamped.

    Clamped rather than trusted: a model asked for a number will occasionally
    return one far outside the range it was given, and an episode nobody
    budgeted for costs real money and real daily quota.
    """
    wpm = words_per_minute(cfg)
    lo, hi = policy_range(cfg, policy)
    words = planned_words(outline)
    floor, ceiling = lo * wpm, hi * wpm
    if words < floor:
        return floor, (f"the outline planned {words} words ({words / wpm:.0f} min), "
                       f"below the {lo} minute floor; raised to {floor}")
    if words > ceiling:
        return ceiling, (f"the outline planned {words} words ({words / wpm:.0f} min), "
                         f"above the {hi} minute ceiling; cut to {ceiling}")
    return words, ""


def as_brief(outline: dict, kind: str = arc_mod.DEFAULT) -> str:
    """The beat sheet as the script prompt sees it."""
    lines = []
    for segment in arc_mod.segments(kind):
        beats = [b for b in outline.get("beats") or [] if b["segment"] == segment]
        if not beats:
            continue
        lines.append(f"{segment}:")
        for beat in beats:
            lines.append(f"  - ({beat['words']} words) {beat['covers']}")
            for fact in beat["facts"]:
                lines.append(f"      must land: {fact}")
    return "\n".join(lines)


def stored(row) -> dict | None:
    try:
        raw = row["outline_json"]
    except (IndexError, KeyError, TypeError):
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) and data.get("beats") else None


def build_outline(episode_id: str, cfg: dict) -> None:
    """Stage 'outlining'. Plans the episode and sets its length."""
    paths = db.paper_paths(episode_id)
    if not paths:
        raise PipelineError(f"no source PDF stored for episode {episode_id}")
    ep = db.get_episode(episode_id)
    policy = (ep["length_policy"] if ep and ep["length_policy"] else "auto")
    lo, hi = policy_range(cfg, policy)

    kind = arc_mod.kind_of(ep)
    research = dossier_mod.stored(ep)
    user = (
        load_prompt("outline.md")
        .replace("$ARC", arc_mod.text(kind))
        .replace("$MIN_MINUTES", str(lo))
        .replace("$MAX_MINUTES", str(hi))
        .replace("$WORDS_PER_MINUTE", str(words_per_minute(cfg)))
    )
    if research:
        # The reception is not in the work, so without this the Context beats
        # can only be planned as "say something about the literature".
        user += ("\n\nRESEARCH ON HOW THIS WORK LANDED — plan the Context beats "
                 "around what is actually here, and use it anywhere else it "
                 "helps:\n" + dossier_mod.as_brief(research))
    parts = [pdf_part(p) for p in paths]
    wanted = (ep["script_model_wanted"] if ep and ep["script_model_wanted"] else None)

    last = None
    for model in _script_models(cfg, prefer=wanted):
        try:
            resp = call_with_retry(
                lambda m=model: client().models.generate_content(
                    model=m, contents=[*parts, user],
                    config=_script_config(cfg, "You plan podcast episodes."),
                ),
                cfg, model, label="outline",
            )
            record_cost(episode_id, model, resp, cfg, stage="outline")
            outline = _parse(resp.text or "", arc_mod.segments(kind))
            break
        except ModelUnusable as e:
            log.warning("outline model %s unusable (%s); trying the next", model, e)
            last = e
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("outline from %s did not parse (%s); trying the next", model, e)
            last = e
    else:
        raise PipelineError(f"could not produce an outline: {last}")

    target, clamp = resolve_length(outline, cfg, policy)
    outline["kind"] = kind
    db.update_episode(episode_id, outline_json=json.dumps(outline),
                      target_words=target)
    log.info("episode %s outlined: %d beats, %d words (~%d min)", episode_id,
             len(outline["beats"]), target, round(target / words_per_minute(cfg)))
    if clamp:
        # Recorded rather than applied quietly: a length nobody asked for is
        # exactly the thing to be able to see afterwards.
        db.stage_start(episode_id, "outlining:clamped")
        db.stage_end(episode_id, "outlining:clamped", ok=False, detail=clamp)
        log.warning("episode %s length clamped: %s", episode_id, clamp)
