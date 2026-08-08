"""Positions: several works in, a map of how they stand to each other out.

The stage that exists because of one failure mode. Told "these two papers
disagree, say which is right", a model will say which is right — including when
the papers are answering different questions and the disagreement is an
artefact of comparing unlike things. Different outcome variable, different
population, different decade, the same word meaning two things. A confident
adjudication of a conflict that was never there is worse than no episode.

So the comparison is located before it is written. The stage returns each
work's claim in its own terms, an explicit commensurability check, and the crux
a real disagreement rests on — and it is allowed to come back saying these do
not actually conflict, which is usually the better episode.

It also names the relation, which is what picks the arc. The wizard defaults to
`auto` and this is what `auto` resolves to; anything chosen by hand is left
alone, because a person who has read both papers knows something this stage
does not.

A no-op for a single-paper episode: there is nothing to stand in relation to.
"""

import json
import logging

import db
from config import load_prompt
from . import ModelUnusable, PipelineError
from . import arc as arc_mod
from .gemini import call_with_retry, client, pdf_part, record_cost, strip_fences
from .script import _script_config, _script_models

log = logging.getLogger("paperpod.positions")


def wanted(episode_id: str) -> bool:
    return len(db.papers_for(episode_id)) > 1


def _parse(text: str, expected: int) -> dict:
    data = json.loads(strip_fences(text or ""))
    if not isinstance(data, dict):
        raise ValueError("positions is not a JSON object")

    papers = []
    for raw in data.get("papers") or []:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()
        claim = str(raw.get("claim") or "").strip()
        if not (title and claim):
            continue
        papers.append({
            "title": title,
            "claim": claim,
            "construct": str(raw.get("construct") or "").strip(),
            "population": str(raw.get("population") or "").strip(),
            "period": str(raw.get("period") or "").strip(),
        })
    if len(papers) < 2:
        raise ValueError("positions covered fewer than two works")
    if len(papers) != expected:
        # Not fatal, but worth having in the log: a work the stage did not see
        # is a work the episode will be thin about.
        log.warning("positions covered %d works; %d were attached",
                    len(papers), expected)

    return {
        "question": str(data.get("question") or "").strip(),
        "papers": papers,
        # Absent means "did not check", and the whole point of the field is
        # that it was checked. Default to the cautious reading.
        "commensurable": bool(data.get("commensurable")),
        "why": str(data.get("why") or "").strip(),
        "crux": str(data.get("crux") or "").strip(),
        "relation": arc_mod.clean_relation(data.get("relation")),
        "relation_why": str(data.get("relation_why") or "").strip(),
    }


def stored(row) -> dict | None:
    try:
        raw = row["positions_json"]
    except (IndexError, KeyError, TypeError):
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) and data.get("papers") else None


def as_brief(data: dict) -> str:
    """The map as the outline and script prompts see it."""
    lines = []
    if data.get("question"):
        lines.append(f"The question all of these answer: {data['question']}")
    for entry in data.get("papers") or []:
        lines.append(f"\n{entry['title']}")
        lines.append(f"  claims: {entry['claim']}")
        for field, label in (("construct", "measures"), ("population", "covers"),
                             ("period", "period")):
            if entry.get(field):
                lines.append(f"  {label}: {entry[field]}")
    if data.get("commensurable"):
        lines.append(f"\nThese are answering the same question. {data.get('why', '')}".rstrip())
        if data.get("crux"):
            lines.append(f"The disagreement rests on: {data['crux']}")
    else:
        # Stated as an instruction rather than a fact, because it changes what
        # the episode is: the interesting thing is now why it looked like a
        # disagreement, and an adjudication here would be adjudicating nothing.
        lines.append(
            "\nTHESE ARE NOT ANSWERING THE SAME QUESTION. "
            f"{data.get('why', '')}".rstrip())
        lines.append(
            "Do not adjudicate between them. The episode's job is to show what "
            "each one actually establishes and why the two get mistaken for a "
            "disagreement.")
    return "\n".join(lines)


def build_positions(episode_id: str, cfg: dict) -> None:
    """Stage 'positioning'. Locates the comparison and settles the arc."""
    if not wanted(episode_id):
        log.info("episode %s is about one work; nothing to position", episode_id)
        return

    paths = db.paper_paths(episode_id)
    if len(paths) < 2:
        raise PipelineError(
            f"episode {episode_id} names several papers but only {len(paths)} "
            "PDF(s) are on disk")

    ep = db.get_episode(episode_id)
    parts = [pdf_part(p) for p in paths]
    user = load_prompt("positions.md")
    wanted_model = (ep["script_model_wanted"] if ep and ep["script_model_wanted"]
                    else None)

    last = None
    for model in _script_models(cfg, prefer=wanted_model):
        try:
            resp = call_with_retry(
                lambda m=model: client().models.generate_content(
                    model=m, contents=[*parts, user],
                    config=_script_config(cfg, "You compare academic works."),
                ),
                cfg, model, label="positions",
            )
            record_cost(episode_id, model, resp, cfg, stage="positions")
            data = _parse(resp.text or "", len(paths))
            break
        except ModelUnusable as e:
            log.warning("positions model %s unusable (%s); trying the next", model, e)
            last = e
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("positions from %s did not parse (%s); trying the next",
                        model, e)
            last = e
    else:
        # Fatal, unlike research. An episode about several papers with no idea
        # how they relate would be written on the model's first impression of
        # two documents at once, which is the thing this stage exists to stop.
        raise PipelineError(f"could not work out how these works relate: {last}")

    fields = {"positions_json": json.dumps(data)}
    chosen = (ep["relation"] if ep else None) or arc_mod.AUTO
    if chosen == arc_mod.AUTO:
        fields["relation"] = data["relation"]
        log.info("episode %s: relation resolved to %s (%s)", episode_id,
                 data["relation"], data["relation_why"] or "no reason given")
    elif chosen != data["relation"]:
        # Kept, not overridden: whoever picked it has read both papers. But
        # recorded, because a disagreement here is worth seeing on the page.
        db.stage_start(episode_id, "positioning:relation")
        db.stage_end(
            episode_id, "positioning:relation", ok=False,
            detail=(f"you asked for {chosen}; reading the papers suggests "
                    f"{data['relation']}. {data['relation_why']} Using {chosen}."))

    db.update_episode(episode_id, **fields)
    log.info("episode %s positioned: %d works, commensurable=%s", episode_id,
             len(data["papers"]), data["commensurable"])
    if not data["commensurable"]:
        db.stage_start(episode_id, "positioning:incommensurable")
        db.stage_end(
            episode_id, "positioning:incommensurable", ok=False,
            detail=(f"these works are not answering the same question. "
                    f"{data['why']} The episode will explain the mismatch "
                    "rather than pick a winner."))
