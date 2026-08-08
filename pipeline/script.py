"""Script generation: PDF in (natively), speaker-tagged Markdown dialogue out.

Also home of the citation-flag validator, which regex-scans the finished script
for citation-shaped strings and surfaces them for human review in the UI.
"""

import json
import logging
import re

import db
from config import load_prompt
from . import ModelUnusable, PipelineError
from . import arc as arc_mod
from .gemini import (call_with_retry, client, pdf_part, record_cost,
                     resolved_model, strip_fences)
from prose import title_case

log = logging.getLogger("paperpod.script")

LINE_RE = re.compile(r"^HOST_[AB]:\s+\S")

# Capitalized words that are almost always sentence-position artifacts rather
# than proper nouns. Kept deliberately small: over-flagging costs a glance,
# under-flagging lets a fabricated citation through.
_COMMON_SENTENCE_STARTERS = {
    "The", "But", "And", "So", "That", "This", "There", "They", "Them", "Their",
    "These", "Those", "What", "When", "Where", "Why", "How", "Who", "If", "In",
    "On", "At", "By", "For", "From", "To", "With", "It", "Its", "We", "Our",
    "You", "Your", "He", "She", "His", "Her", "Not", "Now", "Then", "Also",
    "Well", "Yeah", "Okay", "Right", "Wait", "Sure", "Look", "Because",
    "Before", "After", "Since", "During", "Until", "While", "Though",
    "Although", "Even", "Just", "Only", "Still", "Yet", "Again", "About",
    "Around", "Over", "Under", "Between", "Both", "Each", "Every", "Some",
    "Most", "Many", "More", "Less", "All", "Any", "Another", "Such", "Here",
    "One", "Two", "Three", "Let", "Yes", "No", "Maybe", "Actually", "Honestly",
    "Basically", "Anyway", "Roughly", "Nearly", "Almost", "Was", "Were", "Is",
    "Are", "Do", "Does", "Did", "Can", "Could", "Would", "Should", "Will",
    "Might", "Must", "Have", "Has", "Had", "Host",
}

# Institutions and bodies. They are proper nouns and they sit near years
# constantly ("Congress wrote into the 1964 statute"), but they are never the
# author of a fabricated citation.
_NON_AUTHOR_ENTITIES = {
    "Congress", "Senate", "Parliament", "Court", "Supreme", "Government",
    "Federal", "Reserve", "Treasury", "Bureau", "Department", "Ministry",
    "Commission", "Council", "Committee", "Agency", "Administration",
    "Union", "Party", "Republicans", "Democrats", "Labour", "Republican",
    "Democratic", "Census", "Survey", "Act", "Amendment", "Constitution",
    "America", "American", "British", "European", "Union's", "States",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "Spring", "Summer", "Autumn", "Fall", "Winter",
}

_NOT_AUTHOR_TOKENS = _COMMON_SENTENCE_STARTERS | _NON_AUTHOR_ENTITIES

# Name (2004) / Name and Name (2004) / Name et al. (2004)
_NAME_YEAR_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+|\s+et\s+al\.?)?\s*\(\s*(?:19|20)\d{2}\s*\)"
)
_ET_AL_RE = re.compile(r"\b[A-Z][a-z]+\s+et\s+al\b\.?")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_WORD_RE = re.compile(r"\b[\w'’-]+\b")

# Lower number wins when two patterns overlap.
_PRIORITY = {"name-year cite": 0, "et al.": 1, "proper noun near year": 2}

# Statutes, treaties and the like are named entities that happen to carry a
# year. They are never academic citations, so flagging them is pure noise.
_STATUTE_RE = re.compile(
    r"\b(?:Act|Amendment|Amendments|Law|Bill|Treaty|Convention|Accord|Accords"
    r"|Protocol|Directive|Code|Constitution|Doctrine|Decree|Charter|Ruling"
    r"|Resolution|Reform|Program|Programme|Initiative|Census|Survey|Panel"
    r"|Recession|Crisis|War|Olympics|Election)\b"
)

_LEADING_NAME_RE = re.compile(r"^(?:[A-Z][\w'’-]*(?:\s+(?:and|&|of|for|de|van|von)\s+)?)+")

# How many words either side of a bare year can carry the name.
_YEAR_WINDOW = 2


def _depossess(token: str) -> str:
    """"Acemoglu's" -> "Acemoglu". The paper writes the bare surname."""
    return re.sub(r"['’]s\b", "", token)


def _looks_like_an_author(token: str) -> bool:
    # Strip a possessive so "Acemoglu's" is tested as a name and "There's" is
    # recognised as the stopword it is.
    bare = _depossess(token)
    return (
        len(bare) > 2
        and bare[:1].isupper()
        and bare not in _NOT_AUTHOR_TOKENS
    )


def _script_config(cfg: dict, system: str):
    """Reasoning effort and web grounding for the script call.

    Scripting is a couple of percent of an episode's cost, so thinking is cheap
    to buy here; a dense paper with an identification strategy is exactly what
    it helps with.
    """
    from google.genai import types

    scfg = cfg.get("script", {})
    kwargs: dict = {"system_instruction": system}

    level = (scfg.get("thinking_level") or "").strip().upper()
    budget = scfg.get("thinking_budget")
    if level or budget:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=level or None,
            thinking_budget=int(budget) if budget else None,
        )

    if scfg.get("grounding"):
        kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    return types.GenerateContentConfig(**kwargs)


def collect_grounding(resp) -> dict:
    """Queries the model ran and the pages it drew on.

    Recorded because grounding relaxes the rule that every claim comes from the
    PDF: an outside claim is only acceptable if it can be traced, and this is
    the trace.
    """
    out: dict = {"queries": [], "sources": []}
    try:
        meta = resp.candidates[0].grounding_metadata
    except (AttributeError, IndexError, TypeError):
        return out
    if meta is None:
        return out

    out["queries"] = list(getattr(meta, "web_search_queries", None) or [])
    seen = set()
    for chunk in getattr(meta, "grounding_chunks", None) or []:
        web = getattr(chunk, "web", None)
        if web is None or not (web.uri or web.title):
            continue
        key = web.uri or web.title
        if key in seen:
            continue
        seen.add(key)
        out["sources"].append({
            "title": web.title or web.domain or web.uri,
            "uri": web.uri or "",
            "domain": web.domain or "",
        })
    return out


def _script_models(cfg: dict, prefer: str | None = None) -> list[str]:
    """Preferred model, then the fallback. A preview Pro often returns
    limit: 0, and degrading to Flash beats failing the episode outright.

    `prefer` overrides the configured primary — used when a rewrite names a
    model explicitly. The fallback still applies, because a hand-picked model
    is no less likely to be out of quota than the configured one.
    """
    primary = prefer or cfg["models"]["script"]
    fallback = (cfg.get("script", {}).get("fallback_model") or "").strip()
    return [primary, fallback] if fallback and fallback != primary else [primary]


def script_choices(cfg: dict) -> list[str]:
    """Models offered for writing or rewriting a script, default first."""
    listed = [m for m in cfg.get("script", {}).get("models", []) if m]
    default = cfg["models"]["script"]
    fallback = (cfg.get("script", {}).get("fallback_model") or "").strip()
    out = [default] + [m for m in listed + [fallback] if m and m != default]
    seen, uniq = set(), []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


def _target_words(ep, cfg: dict) -> int:
    """The length the outline settled on, or the configured default.

    The fallback matters: episodes built before the outline stage existed, and
    any retry from `scripting` that skips it, still need a number.
    """
    try:
        stored = ep["target_words"] if ep else None
    except (IndexError, KeyError, TypeError):
        stored = None
    return int(stored) if stored else int(cfg["script"]["target_words"])


def _dossier_brief(ep) -> str:
    """The research, or a line saying there is none."""
    from . import dossier as dossier_mod

    data = dossier_mod.stored(ep) if ep is not None else None
    if not data:
        return ("No research was done for this episode. Hedge every claim about "
                "outside work, as your instructions require.")
    return (
        "Every entry below was looked up and carries a source, so you may state "
        "it plainly rather than hedging. Describe positions in your own words: "
        "do NOT quote anyone here, and quote only from the attached work.\n\n"
        + dossier_mod.as_brief(data)
    )


def _positions_brief(ep) -> str:
    """How the works stand to each other, for an episode about several."""
    from . import positions as positions_mod

    data = positions_mod.stored(ep) if ep is not None else None
    if not data:
        return ""
    return (
        "\n\nHOW THESE WORKS STAND TO EACH OTHER — worked out by reading all of "
        "them side by side, before this script was planned. Every position below "
        "is the strongest version of itself; argue against the strong version or "
        "not at all.\n\n"
        + positions_mod.as_brief(data)
    )


def _brief(ep) -> str:
    """The beat sheet, or a line saying there is not one."""
    from . import outline as outline_mod

    plan = outline_mod.stored(ep) if ep is not None else None
    if not plan:
        return ("No beat sheet for this episode -- follow the arc in your "
                "instructions and choose the emphasis yourself.")
    why = f"Why this length: {plan['why']}\n\n" if plan.get("why") else ""
    return why + outline_mod.as_brief(plan)


def revise_script(episode_id: str, cfg: dict, instructions: str,
                  model: str | None = None) -> str:
    """Rewrite the stored script to an editor's notes, keeping the rest intact.

    The paper goes along with it: a note asking for more on some part of the
    paper cannot be satisfied from the script alone, and sending the PDF is
    also what keeps the no-fabrication constraint enforceable.
    """
    ep = db.get_episode(episode_id)
    if not ep or not (ep["script_md"] or "").strip():
        raise PipelineError("no script to revise; run the scripting stage first")
    if not instructions.strip():
        raise PipelineError("no revision notes given")

    target = _target_words(ep, cfg)
    user = (
        load_prompt("script_revise.md")
        .replace("$INSTRUCTIONS", instructions.strip())
        .replace("$TARGET_WORDS", str(target))
        .replace("$SCRIPT", ep["script_md"])
    )
    return _write_script(episode_id, cfg, user, model, label="script revision")


def generate_script(episode_id: str, cfg: dict, instructions: str | None = None,
                    model: str | None = None) -> str:
    """Write a script from the paper, ignoring whatever script exists."""
    ep = db.get_episode(episode_id)
    target = _target_words(ep, cfg)
    user = (
        load_prompt("script_user.md")
        .replace("$TARGET_WORDS", str(target))
        .replace("$MIN_WORDS", str(int(target * 0.875)))
        .replace("$MAX_WORDS", str(int(target * 1.125)))
        .replace("$OUTLINE", _brief(ep))
        .replace("$RESEARCH", _dossier_brief(ep))
    )
    user += _positions_brief(ep)
    angle = ""
    try:
        angle = (ep["angle"] or "").strip() if ep else ""
    except (IndexError, KeyError, TypeError):
        angle = ""
    if angle:
        user += (
            "\n\nTHE ANGLE ASKED FOR — what this episode is meant to be about. "
            "It sets the emphasis; it does not license going beyond the works or "
            "past the hard constraints:\n" + angle
        )
    if instructions and instructions.strip():
        user += (
            "\n\nADDITIONAL DIRECTION FROM THE EDITOR — these take precedence "
            "over the default emphasis, but not over the hard constraints:\n"
            + instructions.strip()
        )
    return _write_script(episode_id, cfg, user, model, label="script generation")


def _write_script(episode_id: str, cfg: dict, user: str, model: str | None,
                  label: str) -> str:
    paths = db.paper_paths(episode_id)
    if not paths:
        raise PipelineError(f"no source PDF stored for episode {episode_id}")

    system = load_prompt("script_system.md").replace(
        "$ARC", arc_mod.text(arc_mod.of(db.get_episode(episode_id), len(paths))))
    if cfg.get("script", {}).get("grounding"):
        system += "\n\n" + load_prompt("script_grounding.md")
    gen_cfg = _script_config(cfg, system)
    parts = [pdf_part(p) for p in paths]

    candidates = _script_models(cfg, prefer=model)
    resp = None
    for i, model in enumerate(candidates):
        try:
            resp = call_with_retry(
                lambda m=model: client().models.generate_content(
                    model=m, contents=[*parts, user], config=gen_cfg
                ),
                cfg, model, label=label,
            )
            break
        except ModelUnusable as e:
            # Quota exhausted or the model retired: either way this model can
            # never serve the request, and the fallback is the whole point.
            if i + 1 >= len(candidates):
                raise
            log.warning("script model %s unusable (%s); falling back to %s",
                        model, e, candidates[i + 1])
            db.stage_start(episode_id, "scripting:fallback")
            db.stage_end(
                episode_id, "scripting:fallback", ok=False,
                detail=(f"{model} could not be used ({e}) so the script was "
                        f"written by {candidates[i + 1]} instead — expect lower "
                        f"quality on a technical paper."),
            )

    record_cost(episode_id, model, resp, cfg, stage="script")
    # Store what actually ran. "gemini-pro-latest" on an episode from March
    # tells you nothing about which model wrote it.
    db.update_episode(episode_id, script_model=resolved_model(resp, model),
                      grounding_json=json.dumps(collect_grounding(resp)))
    script = _clean(resp.text or "")
    violations = _format_violations(script)

    if violations:
        # One retry, with the format violations quoted back at the model.
        retry_msg = (
            user
            + "\n\nYour previous attempt violated the output format. Every line must be "
            "`HOST_A: ...` or `HOST_B: ...` with no other text. These lines were invalid:\n"
            + "\n".join(f"  {v!r}" for v in violations[:10])
            + "\nRegenerate the full script in the correct format."
        )
        resp = call_with_retry(
            lambda: client().models.generate_content(
                model=model, contents=[part, retry_msg], config=gen_cfg
            ),
            cfg, model, label=f"{label} (format retry)",
        )
        db.update_episode(episode_id, grounding_json=json.dumps(collect_grounding(resp)))
        record_cost(episode_id, model, resp, cfg, stage="script")
        script = _clean(resp.text or "")
        violations = _format_violations(script)
        if violations:
            raise PipelineError(
                "script failed speaker-format validation after one retry; first bad "
                f"lines: {violations[:5]!r}"
            )

    if not script.strip():
        raise PipelineError("model returned an empty script")
    return script


def generate_title(episode_id: str, script: str, cfg: dict) -> str | None:
    """Title the episode from its own script, so the title reflects the angle
    the hosts actually took rather than the paper's academic title."""
    from google.genai import types

    ep = db.get_episode(episode_id)
    # The same short, cheap model the metadata came from: titling is the same
    # kind of job, and two different models for one episode would be a puzzle
    # rather than a choice.
    model = (ep["metadata_model"] if ep and ep["metadata_model"]
             else cfg["models"]["metadata"])
    try:
        resp = call_with_retry(
            lambda: client().models.generate_content(
                model=model,
                contents=load_prompt("episode_title.md") + script,
                config=types.GenerateContentConfig(max_output_tokens=8000),
            ),
            cfg, model, label="episode title",
        )
        record_cost(episode_id, model, resp, cfg, stage="title")
    except Exception as e:
        # A missing title is cosmetic; never fail the episode over it.
        log.warning("episode title generation failed: %s", e)
        return None

    title = strip_fences(resp.text or "").strip().strip('"“”').strip()
    title = " ".join(title.splitlines()[0].split()) if title else ""
    if not title or len(title) > 120:
        return None
    # The prompt asks for Title Case; this makes it so. A model follows a
    # capitalization instruction most of the time, and "most of the time" is
    # precisely the failure -- one title in sentence case beside one in title
    # case reads as sloppiness. Applied only to generated titles: a title typed
    # by a person is left exactly as typed.
    return title_case(title)


def _clean(text: str) -> str:
    """Strip code fences and markdown emphasis; drop empty decoration lines."""
    text = strip_fences(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip() not in {"---", "***"})


def _format_violations(script: str) -> list[str]:
    return [
        ln for ln in script.splitlines() if ln.strip() and not LINE_RE.match(ln.strip())
    ]


def parse_turns(script: str) -> list[tuple[str, str]]:
    """Parse a validated script into [(speaker, text), ...]."""
    turns = []
    for ln in script.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        speaker, _, text = ln.partition(":")
        turns.append((speaker.strip(), text.strip()))
    return turns


def _normalize(text: str) -> str:
    return " ".join(text.replace("’", "'").split()).casefold()


def appears_in_paper(flag_text: str, paper_text_normalized: str) -> bool:
    """Whether a flagged string traces back to the source PDF.

    The whole phrase rarely appears verbatim -- the hosts paraphrase -- so the
    test is whether the *name* does. "William Gould for the 1969" is fine if
    the paper mentions Gould anywhere, and a fabrication if it does not.
    """
    if _normalize(flag_text) in paper_text_normalized:
        return True
    # The hosts say "Gould's 1969 decision"; the paper says "Gould". Match on
    # the bare name or every possessive citation reads as uncorroborated.
    flag_text = _depossess(flag_text)
    if _normalize(flag_text) in paper_text_normalized:
        return True
    # Strip the trailing year and any connective tail, leaving the name.
    stem = re.sub(r"\s*\(?\b(?:19|20)\d{2}\b\)?\s*$", "", flag_text).strip()
    stem = re.sub(r"\s+(?:in|of|for|from|at|the|and|by|to)$", "", stem).strip()
    if stem and _normalize(stem) in paper_text_normalized:
        return True
    m = _LEADING_NAME_RE.match(stem or flag_text)
    if m:
        name = m.group(0).strip()
        # A single short token is too weak to prove anything either way.
        if len(name) > 3 and _normalize(name) in paper_text_normalized:
            return True

    # Last resort: every distinguishing name in the citation appears somewhere
    # in the corpus, even if not as one phrase. Needed for web grounding, where
    # a source is titled after the paper rather than its authors, so
    # "Card and Krueger (1994)" never appears verbatim but both surnames do.
    names = [
        w for w in _WORD_RE.findall(flag_text)
        if len(w) > 2 and w[:1].isupper() and w not in _NOT_AUTHOR_TOKENS
    ]
    if names and all(_normalize(n) in paper_text_normalized for n in names):
        return True
    return False


def citation_flags(script: str, paper_text: str | None = None,
                   grounding_text: str | None = None) -> list[dict]:
    """Scan for citation-shaped patterns and return them for human review.

    Never auto-fails: the point is to put every string that *could* be a
    fabricated citation in front of a person, so a bare year sitting near a
    proper noun counts even though most such hits are innocent.

    When `paper_text` is supplied, each flag also carries `in_paper`, which is
    what separates a real fabrication from a name the paper actually uses.

    With grounding on, the model may legitimately cite work absent from the PDF,
    so `grounding_text` (the titles and domains it actually consulted) counts as
    corroboration too. Each flag records which corpus vouched for it in
    `source`: "paper", "web", or nothing at all.
    """
    flags: list[dict] = []
    for lineno, line in enumerate(script.splitlines(), start=1):
        # Drop the "HOST_A:" tag so the speaker name never flags itself.
        _, sep, body = line.partition(":")
        if not sep:
            body = line

        spans: list[tuple[int, int, str, str]] = []
        for m in _NAME_YEAR_RE.finditer(body):
            spans.append((m.start(), m.end(), "name-year cite", m.group(0)))
        for m in _ET_AL_RE.finditer(body):
            spans.append((m.start(), m.end(), "et al.", m.group(0)))

        # A bare year is suspicious when a proper noun sits right against it.
        # Scanning outward from the year (rather than forward from a capital)
        # avoids the non-overlapping-match trap, where a leading "The" would
        # otherwise swallow the real proper noun behind it.
        #
        # The window is deliberately tight. At four words it swept up ordinary
        # conversation -- "a Tuesday afternoon back in 2018" -- because any
        # capitalised word loosely near a year matched. A citation keeps the
        # name against the year: "Acemoglu's 2001 paper", "a 2019 study by Roe".
        for m in _YEAR_RE.finditer(body):
            hit = None
            for w in list(_WORD_RE.finditer(body[: m.start()]))[-_YEAR_WINDOW:]:
                if _looks_like_an_author(w.group(0)):
                    hit = (w.start(), m.end())
                    break
            if hit is None:
                # Names also follow the year: "the 2019 Roe paper".
                for w in list(_WORD_RE.finditer(body[m.end():]))[:_YEAR_WINDOW]:
                    if _looks_like_an_author(w.group(0)):
                        hit = (m.start(), m.end() + w.end())
                        break
            if hit is not None:
                spans.append((hit[0], hit[1], "proper noun near year",
                              body[hit[0]:hit[1]]))

        # Collapse overlaps: a full name-year cite subsumes the weaker hits
        # inside it, so each real citation is reported once.
        spans.sort(key=lambda s: (_PRIORITY[s[2]], s[0]))
        kept: list[tuple[int, int, str, str]] = []
        for span in spans:
            if any(span[0] < k[1] and k[0] < span[1] for k in kept):
                continue
            kept.append(span)

        for start, _end, kind, text in sorted(kept):
            text = text.strip()
            # A statute or named event carrying a year is not a citation.
            if kind == "proper noun near year" and _STATUTE_RE.search(text):
                continue
            flags.append({"line": lineno, "kind": kind, "text": text})

    if paper_text is not None or grounding_text is not None:
        paper_norm = _normalize(paper_text or "")
        web_norm = _normalize(grounding_text or "")
        for flag in flags:
            if paper_norm and appears_in_paper(flag["text"], paper_norm):
                flag["in_paper"], flag["source"] = True, "paper"
            elif web_norm and appears_in_paper(flag["text"], web_norm):
                flag["in_paper"], flag["source"] = True, "web"
            else:
                flag["in_paper"], flag["source"] = False, None
    return flags
