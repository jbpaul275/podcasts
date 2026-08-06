"""Script generation: PDF in (natively), speaker-tagged Markdown dialogue out.

Also home of the citation-flag validator, which regex-scans the finished script
for citation-shaped strings and surfaces them for human review in the UI.
"""

import logging
import re

import db
from config import PAPERS_DIR, load_prompt
from . import PipelineError
from .gemini import client, pdf_part, record_cost, strip_fences

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


def generate_script(episode_id: str, cfg: dict) -> str:
    pdf_path = PAPERS_DIR / f"{episode_id}.pdf"
    if not pdf_path.exists():
        raise PipelineError(f"missing source PDF {pdf_path}")

    from google.genai import types

    model = cfg["models"]["script"]
    target = cfg["script"]["target_words"]
    system = load_prompt("script_system.md")
    user = (
        load_prompt("script_user.md")
        .replace("$TARGET_WORDS", str(target))
        .replace("$MIN_WORDS", str(int(target * 0.875)))
        .replace("$MAX_WORDS", str(int(target * 1.125)))
    )
    gen_cfg = types.GenerateContentConfig(system_instruction=system)
    part = pdf_part(pdf_path)

    resp = client().models.generate_content(
        model=model, contents=[part, user], config=gen_cfg
    )
    record_cost(episode_id, model, resp, cfg, stage="script")
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
        resp = client().models.generate_content(
            model=model, contents=[part, retry_msg], config=gen_cfg
        )
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

    model = cfg["models"]["metadata"]  # a short, cheap call
    try:
        resp = client().models.generate_content(
            model=model,
            contents=load_prompt("episode_title.md") + script,
            config=types.GenerateContentConfig(max_output_tokens=8000),
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
    return title


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
    return False


def citation_flags(script: str, paper_text: str | None = None) -> list[dict]:
    """Scan for citation-shaped patterns and return them for human review.

    Never auto-fails: the point is to put every string that *could* be a
    fabricated citation in front of a person, so a bare year sitting near a
    proper noun counts even though most such hits are innocent.

    When `paper_text` is supplied, each flag also carries `in_paper`, which is
    what separates a real fabrication from a name the paper actually uses.
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

        # A bare year is suspicious when a proper noun sits just before it.
        # Scanning backward from the year (rather than forward from a capital)
        # avoids the non-overlapping-match trap, where a leading "The" would
        # otherwise swallow the real proper noun behind it.
        for m in _YEAR_RE.finditer(body):
            preceding = _WORD_RE.finditer(body[: m.start()])
            for w in list(preceding)[-4:]:
                token = w.group(0)
                if (
                    len(token) > 2
                    and token[:1].isupper()
                    and token not in _NOT_AUTHOR_TOKENS
                ):
                    spans.append(
                        (w.start(), m.end(), "proper noun near year",
                         body[w.start():m.end()])
                    )
                    break

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

    if paper_text is not None:
        normalized = _normalize(paper_text)
        for flag in flags:
            flag["in_paper"] = appears_in_paper(flag["text"], normalized)
    return flags
