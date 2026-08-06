"""Script generation: PDF in (natively), speaker-tagged Markdown dialogue out.

Also home of the citation-flag validator, which regex-scans the finished script
for citation-shaped strings and surfaces them for human review in the UI.
"""

import re

import db
from config import PAPERS_DIR, load_prompt
from . import PipelineError
from .gemini import client, pdf_part, record_cost, strip_fences

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

# Name (2004) / Name and Name (2004) / Name et al. (2004)
_NAME_YEAR_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+|\s+et\s+al\.?)?\s*\(\s*(?:19|20)\d{2}\s*\)"
)
_ET_AL_RE = re.compile(r"\b[A-Z][a-z]+\s+et\s+al\b\.?")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_WORD_RE = re.compile(r"\b[\w'’-]+\b")

# Lower number wins when two patterns overlap.
_PRIORITY = {"name-year cite": 0, "et al.": 1, "proper noun near year": 2}


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
    record_cost(episode_id, model, resp, cfg)
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
        record_cost(episode_id, model, resp, cfg)
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


def citation_flags(script: str) -> list[dict]:
    """Scan for citation-shaped patterns and return them for human review.

    Never auto-fails: the point is to put every string that *could* be a
    fabricated citation in front of a person, so a bare year sitting near a
    proper noun counts even though most such hits are innocent.
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
                    and token not in _COMMON_SENTENCE_STARTERS
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
            flags.append({"line": lineno, "kind": kind, "text": text.strip()})
    return flags
