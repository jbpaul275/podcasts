"""Which shape an episode takes, and where that shape is defined.

The seven-segment arc was built for empirical papers: it asks for an
identification strategy, results with magnitudes, and a missing robustness
check. Hand it a work of history or philosophy and the model dutifully
manufactures quantitative framing for a book that has none — three thin,
awkward segments where the interesting material is elsewhere.

So there are two arcs, and the metadata stage says which one a work needs.

They live in `prompts/arc_*.md` rather than in code or config, for two reasons.
They are editable in the browser like every other prompt; and defining them once
means the outline stage and the writing stage cannot drift apart, which they
would immediately if each carried its own copy.
"""

import re

from config import load_prompt

EMPIRICAL = "empirical"
THEORETICAL = "theoretical"
KINDS = (EMPIRICAL, THEORETICAL)
DEFAULT = EMPIRICAL

# "3. Identification — How the authors got leverage..." The name is what the
# outline groups beats by; the rest is direction for whoever is writing.
_LINE = re.compile(r"^\s*\d+\.\s*(.+?)\s*[—-]\s*\S", re.MULTILINE)


def kind_of(row) -> str:
    """The arc this work needs. Empirical unless the paper says otherwise:
    most uploads are papers, and the empirical arc is the one that has been
    exercised."""
    try:
        stored = (row["work_kind"] or "").strip().casefold() if row is not None else ""
    except (IndexError, KeyError, TypeError):
        stored = ""
    return stored if stored in KINDS else DEFAULT


def clean_kind(value) -> str:
    value = str(value or "").strip().casefold()
    return value if value in KINDS else DEFAULT


def text(kind: str) -> str:
    return load_prompt(f"arc_{clean_kind(kind)}.md").strip()


def segments(kind: str) -> list[str]:
    """Segment names, in order, parsed from the arc.

    Parsed rather than duplicated in code: a list here that disagreed with the
    prompt would have the outline planning beats the writer does not recognise,
    and nothing would report the mismatch.
    """
    return [m.group(1).strip() for m in _LINE.finditer(text(kind))]
