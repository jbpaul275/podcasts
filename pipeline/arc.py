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

# An episode about several works needs a shape too, and it is not a property of
# any one of them -- it is how they stand to each other. So a second vocabulary,
# picked per episode rather than per paper.
CONFLICT = "conflict"
CONVERGENT = "convergent"
EXTENSION = "extension"
RELATIONS = (CONFLICT, CONVERGENT, EXTENSION)
AUTO = "auto"

# Where an unreadable or missing relation lands. Conflict rather than something
# more neutral-sounding, because its third beat is the commensurability check:
# it is the one arc that can conclude "these do not actually disagree" and still
# be a good episode. Defaulting to convergent would have the hosts paper over a
# disagreement they never checked for, which is the worse way to be wrong.
DEFAULT_RELATION = CONFLICT

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


def clean_relation(value) -> str:
    """A relation name, or the default. `auto` is not one: it means "the
    positions stage decides", and by the time an arc is needed it has."""
    value = str(value or "").strip().casefold()
    return value if value in RELATIONS else DEFAULT_RELATION


def of(row, paper_count: int = 1) -> str:
    """Which arc this episode takes.

    One work and the shape is a property of the work -- an argument and an
    experiment need different beats. Several works and the shape is how they
    stand to each other instead, which is a fact about the episode and comes
    from its relation.
    """
    if paper_count > 1:
        try:
            stored = row["relation"] if row is not None else None
        except (IndexError, KeyError, TypeError):
            stored = None
        return clean_relation(stored)
    return kind_of(row)


def text(kind: str) -> str:
    name = kind if kind in RELATIONS else clean_kind(kind)
    return load_prompt(f"arc_{name}.md").strip()


def segments(kind: str) -> list[str]:
    """Segment names, in order, parsed from the arc.

    Parsed rather than duplicated in code: a list here that disagreed with the
    prompt would have the outline planning beats the writer does not recognise,
    and nothing would report the mismatch.
    """
    return [m.group(1).strip() for m in _LINE.finditer(text(kind))]
