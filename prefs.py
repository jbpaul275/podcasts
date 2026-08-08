"""Sticky choices for the creation wizard.

Three layers, in order: what `config.toml` ships, what you last chose, and what
you pick for the episode in front of you. Each overrides the one before it.

The stored layer is deletions rather than a copy of the defaults. Writing the
defaults into the database on first use would freeze them: a later edit to
`config.toml` would then be silently ignored, and "restore defaults" would
restore whatever the defaults happened to be the first time the wizard was
opened. An absent key means "whatever the config says today".
"""

import db
from pipeline import arc as arc_mod

# Wizard field -> the episode column it is written to. Voices and models are
# stored per episode as well as remembered, so an episode built last week is
# still reproducible after the preferences move on.
FIELDS = {
    "metadata_model": "metadata_model",
    "script_model": "script_model_wanted",
    "tts_model": "tts_model",
    "voice_a": "voice_a",
    "voice_b": "voice_b",
    "length_policy": "length_policy",
    "research": "research",
    "relation": "relation",
}

# Fields that only mean something for an episode built on several works. Shown
# and saved only then, so the wizard for a single paper is not asking how it
# stands to works that are not there.
COMPARISON_ONLY = ("relation",)

LABELS = {
    "metadata_model": "Metadata model",
    "script_model": "Script model",
    "tts_model": "Voice model",
    "voice_a": "Host A voice",
    "voice_b": "Host B voice",
    "length_policy": "Episode length",
    "research": "Research the reception",
    "relation": "How they relate",
}


def _dedupe(names) -> list[str]:
    """Order-preserving, dropping blanks. The first entry is the default, so
    order carries meaning and a set would lose it."""
    out = []
    for name in names:
        name = (name or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def defaults(cfg: dict) -> dict[str, str]:
    models, voices = cfg.get("models", {}), cfg.get("voices", {})
    return {
        "metadata_model": models.get("metadata", ""),
        "script_model": models.get("script", ""),
        "tts_model": models.get("tts", ""),
        "voice_a": voices.get("host_a", ""),
        "voice_b": voices.get("host_b", ""),
        # "auto" is the point of the outline stage: the paper decides.
        "length_policy": "auto",
        # Off by default: a search-grounded call on top of an already expensive
        # pipeline, and most papers do not need it. Worth it for work whose
        # interest is in how it was received rather than what it says.
        "research": "off",
        # "auto" means the positions stage decides after reading the works,
        # which is the honest default: the relation is a fact about them, and
        # picking it before reading is guessing.
        "relation": "auto",
    }


def choices(cfg: dict) -> dict[str, list[str]]:
    """What each dropdown may offer.

    The configured default is always first and always present, so a config that
    names a model missing from the lists still produces a usable wizard rather
    than one that silently cannot express the current setting.
    """
    scfg, tcfg = cfg.get("script", {}), cfg.get("tts", {})
    base = defaults(cfg)
    text = _dedupe([base["script_model"], base["metadata_model"],
                    *(scfg.get("models") or []), scfg.get("fallback_model")])
    voices = _dedupe(cfg.get("voices", {}).get("choices") or [])
    return {
        "metadata_model": _dedupe([base["metadata_model"], *text]),
        "script_model": _dedupe([base["script_model"], *text]),
        "tts_model": _dedupe([base["tts_model"], *(tcfg.get("models") or [])]),
        "voice_a": _dedupe([base["voice_a"], *voices]),
        "voice_b": _dedupe([base["voice_b"], *voices]),
        "length_policy": _dedupe(["auto", *(scfg.get("lengths") or {})]),
        "research": ["off", "on"],
        "relation": ["auto", *arc_mod.RELATIONS],
    }


def visible(comparison: bool) -> list[str]:
    """Which wizard fields apply to this episode, in FIELDS order."""
    return [f for f in FIELDS if comparison or f not in COMPARISON_ONLY]


def current(cfg: dict, comparison: bool = False) -> dict[str, str]:
    """The defaults, overlaid with anything chosen last time.

    A stored value that is no longer offered is dropped rather than kept. Models
    get retired and voices get renamed; carrying a dead one forward would fail
    the episode at the first API call, days after the choice was made.
    """
    stored = db.get_settings()
    allowed = choices(cfg)
    out = defaults(cfg)
    # Whether a work was argued with is most of what makes a comparison worth
    # making, so research starts on when there are several. Only as a default:
    # an explicit "off" stored from last time still wins.
    if comparison:
        out["research"] = "on"
    for field in FIELDS:
        value = stored.get(field)
        if value and value in allowed[field]:
            out[field] = value
    return out


def validate(values: dict, cfg: dict) -> dict[str, str]:
    """Keep the fields that name a real choice; raise on one that does not.

    Rejecting rather than falling back to the default: a typo'd model in a form
    post is a bug somewhere, and quietly substituting something else would
    produce an episode built with settings nobody picked.
    """
    allowed = choices(cfg)
    out = {}
    for field in FIELDS:
        value = (values.get(field) or "").strip()
        if not value:
            continue
        if value not in allowed[field]:
            raise ValueError(f"{LABELS[field]}: {value!r} is not one of the choices")
        out[field] = value
    return out


def save(values: dict, cfg: dict, comparison: bool = False) -> dict[str, str]:
    """Remember these for next time, and return the full effective set."""
    clean = validate(values, cfg)
    if clean:
        db.set_settings(clean)
    return current(cfg, comparison=comparison)


def reset() -> None:
    db.clear_settings()


def apply_to_episode(episode_id: str, values: dict[str, str]) -> None:
    db.update_episode(episode_id, **{col: values[field]
                                     for field, col in FIELDS.items()
                                     if values.get(field)})


def for_episode(row, cfg: dict, comparison: bool = False) -> dict[str, str]:
    """What this episode was actually built with, falling back to the current
    preference for anything it does not pin."""
    out = current(cfg, comparison=comparison)
    if row is None:
        return out
    for field, col in FIELDS.items():
        try:
            value = row[col]
        except (IndexError, KeyError):
            value = None
        if value:
            out[field] = value
    return out
