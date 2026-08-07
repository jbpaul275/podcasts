"""Config loading. config.toml is read once at startup and passed around as a dict."""

import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Deployed, this points at a mounted volume so the library survives redeploys.
DATA_DIR = Path(os.environ.get("PAPERPOD_DATA_DIR") or (ROOT / "data"))
INBOX_DIR = DATA_DIR / "inbox"
PROCESSED_DIR = INBOX_DIR / "processed"
PAPERS_DIR = DATA_DIR / "papers"
CHUNKS_DIR = DATA_DIR / "audio" / "chunks"
FINAL_DIR = DATA_DIR / "audio" / "final"
PROMPTS_DIR = ROOT / "prompts"


def load_config() -> dict:
    with open(ROOT / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    # Deployment overrides: the container knows its own URL and port, and
    # neither belongs in a file that is committed to the repo.
    if base_url := os.environ.get("PAPERPOD_BASE_URL"):
        cfg.setdefault("server", {})["base_url"] = base_url
    if port := os.environ.get("PORT"):
        cfg.setdefault("server", {})["port"] = int(port)
    # Concurrency is a property of the deployment -- the account's rate limits
    # and the machine's size -- so it is tunable without a code change.
    if workers := os.environ.get("PAPERPOD_WORKERS"):
        try:
            cfg.setdefault("server", {})["workers"] = max(1, int(workers))
        except ValueError:
            pass
    # Model choice differs between a laptop on the free tier and a deployed
    # instance, and editing the committed file for that collides with every
    # pull. Environment wins.
    for stage in ("metadata", "script", "tts"):
        if model := os.environ.get(f"PAPERPOD_MODEL_{stage.upper()}"):
            cfg.setdefault("models", {})[stage] = model
    for host in ("a", "b"):
        if voice := os.environ.get(f"PAPERPOD_VOICE_{host.upper()}"):
            cfg.setdefault("voices", {})[f"host_{host}"] = voice
    for d in (INBOX_DIR, PROCESSED_DIR, PAPERS_DIR, CHUNKS_DIR, FINAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return cfg


# Edited prompts live on the data volume, not next to the shipped defaults: the
# repo copy is baked into the image and a redeploy would wipe anything written
# there. Keeping them here also means the defaults stay readable as the thing
# an edit can always be reverted to.
PROMPT_OVERRIDE_DIR = DATA_DIR / "prompts"


def prompt_names() -> list[str]:
    """Every prompt the pipeline ships. This is the allowlist -- names arrive
    from URLs, and anything not on it must never reach a path."""
    return sorted(p.name for p in PROMPTS_DIR.glob("*.md"))


def is_prompt_name(name: str) -> bool:
    return name in set(prompt_names())


def prompt_default(name: str) -> str:
    """The shipped text, always readable regardless of any override."""
    if not is_prompt_name(name):
        raise KeyError(name)
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def prompt_override(name: str) -> str | None:
    """The edited text, or None if this prompt is unedited."""
    if not is_prompt_name(name):
        raise KeyError(name)
    path = PROMPT_OVERRIDE_DIR / name
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return text if text.strip() else None


def save_prompt(name: str, body: str) -> None:
    if not is_prompt_name(name):
        raise KeyError(name)
    PROMPT_OVERRIDE_DIR.mkdir(parents=True, exist_ok=True)
    (PROMPT_OVERRIDE_DIR / name).write_text(body, encoding="utf-8")


def reset_prompt(name: str) -> None:
    """Drop the override; the shipped default takes over again."""
    if not is_prompt_name(name):
        raise KeyError(name)
    (PROMPT_OVERRIDE_DIR / name).unlink(missing_ok=True)


def load_prompt(name: str) -> str:
    """Read at call time, so an edit takes effect on the next episode without
    a restart."""
    return prompt_override(name) or prompt_default(name)


# Phrases the pipeline's own safeguards assume are in the prompt. Losing one
# does not raise anything -- it quietly removes a guarantee -- so an edit that
# drops it is worth saying out loud.
_EXPECTED_PHRASES = {
    "script_system.md": [
        ("HOST_A", "the speaker-tag format. Without it the script fails "
                   "format validation and the episode fails."),
        ("fabricat", "the no-fabricated-citations rule. The citation flags "
                     "only catch what slips past it; they are not a substitute."),
    ],
}


def prompt_warnings(name: str, body: str) -> list[str]:
    """Ways an edited prompt would silently break something.

    Warnings, not errors: it is the operator's prompt. But a missing `$SCRIPT`
    means revisions quietly send no script at all, and that is not the kind of
    thing anyone notices from the output.
    """
    out = []
    default = prompt_default(name)
    for token in sorted(set(re.findall(r"\$[A-Z_]{3,}", default))):
        if token not in body:
            out.append(
                f"{token} is gone. The code still substitutes it, so whatever "
                f"it carried is now missing from the request entirely."
            )
    for phrase, why in _EXPECTED_PHRASES.get(name, []):
        if phrase.casefold() not in body.casefold():
            out.append(f"No mention of “{phrase}” — that was {why}")
    if not body.strip():
        out.append("Empty. Saving this is the same as having no prompt at all.")
    return out
