"""Config loading. config.toml is read once at startup and passed around as a dict."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INBOX_DIR = DATA_DIR / "inbox"
PROCESSED_DIR = INBOX_DIR / "processed"
PAPERS_DIR = DATA_DIR / "papers"
CHUNKS_DIR = DATA_DIR / "audio" / "chunks"
FINAL_DIR = DATA_DIR / "audio" / "final"
PROMPTS_DIR = ROOT / "prompts"


def load_config() -> dict:
    with open(ROOT / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    for d in (INBOX_DIR, PROCESSED_DIR, PAPERS_DIR, CHUNKS_DIR, FINAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return cfg


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")
