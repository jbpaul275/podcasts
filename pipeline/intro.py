"""The spoken AI disclosure that opens every episode.

Apple requires AI disclosure in three places when the audio is machine
generated: the show metadata, each episode's metadata, and the content itself.
The first two are the feed's `description` and the per-episode attribution
line; this module is the third.

Two things are deliberate:

- The wording is a template, not something the script model writes. A
  compliance statement that a model paraphrases is a compliance statement that
  can drift, and nothing downstream would notice.
- It is synthesized in its own single-voice call rather than as an extra
  speaker in the conversation. The multi-speaker API takes *exactly* two
  speaker configs ("Exactly two speaker voice configurations must be
  provided"), so a third voice cannot come from the same call. A separate call
  also keeps the announcer out of the host chunking, so the disclosure is
  never split across a chunk boundary or reworded to fit the dialogue.

The rendered text is stored next to the WAV. Editing a paper's title or
authors changes the text, and the mismatch is what triggers a re-synthesis;
otherwise a resumed run reuses what is already on disk.
"""

import logging
import re

import db
from config import CHUNKS_DIR
from prose import author_credit, decaps
from . import NoAudioError, PipelineError
from .gemini import call_with_retry, client, record_cost

log = logging.getLogger("paperpod.intro")

WAV_NAME = "intro.wav"
TEXT_NAME = "intro.txt"

DEFAULT_TEMPLATE = (
    "This is a Paperpod, an AI generated podcast with discussions of important "
    "academic papers. Today's episode is about $TITLE, by $AUTHORS."
)
DEFAULT_TEMPLATE_NO_AUTHORS = (
    "This is a Paperpod, an AI generated podcast with discussions of important "
    "academic papers. Today's episode is about $TITLE."
)
DEFAULT_VOICE = "Charon"

# Style direction for the announcer. Kept away from the disclosure text itself
# so the model cannot mistake one for the other.
DIRECTION = (
    "Read the following announcement aloud, once, exactly as written. Calm, "
    "clear, unhurried announcer delivery. Do not add words, greetings, or "
    "commentary."
)


def enabled(cfg: dict) -> bool:
    return bool(cfg.get("intro", {}).get("enabled", True))


def _voice(cfg: dict) -> str:
    return (cfg.get("intro", {}).get("voice") or DEFAULT_VOICE).strip() or DEFAULT_VOICE


def intro_text(ep, cfg: dict) -> str | None:
    """The disclosure as it will be spoken, or None when disabled. Takes the
    episode row rather than an id: the library renders one of these per row."""
    if not enabled(cfg) or ep is None:
        return None

    icfg = cfg.get("intro", {})
    title = decaps((ep["title"] or "").strip()) or "an untitled paper"
    authors = db.episode_authors(ep)
    if authors:
        template = icfg.get("template") or DEFAULT_TEMPLATE
        credit = author_credit(authors)
    else:
        template = icfg.get("template_no_authors") or DEFAULT_TEMPLATE_NO_AUTHORS
        credit = ""

    text = template.replace("$TITLE", title).replace("$AUTHORS", credit)
    # "Smith et al." already ends a sentence, so the template's own full stop
    # would double it. Collapse a bare pair only -- an ellipsis in a title is
    # a legitimate three.
    return re.sub(r"(?<!\.)\.\.(?!\.)", ".", text).strip()


def wav_path(episode_id: str):
    return CHUNKS_DIR / episode_id / WAV_NAME


def recorded_text(episode_id: str) -> str | None:
    """The disclosure the WAV on disk actually says, or None if there is no
    intro audio. Compared against intro_text() to spot audio that announces a
    title or authors someone has since edited."""
    out_dir = CHUNKS_DIR / episode_id
    wav = out_dir / WAV_NAME
    if not wav.exists() or wav.stat().st_size <= 44:
        return None
    try:
        return (out_dir / TEXT_NAME).read_text(encoding="utf-8")
    except OSError:
        return None


def synthesize_intro(episode_id: str, cfg: dict) -> None:
    """Write the intro WAV, unless one matching the current text already
    exists. Failure raises: an episode published without the disclosure is a
    compliance problem, not a cosmetic gap like a missing dialogue chunk."""
    ep = db.get_episode(episode_id)
    if not ep:
        raise PipelineError("no such episode")
    text = intro_text(ep, cfg)
    if not text:
        return

    out_dir = CHUNKS_DIR / episode_id
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / WAV_NAME
    stamp = out_dir / TEXT_NAME
    if wav.exists() and wav.stat().st_size > 44:
        try:
            if stamp.read_text(encoding="utf-8") == text:
                log.info("intro already synthesized for %s, skipping", episode_id)
                return
        except OSError:
            pass
        log.info("intro text changed for %s, re-synthesizing", episode_id)

    db.set_progress(episode_id, "synthesizing intro")
    _synthesize(episode_id, text, wav, cfg)
    stamp.write_text(text, encoding="utf-8")


def _synthesize(episode_id: str, text: str, wav, cfg: dict) -> None:
    from google.genai import types

    from .tts import _extract_audio, _write_wav, model_for

    model = model_for(episode_id, cfg)
    speech_config = types.SpeechConfig(
        # Single-voice output. Mutually exclusive with the multi-speaker config
        # the two-host chunks use, which is why this is its own call.
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=_voice(cfg))
        )
    )

    def once():
        resp = client().models.generate_content(
            model=model,
            contents=f"{DIRECTION}\n\n{text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=speech_config,
            ),
        )
        record_cost(episode_id, model, resp, cfg, stage="intro")
        return _extract_audio(resp)

    data, mime = call_with_retry(
        once, cfg, model, label="tts intro", extra_retryable=(NoAudioError,),
    )
    _write_wav(wav, data, mime)
