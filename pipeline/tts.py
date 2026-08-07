"""TTS: script in, per-chunk WAV files out.

Design constraints (from the Gemini speech-generation docs):
- Multi-speaker supports at most two speakers.
- Quality drifts on long generations, so the script is chunked to ~200-300
  words, splitting only on speaker-turn boundaries.
- The model occasionally 500s, rate-limits, or returns text tokens instead of
  audio. Those are retried on the server's own schedule (see gemini.py); a
  chunk that still fails becomes a logged gap rather than sinking the episode.
- Output is raw PCM (24 kHz, 16-bit, mono unless the response says otherwise);
  we wrap it in a WAV header.

Resumability: chunking is deterministic from the stored script, and a chunk
whose WAV already exists on disk is skipped, so killing the process mid-TTS
and retrying resumes where it left off.
"""

import json
import logging
import re
import wave

import db
from config import CHUNKS_DIR
from . import NoAudioError, PipelineError, QuotaUnavailable
from .gemini import call_with_retry, client, record_cost
from .script import parse_turns

log = logging.getLogger("paperpod.tts")

DEFAULT_RATE = 24000


def chunk_turns(turns: list[tuple[str, str]], target_words: int, max_words: int) -> list[dict]:
    """Greedy chunking on turn boundaries. A chunk closes once it reaches
    target_words, or before a turn that would push it past max_words."""
    chunks: list[dict] = []
    current: list[tuple[str, str]] = []
    count = 0
    for speaker, text in turns:
        words = len(text.split())
        if current and (count >= target_words or count + words > max_words):
            chunks.append({"turns": current, "words": count})
            current, count = [], 0
        current.append((speaker, text))
        count += words
    if current:
        chunks.append({"turns": current, "words": count})
    return chunks


def synthesize(episode_id: str, cfg: dict) -> None:
    ep = db.get_episode(episode_id)
    if not ep or not ep["script_md"]:
        raise PipelineError("no script stored; run the scripting stage first")

    turns = parse_turns(ep["script_md"])
    if not turns:
        raise PipelineError("script parsed to zero turns")

    tcfg = cfg.get("tts", {})
    chunks = chunk_turns(
        turns,
        tcfg.get("chunk_target_words", 250),
        tcfg.get("chunk_max_words", 320),
    )
    context_turns = tcfg.get("context_turns", 2)

    out_dir = CHUNKS_DIR / episode_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest on disk so a partial run is inspectable.
    manifest = []
    turn_offset = 0
    for seq, chunk in enumerate(chunks):
        manifest.append({
            "seq": seq,
            "words": chunk["words"],
            "turns": [f"{s}: {t}" for s, t in chunk["turns"]],
            "context": [
                f"{s}: {t}" for s, t in turns[max(0, turn_offset - context_turns):turn_offset]
            ],
        })
        turn_offset += len(chunk["turns"])
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    failed = []
    last_error: Exception | None = None
    total = len(manifest)
    for entry in manifest:
        seq = entry["seq"]
        wav_path = out_dir / f"{seq:03d}.wav"
        if wav_path.exists() and wav_path.stat().st_size > 44:
            log.info("chunk %03d already synthesized, skipping", seq)
            continue
        # Written before the call, not after: a stalled chunk is exactly the
        # case you want to see, and its timestamp is what shows it stalled.
        db.set_progress(episode_id, f"synthesizing chunk {seq + 1} of {total}")
        try:
            _synthesize_chunk(episode_id, entry, wav_path, cfg)
        except QuotaUnavailable:
            # Every remaining chunk would fail identically; stop rather than
            # grinding through the whole script to produce nothing.
            raise
        except Exception as e:
            log.error("chunk %03d failed, giving up on it: %s", seq, e)
            failed.append(seq)
            last_error = e

    if failed:
        if len(failed) == len(manifest):
            # Carry the underlying reason: "every chunk failed" on its own sends
            # you to the logs to find out what is actually wrong.
            detail = str(last_error or "no error recorded")
            raise PipelineError(
                f"every TTS chunk failed using {model_for(episode_id, cfg)}; "
                f"nothing to assemble. Last error: {detail[:400]}"
            )
        # Partial failure: assemble the rest with a logged gap rather than
        # failing the whole episode.
        db.stage_start(episode_id, "synthesizing:gaps")
        db.stage_end(
            episode_id, "synthesizing:gaps", ok=False,
            detail=f"chunks failed and will be gaps in the final audio: {failed}",
        )


def _build_prompt(entry: dict, cfg: dict) -> str:
    lines = []
    lines.append(
        "Read aloud the following two-host podcast conversation between HOST_A and "
        "HOST_B. Natural, warm, conversational delivery at a moderate pace."
    )
    if entry["context"]:
        lines.append(
            "This segment continues an ongoing conversation. For continuity of tone "
            "and prosody only, the immediately preceding lines were:\n"
            + "\n".join(f"  {c}" for c in entry["context"])
            + "\nDo NOT read those preceding lines aloud. Speak ONLY the dialogue below, "
            "starting mid-conversation as if the discussion is already underway."
        )
    lines.append("\n".join(entry["turns"]))
    return "\n\n".join(lines)


def model_for(episode_id: str, cfg: dict) -> str:
    """The episode's own TTS model if it has one, else the configured default.
    Pinned per episode so a config change mid-library cannot produce audio that
    switches voice model partway through."""
    ep = db.get_episode(episode_id)
    return (ep["tts_model"] if ep and ep["tts_model"] else cfg["models"]["tts"])


def _synthesize_chunk(episode_id: str, entry: dict, wav_path, cfg: dict) -> None:
    from google.genai import types

    model = model_for(episode_id, cfg)
    speech_config = types.SpeechConfig(
        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
            speaker_voice_configs=[
                types.SpeakerVoiceConfig(
                    speaker="HOST_A",
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=cfg["voices"]["host_a"]
                        )
                    ),
                ),
                types.SpeakerVoiceConfig(
                    speaker="HOST_B",
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=cfg["voices"]["host_b"]
                        )
                    ),
                ),
            ]
        )
    )
    prompt = _build_prompt(entry, cfg)

    def once():
        resp = client().models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=speech_config,
            ),
        )
        record_cost(episode_id, model, resp, cfg, stage="tts")
        # Raises NoAudioError when the model answers with text tokens, which
        # call_with_retry treats as retryable.
        return _extract_audio(resp)

    data, mime = call_with_retry(
        once, cfg, model, label=f"tts chunk {wav_path.stem}",
        extra_retryable=(NoAudioError,),
    )
    _write_wav(wav_path, data, mime)


def _extract_audio(resp) -> tuple[bytes, str]:
    """Pull inline audio bytes out of the response; a text-only response is the
    documented intermittent failure mode and is treated as an error."""
    try:
        parts = resp.candidates[0].content.parts or []
    except (AttributeError, IndexError, TypeError):
        parts = []
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is not None and inline.data:
            return inline.data, (inline.mime_type or "")
    text = (getattr(resp, "text", None) or "")[:200]
    raise NoAudioError(f"model returned no audio (text instead: {text!r})")


def _write_wav(wav_path, data: bytes, mime: str) -> None:
    """The API returns raw 16-bit PCM; the sample rate is declared in the mime
    type (e.g. audio/L16;codec=pcm;rate=24000). Wrap in a WAV header rather
    than guessing — fall back to the documented 24 kHz."""
    if mime.strip().lower().startswith("audio/wav"):
        wav_path.write_bytes(data)
        return
    m = re.search(r"rate=(\d+)", mime or "")
    rate = int(m.group(1)) if m else DEFAULT_RATE
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)
