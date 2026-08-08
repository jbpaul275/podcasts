"""Tests for the parts that are pure logic: chunking, script validation,
citation flagging, WAV wrapping, range requests, feed generation, resume.

Nothing here calls the Gemini API.
"""

import json
import os
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import tts  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Point the DB at a temp file for every test."""
    import config
    import db

    from pipeline import gemini

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db._local.__dict__.clear()
    db.init_db()
    # Process-global: a rate-limit window one test opens would make the next
    # one sleep for it.
    gemini.THROTTLE.reset()
    yield
    gemini.THROTTLE.reset()
    db._local.__dict__.clear()


# ---------------------------------------------------------------- chunking

def test_chunking_never_splits_mid_turn():
    from pipeline.tts import chunk_turns

    turns = [("HOST_A" if i % 2 == 0 else "HOST_B", " ".join(["word"] * 40)) for i in range(20)]
    chunks = chunk_turns(turns, target_words=250, max_words=320)

    rebuilt = [t for c in chunks for t in c["turns"]]
    assert rebuilt == turns, "chunking must preserve every turn in order"
    assert all(c["words"] <= 320 for c in chunks), "no chunk may exceed max_words"
    assert len(chunks) > 1


def test_chunking_keeps_oversized_single_turn_intact():
    from pipeline.tts import chunk_turns

    turns = [("HOST_A", " ".join(["word"] * 500)), ("HOST_B", "Short reply.")]
    chunks = chunk_turns(turns, target_words=250, max_words=320)
    assert chunks[0]["turns"] == [turns[0]], "a long turn is never split mid-turn"
    assert len(chunks) == 2


def test_chunking_empty():
    from pipeline.tts import chunk_turns
    assert chunk_turns([], 250, 320) == []


# ------------------------------------------------------ script validation

def test_strip_fences():
    from pipeline.gemini import strip_fences

    assert strip_fences("```markdown\nHOST_A: Hi.\n```") == "HOST_A: Hi."
    assert strip_fences("```\nHOST_A: Hi.\n```") == "HOST_A: Hi."
    assert strip_fences("HOST_A: Hi.") == "HOST_A: Hi."


def test_clean_strips_markdown_emphasis():
    from pipeline.script import _clean

    out = _clean("HOST_A: That's **really** big.\nHOST_B: The *whole* point.")
    assert "**" not in out and "*" not in out
    assert "really" in out and "whole" in out


def test_format_violations_detects_stray_prose():
    from pipeline.script import _format_violations

    good = "HOST_A: One.\n\nHOST_B: Two."
    assert _format_violations(good) == []

    bad = "HOST_A: One.\n[Music fades in]\nHOST_B: Two.\n## Segment 2"
    violations = _format_violations(bad)
    assert "[Music fades in]" in violations
    assert "## Segment 2" in violations


def test_parse_turns_roundtrip():
    from pipeline.script import parse_turns

    turns = parse_turns("HOST_A: Hello there.\n\nHOST_B: Hi: with a colon.")
    assert turns == [
        ("HOST_A", "Hello there."),
        ("HOST_B", "Hi: with a colon."),
    ]


# ------------------------------------------------------- citation flagging

def test_citation_flags_catch_fabrication_shapes():
    from pipeline.script import citation_flags

    script = "\n".join([
        "HOST_A: Card and Krueger (1994) found the opposite.",
        "HOST_B: Right, and Chetty et al. push on that.",
        "HOST_A: Acemoglu's 2001 paper argued the reverse.",
        "HOST_B: The 1969 Gould decision changed the board.",
    ])
    flags = citation_flags(script)
    texts = " ".join(f["text"] for f in flags)

    assert "Card and Krueger (1994)" in texts
    assert any("et al" in f["text"] for f in flags)
    assert any(f["line"] == 3 for f in flags), "a name possessive of a year is a cite"
    assert any(f["line"] == 4 for f in flags), "a name after the year is a cite too"


def test_a_year_in_ordinary_conversation_does_not_flag():
    """The window is deliberately tight. At four words any capitalised word
    loosely near a year matched, and scripts are full of those."""
    from pipeline.script import citation_flags

    for line in [
        "HOST_A: does that match what you were doing on a Tuesday afternoon back in 2018?",
        "HOST_B: The Mariel boatlift in 1980 is the classic case.",
        "HOST_A: Your LinkedIn profile from 2019 says otherwise.",
        "HOST_B: We spoke about this last Friday in 2021.",
        "HOST_A: There's a 2019 study by Fabricated somewhere.",
    ]:
        assert citation_flags(line) == [], f"should not flag: {line!r}"


def test_possessive_stopwords_are_not_authors():
    from pipeline.script import _looks_like_an_author

    assert _looks_like_an_author("Acemoglu's")
    assert not _looks_like_an_author("There's"), "a contraction is not a surname"
    assert not _looks_like_an_author("It's")


def test_citation_flags_ignore_ordinary_dialogue():
    from pipeline.script import citation_flags

    clean = "\n".join([
        "HOST_A: So the wage effect is about three percent. That's small.",
        "HOST_B: There's a literature suggesting the opposite, but I'd hedge on that.",
        "HOST_A: Wait, that's the whole sample?",
    ])
    assert citation_flags(clean) == []


def test_statutes_and_named_events_are_not_flagged():
    """Real false positives from a live run: a law and a named event that
    happen to carry a year are never academic citations."""
    from pipeline.script import citation_flags

    script = "\n".join([
        "HOST_A: The Civil Rights Act of 1964 changed the calculus.",
        "HOST_B: What Congress wrote into the 1964 statute mattered.",
        "HOST_A: That was before the 2008 Crisis reshaped everything.",
        "HOST_B: The 1970 Census is the backbone of the sample.",
    ])
    assert citation_flags(script) == []


def test_flags_are_checked_against_the_paper():
    """The signal is whether a name traces back to the source PDF."""
    from pipeline.script import citation_flags

    script = (
        "HOST_A: Gould's 1969 board decision set the precedent.\n"
        "HOST_B: And Fictitious's 1977 ruling supposedly agreed."
    )
    paper = "... appointed William Gould to the board in 1969, which ..."

    flags = {f["text"]: f["in_paper"] for f in citation_flags(script, paper)}
    assert flags["Gould's 1969"] is True
    assert flags["Fictitious's 1977"] is False


def test_in_paper_absent_without_paper_text():
    from pipeline.script import citation_flags

    flags = citation_flags("HOST_B: See Angrist (2009).")
    assert flags and "in_paper" not in flags[0]


def test_appears_in_paper_matching():
    from pipeline.script import _normalize, appears_in_paper

    paper = _normalize("We follow Card and Krueger (1994). Later, Chetty et al. show ...")
    assert appears_in_paper("Card and Krueger (1994)", paper)
    assert appears_in_paper("Chetty et al.", paper)
    assert not appears_in_paper("Imaginary and Fake (2011)", paper)


def test_unverified_flag_count_drives_the_ui(tmp_path, monkeypatch):
    """The library badge counts only what needs a human."""
    import app as app_mod
    import db

    db.create_episode("EP1", "/tmp/a.pdf", "h")
    db.update_episode(
        "EP1", status="done",
        script_md="HOST_A: The Civil Rights Act of 1964 and also Ghostwriter (1988).",
    )
    monkeypatch.setattr(app_mod, "_paper_text", lambda _id: "A paper mentioning nothing relevant.")

    view = app_mod._episode_view(db.get_episode("EP1"))
    assert view["flag_count"] == 1, "statute suppressed, fabrication counted"
    assert view["flags_unverified"][0]["text"] == "Ghostwriter (1988)"


def test_citation_flags_report_line_numbers():
    from pipeline.script import citation_flags

    flags = citation_flags("HOST_A: Fine.\nHOST_B: See Angrist (2009).")
    assert flags and all(f["line"] == 2 for f in flags)


# ------------------------------------------------------------------ retry

def _api_error(code, details):
    from google.genai import errors
    return errors.APIError(code, details)


# The real body Gemini returned on a rate limit, trimmed to the parts we read.
RATE_LIMITED = {
    "error": {
        "code": 429,
        "message": "You exceeded your current quota. Please retry in 12.595337758s.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaMetric": "generate_content_free_tier_requests"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"},
        ],
    }
}

ZERO_QUOTA = {
    "error": {
        "code": 429,
        "message": ("Quota exceeded for metric: generate_content_free_tier_requests, "
                    "limit: 0, model: gemini-3.1-pro"),
        "status": "RESOURCE_EXHAUSTED",
        "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                     "retryDelay": "12s"}],
    }
}

RETRY_CFG = {"retry": {"attempts": 3, "base_delay_s": 2, "max_delay_s": 60}}


def test_retry_delay_read_from_server_response():
    from pipeline.gemini import retry_delay

    assert retry_delay(_api_error(429, RATE_LIMITED)) == 12.0


def test_retry_delay_absent_when_not_supplied():
    from pipeline.gemini import retry_delay

    assert retry_delay(_api_error(500, {"error": {"code": 500, "message": "boom"}})) is None


def test_rate_limit_is_retried_on_the_servers_schedule(monkeypatch):
    """The bug: fixed 2s/4s backoff gave up before a 12s window reopened."""
    from pipeline.gemini import call_with_retry

    slept = []
    monkeypatch.setattr("pipeline.gemini.time.sleep", slept.append)

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _api_error(429, RATE_LIMITED)
        return "audio"

    assert call_with_retry(flaky, RETRY_CFG, "m", "chunk") == "audio"
    assert len(calls) == 3
    assert slept == [12.5, 12.5], "must wait the server's 12s, not 2s then 4s"


def test_zero_quota_fails_immediately_with_actionable_message(monkeypatch):
    from pipeline import QuotaUnavailable
    from pipeline.gemini import call_with_retry

    slept = []
    monkeypatch.setattr("pipeline.gemini.time.sleep", slept.append)
    calls = []

    def always_zero():
        calls.append(1)
        raise _api_error(429, ZERO_QUOTA)

    with pytest.raises(QuotaUnavailable, match="no quota"):
        call_with_retry(always_zero, RETRY_CFG, "gemini-3.1-pro", "script")
    assert calls == [1], "limit: 0 can never clear; do not retry"
    assert slept == []


def test_permanent_errors_are_not_retried(monkeypatch):
    from pipeline.gemini import call_with_retry

    monkeypatch.setattr("pipeline.gemini.time.sleep", lambda s: None)
    calls = []

    def bad_request():
        calls.append(1)
        raise _api_error(400, {"error": {"code": 400, "message": "bad model"}})

    with pytest.raises(Exception):
        call_with_retry(bad_request, RETRY_CFG, "m", "x")
    assert calls == [1], "a 400 fails the same way every time"


def test_server_errors_use_exponential_backoff(monkeypatch):
    from pipeline.gemini import call_with_retry

    slept = []
    monkeypatch.setattr("pipeline.gemini.time.sleep", slept.append)

    def always_500():
        raise _api_error(503, {"error": {"code": 503, "message": "overloaded"}})

    with pytest.raises(Exception):
        call_with_retry(always_500, RETRY_CFG, "m", "x")
    assert slept == [2.5, 4.5], "no server hint, so back off exponentially"


def test_delay_is_capped(monkeypatch):
    from pipeline.gemini import call_with_retry

    slept = []
    monkeypatch.setattr("pipeline.gemini.time.sleep", slept.append)
    huge = {"error": {"code": 429, "message": "wait",
                      "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                   "retryDelay": "3600s"}]}}

    def always():
        raise _api_error(429, huge)

    with pytest.raises(Exception):
        call_with_retry(always, {"retry": {"attempts": 2, "max_delay_s": 30}}, "m", "x")
    assert slept == [30], "an absurd server delay must not hang the worker"


def test_text_instead_of_audio_is_retried(monkeypatch):
    """TTS's own failure mode arrives as a valid response, not an HTTP error."""
    from pipeline import NoAudioError
    from pipeline.gemini import call_with_retry

    monkeypatch.setattr("pipeline.gemini.time.sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise NoAudioError("model returned no audio")
        return (b"pcm", "audio/L16;rate=24000")

    got = call_with_retry(flaky, RETRY_CFG, "m", "chunk", extra_retryable=(NoAudioError,))
    assert got == (b"pcm", "audio/L16;rate=24000")
    assert len(calls) == 2


def test_retry_gives_up_and_reraises_the_last_error(monkeypatch):
    from pipeline.gemini import call_with_retry

    monkeypatch.setattr("pipeline.gemini.time.sleep", lambda s: None)

    def always():
        raise _api_error(429, RATE_LIMITED)

    with pytest.raises(Exception, match="429"):
        call_with_retry(always, RETRY_CFG, "m", "x")


# ------------------------------------------------------------ WAV wrapping

def test_write_wav_uses_rate_from_mime(tmp_path):
    from pipeline.tts import _write_wav

    pcm = b"\x00\x01" * 1000
    out = tmp_path / "chunk.wav"
    _write_wav(out, pcm, "audio/L16;codec=pcm;rate=24000")

    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.readframes(w.getnframes()) == pcm


def test_write_wav_falls_back_to_documented_rate(tmp_path):
    from pipeline.tts import DEFAULT_RATE, _write_wav

    out = tmp_path / "chunk.wav"
    _write_wav(out, b"\x00\x01" * 10, "")
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == DEFAULT_RATE


def test_write_wav_passes_through_real_wav(tmp_path):
    from pipeline.tts import _write_wav

    src = tmp_path / "src.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(b"\x00\x01" * 50)
    data = src.read_bytes()

    out = tmp_path / "out.wav"
    _write_wav(out, data, "audio/wav")
    assert out.read_bytes() == data


# ------------------------------------------------------- loudnorm parsing

# Real ffmpeg 6.1 stderr: progress lines before the JSON, stream summary and a
# final progress line after it. The JSON is NOT at the end of stderr.
FFMPEG_STDERR = (
    "size=N/A time=00:15:19.90 bitrate=N/A speed=65.1x elapsed=0:00:14.12\r"
    "size=N/A time=00:15:52.70 bitrate=N/A speed=65.1x elapsed=0:00:14.63\r"
    '[Parsed_loudnorm_0 @ 0x998c3c900]\n{\n\t"input_i" : "-19.56",\n'
    '\t"input_tp" : "0.27",\n\t"input_lra" : "5.80",\n\t"input_thresh" : "-30.04",\n'
    '\t"output_i" : "-16.56",\n\t"output_tp" : "-1.50",\n\t"output_lra" : "5.20",\n'
    '\t"output_thresh" : "-27.03",\n\t"normalization_type" : "dynamic",\n'
    '\t"target_offset" : "0.56"\n}\n'
    "[out#0/null @ 0x998c3c180] video:0KiB audio:358384KiB subtitle:0KiB "
    "other streams:0KiB global headers:0KiB muxing overhead: unknown\n"
    "size=N/A time=00:15:55.70 bitrate=N/A speed=65.3x elapsed=0:00:14.64\r"
)


def test_loudnorm_json_parsed_despite_trailing_ffmpeg_output(monkeypatch):
    import subprocess as sp

    from pipeline import assemble

    monkeypatch.setattr(
        assemble.subprocess, "run",
        lambda *a, **k: sp.CompletedProcess(a, 0, stdout="", stderr=FFMPEG_STDERR),
    )
    measured = assemble._measure_loudnorm("x.wav", -16.0, -1.5, 11.0)

    assert measured["input_i"] == "-19.56"
    assert measured["input_tp"] == "0.27"
    assert measured["target_offset"] == "0.56"


def test_loudnorm_raises_when_measurement_truly_missing(monkeypatch):
    import subprocess as sp

    from pipeline import PipelineError, assemble

    monkeypatch.setattr(
        assemble.subprocess, "run",
        lambda *a, **k: sp.CompletedProcess(a, 1, stdout="", stderr="Invalid argument\n"),
    )
    with pytest.raises(PipelineError, match="no usable JSON"):
        assemble._measure_loudnorm("x.wav", -16.0, -1.5, 11.0)


def test_loudnorm_ignores_unrelated_brace_blocks(monkeypatch):
    """A stray JSON-ish blob must not be mistaken for the measurement."""
    import subprocess as sp

    from pipeline import assemble

    noisy = FFMPEG_STDERR + '\n[something] {"unrelated": "blob"}\n'
    monkeypatch.setattr(
        assemble.subprocess, "run",
        lambda *a, **k: sp.CompletedProcess(a, 0, stdout="", stderr=noisy),
    )
    assert assemble._measure_loudnorm("x.wav", -16.0, -1.5, 11.0)["input_i"] == "-19.56"


def test_extract_audio_rejects_text_response():
    from pipeline import PipelineError
    from pipeline.tts import _extract_audio

    class FakeResp:
        candidates = []
        text = "I'm sorry, I can't generate that audio."

    with pytest.raises(PipelineError, match="no audio"):
        _extract_audio(FakeResp())


# ---------------------------------------------------------------- database

def test_dedupe_by_content_hash_not_title():
    import db

    db.create_episode("EP1", "/tmp/a.pdf", "hash-aaa")
    db.create_episode("EP2", "/tmp/b.pdf", "hash-bbb")
    db.update_principal("EP1", title="Minimum Wages and Employment", year=1994)
    db.update_principal("EP2", title="Minimum Wages and Employment", year=2019)

    assert db.episodes_for_paper(db.find_paper_by_sha("hash-aaa")["id"])[0]["id"] == "EP1"
    assert db.episodes_for_paper(db.find_paper_by_sha("hash-bbb")["id"])[0]["id"] == "EP2"
    assert len(db.list_episodes()) == 2, "same title, different content = two episodes"


def test_cost_accumulates_across_stages():
    import db

    db.create_episode("EP1", "/tmp/a.pdf", "h")
    db.add_cost("EP1", 0.012)
    db.add_cost("EP1", 0.240)
    assert db.get_episode("EP1")["cost_usd"] == pytest.approx(0.252)


def test_cost_is_broken_down_by_stage():
    import db

    db.create_episode("EP1", "/tmp/a.pdf", "h")
    db.add_cost("EP1", 0.0051, "metadata")
    db.add_cost("EP1", 0.0104, "script")
    db.add_cost("EP1", 0.0009, "title")
    for _ in range(7):                      # one call per TTS chunk
        db.add_cost("EP1", 0.0893, "tts")

    row = db.get_episode("EP1")
    breakdown = db.cost_breakdown(row)
    assert breakdown["tts"] == pytest.approx(0.6251)
    assert breakdown["script"] == pytest.approx(0.0104)
    assert sum(breakdown.values()) == pytest.approx(row["cost_usd"])


def test_cost_rows_are_ranked_with_shares():
    import app as app_mod
    import db

    db.create_episode("EP1", "/tmp/a.pdf", "h")
    db.add_cost("EP1", 0.90, "tts")
    db.add_cost("EP1", 0.10, "script")

    rows = app_mod._cost_rows(db.get_episode("EP1"))
    assert [r["stage"] for r in rows] == ["tts", "script"], "largest first"
    assert rows[0]["pct"] == 90 and rows[1]["pct"] == 10
    assert rows[0]["label"] == "Speech synthesis"


def test_cost_rows_empty_when_nothing_spent():
    import app as app_mod
    import db

    db.create_episode("EP1", "/tmp/a.pdf", "h")
    assert app_mod._cost_rows(db.get_episode("EP1")) == []


def test_cost_breakdown_survives_corrupt_json():
    import db

    db.create_episode("EP1", "/tmp/a.pdf", "h")
    db.update_episode("EP1", cost_json="{not json")
    assert db.cost_breakdown(db.get_episode("EP1")) == {}
    db.add_cost("EP1", 0.5, "tts")  # must recover rather than raise
    assert db.cost_breakdown(db.get_episode("EP1")) == {"tts": 0.5}


def test_stage_log_start_and_end():
    import db

    db.create_episode("EP1", "/tmp/a.pdf", "h")
    db.stage_start("EP1", "scripting")
    db.stage_end("EP1", "scripting", ok=True, detail="2 flags")

    rows = db.get_stage_log("EP1")
    assert len(rows) == 1
    assert rows[0]["stage"] == "scripting"
    assert rows[0]["ok"] == 1
    assert rows[0]["ended_at"] is not None


def test_ulids_sort_by_creation_order():
    import db

    ids = [db.new_ulid() for _ in range(50)]
    assert ids == sorted(ids) or len(set(ids)) == 50


# ------------------------------------------------------------------ resume

def test_resume_returns_to_interrupted_stage():
    from pipeline.run import resume_stage_for

    assert resume_stage_for("synthesizing") == "synthesizing", (
        "a crash mid-TTS must resume at TTS, not re-run scripting"
    )
    assert resume_stage_for("queued") == "extracting"
    assert resume_stage_for("failed") == "extracting"


def test_completed_chunks_are_skipped_on_resume(tmp_path, monkeypatch):
    """The resumability guarantee: synthesize() must not re-call the API for
    chunks whose WAV is already on disk."""
    import config
    import db
    from pipeline import tts

    monkeypatch.setattr(tts, "CHUNKS_DIR", tmp_path)
    db.create_episode("EP1", "/tmp/a.pdf", "h")
    script = "\n".join(f"HOST_{'AB'[i % 2]}: " + " ".join(["word"] * 60) for i in range(10))
    db.update_episode("EP1", script_md=script)

    calls = []

    def fake_synth(episode_id, entry, wav_path, cfg):
        calls.append(entry["seq"])
        wav_path.write_bytes(b"\x00" * 100)

    monkeypatch.setattr(tts, "_synthesize_chunk", fake_synth)
    # Intro off: this is about chunk resumability, and the intro is a
    # separate single-voice call that would need its own stub.
    cfg = {"tts": {"chunk_target_words": 120, "chunk_max_words": 200, "context_turns": 2,
                   "retry_pass_delay_s": 0},
           "intro": {"enabled": False}}

    tts.synthesize("EP1", cfg)
    first_pass = list(calls)
    assert len(first_pass) > 1

    calls.clear()
    tts.synthesize("EP1", cfg)
    assert calls == [], "second run must synthesize nothing; all chunks already exist"


def test_chunk_prompt_includes_context_but_only_synthesizes_current(tmp_path, monkeypatch):
    import db
    from pipeline import tts

    monkeypatch.setattr(tts, "CHUNKS_DIR", tmp_path)
    db.create_episode("EP1", "/tmp/a.pdf", "h")
    script = "\n".join(f"HOST_{'AB'[i % 2]}: Turn number {i}." for i in range(12))
    db.update_episode("EP1", script_md=script)

    entries = []
    monkeypatch.setattr(
        tts, "_synthesize_chunk",
        lambda eid, entry, path, cfg: (entries.append(entry), path.write_bytes(b"\x00" * 100)),
    )
    tts.synthesize("EP1", {"tts": {"chunk_target_words": 3, "chunk_max_words": 6, "context_turns": 2,
                                  "retry_pass_delay_s": 0},
                           "intro": {"enabled": False}})

    later = entries[2]
    assert later["context"], "later chunks get preceding turns for prosody"
    prompt = tts._build_prompt(later, {})
    assert "Do NOT read those preceding lines aloud" in prompt
    for line in later["turns"]:
        assert line in prompt


# ------------------------------------------------------ config env overrides

def test_env_overrides_model_and_voice_choice(monkeypatch):
    """Local and deployed instances differ on model choice; editing the
    committed config for that collides with every pull."""
    import importlib

    import config

    monkeypatch.setenv("PAPERPOD_MODEL_SCRIPT", "gemini-3-flash-preview")
    monkeypatch.setenv("PAPERPOD_MODEL_TTS", "some-other-tts")
    monkeypatch.setenv("PAPERPOD_VOICE_A", "Charon")
    importlib.reload(config)
    cfg = config.load_config()

    assert cfg["models"]["script"] == "gemini-3-flash-preview"
    assert cfg["models"]["tts"] == "some-other-tts"
    assert cfg["voices"]["host_a"] == "Charon"
    # Untouched keys keep their committed values.
    assert cfg["models"]["metadata"] and cfg["voices"]["host_b"]


def test_config_file_is_used_when_env_is_absent(monkeypatch):
    import importlib

    import config

    for var in ("PAPERPOD_MODEL_METADATA", "PAPERPOD_MODEL_SCRIPT",
                "PAPERPOD_MODEL_TTS", "PAPERPOD_VOICE_A", "PAPERPOD_VOICE_B"):
        monkeypatch.delenv(var, raising=False)
    importlib.reload(config)
    cfg = config.load_config()
    assert cfg["models"]["tts"].startswith("gemini")


# ------------------------------------------------------- model catalogue

def test_every_offered_tts_model_has_a_price():
    """An unpriced model is silently costed at zero, which is the one number a
    model comparison is trying to read."""
    import tomllib

    cfg = tomllib.load(open("config.toml", "rb"))
    offered = set(cfg["tts"]["models"]) | {cfg["models"]["tts"]}
    missing = sorted(offered - set(cfg["costs"]))
    assert not missing, f"no [costs] entry for {missing}"


def test_picker_offers_no_live_api_models():
    """Live models are bidirectional streaming and unreachable through the
    generateContent call this pipeline makes; one here fails every chunk."""
    import tomllib

    cfg = tomllib.load(open("config.toml", "rb"))
    for name in cfg["tts"]["models"]:
        assert "live" not in name, f"{name} is a Live API model"
        assert "native-audio" not in name, f"{name} is a Live API model"
        assert "tts" in name, f"{name} does not look like a TTS model"


def test_unpriced_model_warns_rather_than_reporting_zero(caplog, monkeypatch):
    import db
    from pipeline import gemini

    monkeypatch.setattr(gemini, "_UNPRICED_WARNED", set())
    db.create_episode("EP1", "/tmp/a.pdf", "h")

    class Resp:
        usage_metadata = type("U", (), {"prompt_token_count": 100,
                                        "candidates_token_count": 5000})()

    with caplog.at_level("WARNING"):
        usd = gemini.record_cost("EP1", "unpriced-model", Resp(), {"costs": {}}, "tts")
    assert usd == 0.0
    assert "reported as $0.00" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        gemini.record_cost("EP1", "unpriced-model", Resp(), {"costs": {}}, "tts")
    assert "reported as $0.00" not in caplog.text, "warns once per model, not per chunk"


# --------------------------------------------- script quality: thinking etc

def test_thinking_level_is_passed_through():
    from pipeline.script import _script_config

    cfg = {"script": {"thinking_level": "high"}}
    gen = _script_config(cfg, "sys")
    assert gen.thinking_config is not None
    assert str(gen.thinking_config.thinking_level).upper().endswith("HIGH")
    assert gen.tools is None, "grounding stays off unless asked for"


def test_no_thinking_config_when_unset():
    from pipeline.script import _script_config

    assert _script_config({"script": {}}, "sys").thinking_config is None
    assert _script_config({"script": {"thinking_level": ""}}, "sys").thinking_config is None


def test_grounding_adds_the_search_tool():
    from pipeline.script import _script_config

    gen = _script_config({"script": {"grounding": True}}, "sys")
    assert gen.tools and gen.tools[0].google_search is not None


def test_fallback_model_ordering():
    from pipeline.script import _script_models

    assert _script_models({"models": {"script": "pro"},
                           "script": {"fallback_model": "flash"}}) == ["pro", "flash"]
    assert _script_models({"models": {"script": "pro"}, "script": {}}) == ["pro"]
    # A fallback identical to the primary is not a fallback.
    assert _script_models({"models": {"script": "pro"},
                           "script": {"fallback_model": "pro"}}) == ["pro"]


def test_collect_grounding_reads_sources_and_queries():
    from pipeline.script import collect_grounding

    class Web:
        def __init__(s, t, u, d): s.title, s.uri, s.domain = t, u, d

    class Chunk:
        def __init__(s, w): s.web = w

    class Meta:
        web_search_queries = ["minimum wage employment elasticity"]
        grounding_chunks = [
            Chunk(Web("Card and Krueger 1994", "https://nber.org/w4509", "nber.org")),
            Chunk(Web("Card and Krueger 1994", "https://nber.org/w4509", "nber.org")),
            Chunk(None),
        ]

    class Resp:
        candidates = [type("C", (), {"grounding_metadata": Meta()})()]

    got = collect_grounding(Resp())
    assert got["queries"] == ["minimum wage employment elasticity"]
    assert len(got["sources"]) == 1, "duplicate sources collapse"
    assert got["sources"][0]["uri"] == "https://nber.org/w4509"


def test_collect_grounding_empty_without_metadata():
    from pipeline.script import collect_grounding

    class Resp:
        candidates = []

    assert collect_grounding(Resp()) == {"queries": [], "sources": []}


def test_grounded_citation_counts_as_corroborated():
    """With search on, a real citation absent from the PDF is legitimate — but
    only if the model actually consulted a page supporting it."""
    from pipeline.script import citation_flags

    script = ("HOST_A: Card and Krueger (1994) found the opposite.\n"
              "HOST_B: And Ghostwriter (2011) supposedly agreed.")
    paper = "This paper studies minimum wages. No prior work is named here."
    web = "Card and Krueger 1994 Minimum Wages and Employment nber.org"

    flags = {f["text"]: f for f in citation_flags(script, paper, web)}
    ck = flags["Card and Krueger (1994)"]
    assert ck["in_paper"] is True and ck["source"] == "web"
    fake = flags["Ghostwriter (2011)"]
    assert fake["in_paper"] is False and fake["source"] is None, (
        "grounding must not turn the flag check off — an invented cite still flags"
    )


def test_paper_beats_web_as_the_recorded_source():
    from pipeline.script import citation_flags

    flags = citation_flags("HOST_A: See Angrist (2009).",
                           "We follow Angrist (2009) closely.",
                           "Angrist 2009 something else")
    assert flags[0]["source"] == "paper"


def test_corroboration_survives_real_shaped_grounding_data():
    """Grounding sources are titled after the paper, not its authors, so
    "Card and Krueger (1994)" never appears verbatim in them — matching has to
    work on the names, or every genuine citation flags as invented."""
    from pipeline.script import citation_flags

    script = ("HOST_A: The classic reference is Card and Krueger (1994).\n"
              "HOST_B: And Ghostwriter (2011) supposedly showed the reverse.")
    paper = "We study the imperial examination system. No prior work is named."
    web = ("Minimum Wages and Employment: A Case Study nber.org "
           "Card Krueger 1994 minimum wage")

    flags = {f["text"]: f for f in citation_flags(script, paper, web)}
    assert flags["Card and Krueger (1994)"]["source"] == "web"
    assert flags["Ghostwriter (2011)"]["in_paper"] is False


def test_name_token_matching_does_not_verify_an_invented_name():
    from pipeline.script import _normalize, appears_in_paper

    corpus = _normalize("Card Krueger 1994 minimum wage employment")
    assert appears_in_paper("Card and Krueger (1994)", corpus)
    assert not appears_in_paper("Card and Fabricated (1994)", corpus), (
        "one real surname must not vouch for an invented co-author"
    )


def test_a_timeout_is_retryable():
    """A stalled connection should cost a retry, not the whole chunk."""
    from pipeline import gemini

    class ReadTimeout(Exception):
        pass

    assert gemini.is_retryable(ReadTimeout("timed out")) is True
    assert gemini.is_retryable(ValueError("bad argument")) is False


def test_the_client_is_built_with_a_deadline(monkeypatch):
    """Without a timeout, one hung call wedges a worker forever."""
    from pipeline import gemini

    captured = {}

    class FakeClient:
        def __init__(self, http_options=None):
            captured["timeout_ms"] = http_options.timeout

    monkeypatch.setattr(gemini, "_client", None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    import google.genai
    monkeypatch.setattr(google.genai, "Client", FakeClient)

    gemini.configure({"retry": {"request_timeout_s": 42}})
    try:
        gemini.client()
    finally:
        gemini._client = None
        gemini._timeout_s = gemini.DEFAULT_TIMEOUT_S
    assert captured["timeout_ms"] == 42_000


# The real body returned when Google retired gemini-3-pro-preview.
_RETIRED_BODY = {
    "error": {
        "code": 404,
        "message": ("This model models/gemini-3-pro-preview is no longer "
                    "available. Please update your code to use a newer model "
                    "for the latest features and improvements."),
        "status": "NOT_FOUND",
    }
}


class _Retired(Exception):
    code = 404
    details = _RETIRED_BODY


def test_a_retired_model_is_terminal_not_retried():
    """404 on the model is a config fact: every retry fails identically."""
    from pipeline import ModelRetired, gemini

    calls = []

    def boom():
        calls.append(1)
        raise _Retired("404 NOT_FOUND")

    cfg = {"retry": {"attempts": 4, "base_delay_s": 0, "max_delay_s": 0}}
    with pytest.raises(ModelRetired) as exc:
        gemini.call_with_retry(boom, cfg, "gemini-3-pro-preview")
    assert len(calls) == 1, "a retired model must not be retried"
    # The message has to name the fix; a bare 404 sends you to the logs.
    assert "/admin/models" in str(exc.value)


def test_a_404_that_is_not_about_the_model_still_raises_plainly():
    from pipeline import ModelRetired, gemini

    class NotFound(Exception):
        code = 404
        details = {"error": {"code": 404, "message": "file not found"}}

    cfg = {"retry": {"attempts": 2, "base_delay_s": 0, "max_delay_s": 0}}
    with pytest.raises(Exception) as exc:
        gemini.call_with_retry(lambda: (_ for _ in ()).throw(NotFound("nope")),
                               cfg, "some-model")
    assert not isinstance(exc.value, ModelRetired)


def test_a_retired_script_model_falls_back(tmp_path, monkeypatch):
    """The fallback exists for exactly this; it must not be quota-only."""
    import db
    from pipeline import script as script_mod

    db.create_episode("ERET", "/tmp/r.pdf", "sha-ret")
    papers = tmp_path / "papers"
    papers.mkdir()
    monkeypatch.setattr(db, "PAPERS_DIR", papers)
    db.paper_pdf(db.principal_paper("ERET")["id"]).write_bytes(b"%PDF-1.4 fake")

    tried = []

    class Resp:
        text = "HOST_A: Hello there everyone.\nHOST_B: Good to be here."
        usage_metadata = None
        candidates = []

    def fake_generate_content(*, model, contents, config):
        tried.append(model)
        if model == "gemini-3-pro-preview":
            raise _Retired("404 NOT_FOUND")
        return Resp()

    monkeypatch.setattr(script_mod, "client",
                        lambda: type("C", (), {"models": type("M", (), {
                            "generate_content": staticmethod(fake_generate_content)})()})())
    monkeypatch.setattr(script_mod, "pdf_part", lambda p: "PART")

    cfg = {
        "models": {"script": "gemini-3-pro-preview"},
        "script": {"target_words": 1600,
                   "fallback_model": "gemini-3-flash-preview"},
        "retry": {"attempts": 2, "base_delay_s": 0, "max_delay_s": 0},
    }

    out = script_mod.generate_script("ERET", cfg)
    assert tried == ["gemini-3-pro-preview", "gemini-3-flash-preview"]
    assert "HOST_A:" in out
    assert db.get_episode("ERET")["script_model"] == "gemini-3-flash-preview"


class _Usage:
    prompt_token_count = 1_000_000
    candidates_token_count = 1_000_000
    thoughts_token_count = 0


class _Resp:
    usage_metadata = _Usage()

    def __init__(self, model_version=None):
        self.model_version = model_version


def test_cost_uses_the_model_that_actually_ran(_isolated_db):
    """An alias can be repointed at a differently-priced model. A table keyed
    only on the requested name cannot see that happen."""
    import db
    from pipeline import gemini

    cfg = {"costs": {
        "gemini-flash-latest": {"input_per_1m": 0.30, "output_per_1m": 2.50},
        "gemini-3.6-flash": {"input_per_1m": 1.50, "output_per_1m": 7.50},
    }}
    db.create_episode("ECOST", "/tmp/c.pdf", "sha-cost")

    usd = gemini.record_cost("ECOST", "gemini-flash-latest",
                             _Resp("gemini-3.6-flash"), cfg)
    # 1M in + 1M out at the resolved model's rate, not the alias's stale one.
    assert usd == pytest.approx(1.50 + 7.50)


def test_cost_falls_back_to_the_requested_name(_isolated_db):
    """Not every response names a model; the alias entry still has to work."""
    import db
    from pipeline import gemini

    cfg = {"costs": {"gemini-pro-latest": {"input_per_1m": 2.0, "output_per_1m": 12.0}}}
    db.create_episode("ECOST2", "/tmp/c.pdf", "sha-cost2")
    usd = gemini.record_cost("ECOST2", "gemini-pro-latest", _Resp(None), cfg)
    assert usd == pytest.approx(14.0)


def test_an_alias_moving_to_an_unpriced_model_warns(_isolated_db, caplog):
    """The failure this is here to catch: silently reading $0.00 forever."""
    import db
    from pipeline import gemini

    gemini._UNPRICED_WARNED.clear()
    cfg = {"costs": {"gemini-flash-latest": {"input_per_1m": 1.5, "output_per_1m": 7.5}}}
    db.create_episode("ECOST3", "/tmp/c.pdf", "sha-cost3")

    # Resolved name is unpriced, but the alias is -- so it still costs, and says
    # the number is only right while the alias has not moved.
    with caplog.at_level("INFO"):
        usd = gemini.record_cost("ECOST3", "gemini-flash-latest",
                                 _Resp("gemini-4-flash"), cfg)
    assert usd == pytest.approx(9.0)
    assert "gemini-4-flash" in caplog.text

    # Neither priced: warn loudly and cost nothing rather than guess.
    gemini._UNPRICED_WARNED.clear()
    caplog.clear()
    with caplog.at_level("WARNING"):
        usd = gemini.record_cost("ECOST3", "unknown-model",
                                 _Resp("also-unknown"), cfg)
    assert usd == 0.0
    assert "$0.00" in caplog.text


def test_shipped_prices_cover_every_model_the_config_can_use():
    """A model with no entry is silently costed at zero."""
    from config import load_config
    from pipeline.script import script_choices

    cfg = load_config()
    used = set(cfg["models"].values()) | set(cfg["tts"]["models"]) | set(script_choices(cfg))
    used.add(cfg["script"]["fallback_model"])
    missing = sorted(m for m in used if m not in cfg["costs"])
    assert not missing, f"unpriced: {missing}"


# ------------------------------------------------------------- citation counts

def test_doi_normalization_rejects_things_that_are_not_dois():
    from pipeline.citations import normalize_doi

    good = "10.1257/aer.90.5.1397"
    for raw in (good, f"https://doi.org/{good}", f"doi: {good}",
                f"  https://dx.doi.org/{good}.  ", f"DOI:{good}"):
        assert normalize_doi(raw) == good, raw
    # Real DOIs get strange. This one is a live Wiley DOI.
    gnarly = "10.1002/(SICI)1097-0258(19980815)17:15<1661::AID-SIM968>3.0.CO;2-2"
    assert normalize_doi(gnarly) == gnarly

    for bad in (None, "", "n/a", "see the paper", "10.1257", "arXiv:2103.00020"):
        assert normalize_doi(bad) is None, bad

    # The suffix is interpolated into a URL path we build, so it must not be
    # able to carry path traversal or a query separator into one.
    for hostile in ("10.1234/../../etc/passwd", "10.1234/x?a=b", "10.1234/x#y",
                    "10.1234/x y", "10.1234/."):
        assert normalize_doi(hostile) is None, hostile


def test_a_title_near_miss_is_not_accepted(monkeypatch):
    """A plausible wrong paper is worse than no number: it would attach someone
    else's citation count and then sort the site by it."""
    from pipeline import citations

    monkeypatch.setattr(citations, "_get", lambda url, cfg: {
        "results": [{"display_name": "Minimum Wages and Employment in Ohio",
                     "cited_by_count": 999}]})
    assert citations.lookup(None, "Minimum Wages and Employment", {}) is None

    monkeypatch.setattr(citations, "_get", lambda url, cfg: {
        "results": [{"display_name": "minimum wages and employment!",
                     "cited_by_count": 4242}]})
    # Same letters and digits, different punctuation and case: that is a match.
    assert citations.lookup(None, "Minimum Wages and Employment", {}) == (4242, "openalex")


def test_doi_lookup_is_preferred_and_exact(monkeypatch):
    from pipeline import citations

    seen = []

    def fake_get(url, cfg):
        seen.append(url)
        return {"cited_by_count": 31337, "display_name": "Whatever"}

    monkeypatch.setattr(citations, "_get", fake_get)
    assert citations.lookup("10.1257/aer.90.5.1397", "A Title", {}) == (31337, "openalex")
    assert seen == ["https://api.openalex.org/works/doi:10.1257/aer.90.5.1397"]


def test_a_failed_lookup_returns_nothing_rather_than_raising(monkeypatch):
    from pipeline import citations

    monkeypatch.setattr(citations, "_get", lambda url, cfg: None)
    assert citations.lookup("10.1257/aer.90.5.1397", "A Title", {}) is None
    assert citations.lookup(None, None, {}) is None


def test_a_lookup_failure_never_touches_the_episode(_isolated_db, monkeypatch):
    """An unreachable third party is not a reason to lose a podcast, or to
    overwrite a count somebody typed in by hand."""
    import db
    from pipeline import citations, ingest

    db.create_episode("ECITE", "/tmp/c.pdf", "sha-cite")
    db.update_principal("ECITE", title="A Paper", cited_by=12, cited_by_source="entered by hand")

    monkeypatch.setattr(citations, "lookup",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert ingest.refresh_citations("ECITE", {}) is None
    assert db.get_episode("ECITE")["cited_by"] == 12

    monkeypatch.setattr(citations, "lookup", lambda *a, **k: None)
    assert ingest.refresh_citations("ECITE", {}) is None
    assert db.get_episode("ECITE")["cited_by"] == 12, "a miss must not clear it"

    # A hand-entered number survives an automatic lookup: somebody typed it
    # because the automatic route did not work, and re-running a stage must not
    # quietly undo that.
    monkeypatch.setattr(citations, "lookup", lambda *a, **k: (500, "openalex"))
    assert ingest.refresh_citations("ECITE", {}) is None
    assert db.get_episode("ECITE")["cited_by"] == 12

    # The explicit button says otherwise.
    assert ingest.refresh_citations("ECITE", {}, force=True) == 500
    assert db.get_episode("ECITE")["cited_by"] == 500

    # A looked-up number carries no such protection.
    monkeypatch.setattr(citations, "lookup", lambda *a, **k: (600, "openalex"))
    assert ingest.refresh_citations("ECITE", {}) == 600


def test_citations_can_be_switched_off(_isolated_db, monkeypatch):
    import db
    from pipeline import citations, ingest

    db.create_episode("ECITE2", "/tmp/c.pdf", "sha-cite2")
    called = []
    monkeypatch.setattr(citations, "lookup", lambda *a, **k: called.append(1) or (9, "x"))
    assert ingest.refresh_citations("ECITE2", {"citations": {"enabled": False}}) is None
    assert not called


# ------------------------------------------------------- spoken AI disclosure

def _intro_cfg(**over):
    cfg = {
        "models": {"tts": "t"},
        "voices": {"host_a": "Puck", "host_b": "Kore"},
        "intro": {"enabled": True, "voice": "Charon"},
    }
    cfg["intro"].update(over)
    return cfg


def test_intro_names_the_paper_and_says_it_is_ai_generated(_isolated_db):
    """Apple wants the disclosure in the audio itself, not only the metadata."""
    import db
    from pipeline import intro

    db.create_episode("EINTRO", "/tmp/i.pdf", "sha-intro")
    db.update_principal("EINTRO", title="Minimum Wages and Employment", authors=json.dumps(["David Card", "Alan Krueger"]))

    text = intro.intro_text(db.get_episode("EINTRO"), _intro_cfg())
    assert "AI generated" in text
    assert "Minimum Wages and Employment" in text
    assert "David Card and Alan Krueger" in text


def test_intro_is_off_when_switched_off(_isolated_db):
    import db
    from pipeline import intro

    db.create_episode("EINTRO2", "/tmp/i.pdf", "sha-intro2")
    assert intro.intro_text(db.get_episode("EINTRO2"), _intro_cfg(enabled=False)) is None


def test_intro_does_not_announce_an_uncredited_author(_isolated_db):
    """"by an uncredited author" is fine to read on a page and absurd to hear."""
    import db
    from pipeline import intro

    db.create_episode("EINTRO3", "/tmp/i.pdf", "sha-intro3")
    db.update_principal("EINTRO3", title="A Paper With No Byline")

    text = intro.intro_text(db.get_episode("EINTRO3"), _intro_cfg())
    assert "uncredited" not in text
    assert text.endswith("A Paper With No Byline.")


def test_intro_does_not_double_the_full_stop_after_et_al(_isolated_db):
    import db
    from pipeline import intro

    db.create_episode("EINTRO4", "/tmp/i.pdf", "sha-intro4")
    db.update_principal("EINTRO4", title="Attention Is All You Need", authors=json.dumps(["A", "B", "C", "D", "E"]))

    text = intro.intro_text(db.get_episode("EINTRO4"), _intro_cfg())
    assert text.endswith("by A et al.")
    assert ".." not in text


def test_intro_uses_one_voice_that_is_neither_host(_isolated_db, monkeypatch):
    """The multi-speaker API takes exactly two speakers, so a third voice has
    to come from a separate single-voice call."""
    import db
    from pipeline import intro

    db.create_episode("EINTRO5", "/tmp/i.pdf", "sha-intro5")
    db.update_principal("EINTRO5", title="A Paper", authors=json.dumps(["Solo"]))
    monkeypatch.setattr(intro, "CHUNKS_DIR", tmp_chunks())

    captured = {}

    class FakeModels:
        def generate_content(self, model=None, contents=None, config=None):
            captured["model"] = model
            captured["speech"] = config.speech_config
            captured["contents"] = contents
            return _audio_response()

    monkeypatch.setattr(intro, "client",
                        lambda: type("C", (), {"models": FakeModels()})())

    cfg = _intro_cfg()
    cfg["retry"] = {"attempts": 2, "base_delay_s": 0, "max_delay_s": 0}
    intro.synthesize_intro("EINTRO5", cfg)

    speech = captured["speech"]
    assert speech.multi_speaker_voice_config is None, (
        "a third voice cannot be added to the two-host call"
    )
    assert speech.voice_config.prebuilt_voice_config.voice_name == "Charon"
    assert speech.voice_config.prebuilt_voice_config.voice_name not in (
        cfg["voices"]["host_a"], cfg["voices"]["host_b"]
    ), "an announcer in a host's voice does not read as a handoff"
    assert "This is Paperpod" in captured["contents"]
    assert intro.wav_path("EINTRO5").exists()


def _audio_response():
    """The shape the SDK returns for an audio generation."""
    from types import SimpleNamespace

    part = SimpleNamespace(inline_data=SimpleNamespace(
        data=b"\x00\x01" * 100, mime_type="audio/L16;codec=pcm;rate=24000"))
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
        usage_metadata=None,
        model_version=None,
    )


def tmp_chunks():
    """A throwaway chunk directory; the intro writes its WAV there."""
    import tempfile
    return Path(tempfile.mkdtemp())


def test_intro_is_resynthesized_only_when_its_words_change(_isolated_db, monkeypatch):
    """Editing a title changes what the disclosure says, and nothing in the
    audio reveals that it is out of date."""
    import db
    from pipeline import intro

    db.create_episode("EINTRO6", "/tmp/i.pdf", "sha-intro6")
    db.update_principal("EINTRO6", title="First Title", authors=json.dumps(["Solo"]))
    monkeypatch.setattr(intro, "CHUNKS_DIR", tmp_chunks())

    calls = []

    def fake_synth(episode_id, text, wav, cfg):
        calls.append(text)
        wav.write_bytes(b"\x00" * 200)

    monkeypatch.setattr(intro, "_synthesize", fake_synth)

    intro.synthesize_intro("EINTRO6", _intro_cfg())
    assert len(calls) == 1
    intro.synthesize_intro("EINTRO6", _intro_cfg())
    assert len(calls) == 1, "unchanged wording must reuse the WAV on disk"

    db.update_principal("EINTRO6", title="Second Title")
    intro.synthesize_intro("EINTRO6", _intro_cfg())
    assert len(calls) == 2
    assert "Second Title" in calls[1]
    assert intro.recorded_text("EINTRO6") == calls[1]


def test_shipped_intro_template_reads_as_a_sentence(_isolated_db):
    """The wording in config.toml is what listeners actually hear, so it is
    worth asserting on rather than only the module default."""
    import db
    from config import load_config
    from pipeline import intro

    db.create_episode("EINTRO7", "/tmp/i.pdf", "sha-intro7")
    db.update_principal("EINTRO7", title="Some Important Paper", authors=json.dumps(["Ada Lovelace"]))

    cfg = load_config()
    text = intro.intro_text(db.get_episode("EINTRO7"), cfg)
    assert text.startswith("This is Paperpod, an AI generated podcast")
    assert text.endswith("Today's episode is about Some Important Paper, by Ada Lovelace.")
    assert "$" not in text, "every placeholder must have been substituted"


# ------------------------------------------------------- TTS reliability

def test_a_rate_limit_holds_back_every_other_call(monkeypatch):
    """The failure this fixes: each chunk discovered the closed window on its
    own, so one 429 became a run of failed chunks rather than a pause."""
    import threading

    from pipeline import gemini

    monkeypatch.setattr("pipeline.gemini.time.sleep", lambda s: None)

    def always():
        raise _api_error(429, RATE_LIMITED)   # RetryInfo says 12s

    with pytest.raises(Exception):
        gemini.call_with_retry(always, RETRY_CFG, "m", "chunk")

    # Another thread -- another chunk, or the other worker's episode -- is made
    # to wait rather than spending its own retries finding out the same thing.
    seen = []
    t = threading.Thread(target=lambda: seen.append(gemini.THROTTLE.remaining()))
    t.start(); t.join()
    assert seen[0] > 0


def test_the_thread_that_hit_the_limit_does_not_wait_twice(monkeypatch):
    """It already backs off on its own schedule in the retry loop; waiting on
    the shared window as well would double every rate-limited retry."""
    from pipeline import gemini

    slept = []
    monkeypatch.setattr("pipeline.gemini.time.sleep", slept.append)

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise _api_error(429, RATE_LIMITED)
        return "audio"

    assert gemini.call_with_retry(flaky, RETRY_CFG, "m", "chunk") == "audio"
    assert slept == [12.5], "one wait for one window, not two"


def test_a_dropped_connection_is_retried_but_a_bug_is_not():
    """These arrive with no HTTP status, so a check that keys on the status
    code called the most transient failure there is permanent."""
    from pipeline.gemini import is_retryable

    class ConnectError(Exception):
        pass

    class RemoteProtocolError(Exception):
        pass

    assert is_retryable(ConnectError("[Errno 104] Connection reset by peer"))
    assert is_retryable(RemoteProtocolError("Server disconnected without response"))
    assert is_retryable(OSError("SSL: EOF occurred in violation of protocol"))

    # Real bugs must still fail on the first attempt.
    assert not is_retryable(ValueError("bad argument"))
    assert not is_retryable(KeyError("models"))
    # And a status the server did send still decides, whatever the words say.
    assert not is_retryable(_api_error(400, {"error": {"code": 400,
                                                       "message": "connection"}}))


def test_the_gap_says_why_not_just_which(_isolated_db, monkeypatch):
    """Which chunks are missing is the symptom. Without the reason, the page
    sends you to the process log to find out what actually went wrong."""
    import db
    from pipeline import tts

    monkeypatch.setattr(tts, "CHUNKS_DIR", tmp_chunks())
    db.create_episode("ERELY", "/tmp/r.pdf", "sha-rely")
    db.update_episode("ERELY", script_md="\n".join(
        f"HOST_{'AB'[i % 2]}: " + " ".join(["word"] * 40) for i in range(6)))

    def half_fail(episode_id, entry, wav_path, cfg):
        if entry["seq"] % 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        wav_path.write_bytes(b"\x00" * 100)

    monkeypatch.setattr(tts, "_synthesize_chunk", half_fail)
    tts.synthesize("ERELY", {"tts": {"chunk_target_words": 40, "chunk_max_words": 60,
                                     "retry_pass_delay_s": 0},
                             "intro": {"enabled": False}})

    detail = [s for s in db.get_stage_log("ERELY") if s["stage"] == "synthesizing:gaps"][-1]["detail"]
    assert "RuntimeError: 429 RESOURCE_EXHAUSTED" in detail
    assert "chunks failed" in detail


def test_repeated_reasons_are_grouped_rather_than_repeated():
    """Eight copies of one sentence hide that it is one problem."""
    from pipeline.tts import _why

    same = {0: "APIError: 429", 1: "APIError: 429", 2: "NoAudioError: text"}
    out = _why([0, 1, 2], same)
    assert out.count("APIError: 429") == 1
    assert "(2 chunks)" in out and "(1 chunk)" in out
    assert out.index("APIError") < out.index("NoAudioError"), "commonest first"


def test_a_chunk_that_fails_once_is_recovered_by_the_second_pass(_isolated_db, monkeypatch):
    """A rate-limited chunk usually succeeds a minute later. Retrying inside
    the stage is the difference between a complete episode and one with a hole
    in it that waits for somebody to notice."""
    import db
    from pipeline import tts

    monkeypatch.setattr(tts, "CHUNKS_DIR", tmp_chunks())
    db.create_episode("ERELY2", "/tmp/r.pdf", "sha-rely2")
    db.update_episode("ERELY2", script_md="\n".join(
        f"HOST_{'AB'[i % 2]}: " + " ".join(["word"] * 40) for i in range(6)))

    seen = []

    def flaky_once(episode_id, entry, wav_path, cfg):
        seen.append(entry["seq"])
        if entry["seq"] == 1 and seen.count(1) == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        wav_path.write_bytes(b"\x00" * 100)

    monkeypatch.setattr(tts, "_synthesize_chunk", flaky_once)
    tts.synthesize("ERELY2", {"tts": {"chunk_target_words": 40, "chunk_max_words": 60,
                                      "retry_pass_delay_s": 0},
                              "intro": {"enabled": False}})

    assert seen.count(1) == 2, "the failed chunk is tried again, the others are not"
    assert not [s for s in db.get_stage_log("ERELY2") if s["stage"] == "synthesizing:gaps"], (
        "a chunk recovered on the second pass is not a gap"
    )


def test_the_second_pass_can_be_switched_off(_isolated_db, monkeypatch):
    import db
    from pipeline import tts

    monkeypatch.setattr(tts, "CHUNKS_DIR", tmp_chunks())
    db.create_episode("ERELY3", "/tmp/r.pdf", "sha-rely3")
    db.update_episode("ERELY3", script_md="\n".join(
        f"HOST_{'AB'[i % 2]}: " + " ".join(["word"] * 40) for i in range(6)))

    seen = []

    def fail_one(episode_id, entry, wav_path, cfg):
        seen.append(entry["seq"])
        if entry["seq"] == 1:
            raise RuntimeError("nope")
        wav_path.write_bytes(b"\x00" * 100)

    monkeypatch.setattr(tts, "_synthesize_chunk", fail_one)
    tts.synthesize("ERELY3", {"tts": {"chunk_target_words": 40, "chunk_max_words": 60,
                                      "retry_pass_delay_s": -1},
                              "intro": {"enabled": False}})
    assert seen.count(1) == 1


def test_the_page_limit_can_be_raised_without_a_redeploy(monkeypatch):
    """One awkward paper should not need a code change: the symptom of a stale
    deploy is a config change that appears to have been ignored."""
    import importlib

    import config

    monkeypatch.setenv("PAPERPOD_MAX_PAGES", "900")
    cfg = importlib.reload(config).load_config()
    assert cfg["script"]["max_pages"] == 900

    # Nonsense is ignored rather than crashing the app at startup.
    monkeypatch.setenv("PAPERPOD_MAX_PAGES", "not a number")
    assert importlib.reload(config).load_config()["script"]["max_pages"] > 0
    monkeypatch.delenv("PAPERPOD_MAX_PAGES")
    importlib.reload(config)


def test_the_shipped_limit_stays_under_pros_price_cliff():
    """Pro's input rate doubles past 200k tokens, which is ~775 pages at the
    documented ~258 tokens each. Past that the cost shown for an episode reads
    low, because [costs] carries the single rate."""
    from config import load_config
    from pipeline.ingest import API_MAX_PAGES

    limit = load_config()["script"]["max_pages"]
    assert limit >= 411, "a 400-page government report is not an exotic case"
    assert limit <= 775, "past this Pro charges double and [costs] understates it"
    assert limit <= API_MAX_PAGES


# ---------------------------------------------- title case and episode numbers

def test_generated_titles_are_forced_into_title_case():
    """The prompt asks for it; this makes it so. A model follows a
    capitalization instruction most of the time, and "most of the time" is
    exactly what looks like sloppiness in a list."""
    from prose import title_case

    assert title_case("The myth of the feudal venture capitalist") == (
        "The Myth of the Feudal Venture Capitalist")
    assert title_case("Why new cars lose value instantly") == (
        "Why New Cars Lose Value Instantly")
    # Small words stay down in the middle, up at either end.
    assert title_case("the market for lemons") == "The Market for Lemons"
    assert title_case("what serfdom was for") == "What Serfdom Was For"


def test_title_case_never_moves_a_capital_somebody_meant():
    """Blindly upper-casing first letters is how "iPhone" becomes "IPhone" --
    and those are the words a reader notices."""
    from prose import title_case

    assert title_case("How iPhone changed eBay") == "How iPhone Changed eBay"
    assert title_case("The GDP illusion") == "The GDP Illusion"
    assert title_case("What McKinsey got wrong") == "What McKinsey Got Wrong"
    # An already-correct title survives untouched.
    same = "The Myth of the Feudal Venture Capitalist"
    assert title_case(same) == same


def test_the_title_prompt_asks_for_one_format():
    """The bug was a prompt that permitted both and got both."""
    body = (Path(__file__).resolve().parents[1] / "prompts" / "episode_title.md").read_text()
    assert "Title Case" in body
    assert "Sentence case or title case" not in body, (
        "offering a choice is what produced the inconsistency"
    )


def test_numbers_are_given_at_publish_and_never_reused(_isolated_db):
    """Numbering at upload would count papers that failed, that stayed
    private, and the extra renderings made when comparing voices -- so the
    public feed would read 1, 2, 5, 9."""
    import db

    for i, eid in enumerate(["EA", "EB", "EC"]):
        db.create_episode(eid, f"/tmp/{eid}.pdf", f"sha-{eid}")

    assert db.assign_episode_number("EB") == 1, "first published, not first uploaded"
    assert db.assign_episode_number("EA") == 2
    # Already numbered: publishing again does not shuffle it.
    assert db.assign_episode_number("EB") == 1

    # Unpublishing keeps the number, so "episode 1" still means the same thing.
    db.update_episode("EB", published=0)
    assert db.get_episode("EB")["episode_number"] == 1
    assert db.assign_episode_number("EC") == 3, "the gap is not backfilled"


def test_a_revoiced_rendering_inherits_its_siblings_number(_isolated_db):
    """Same paper, same discussion, different voice -- and publishing one
    unpublishes the other. A new number would advertise a duplicate."""
    import db

    # A re-voicing is the same paper, not merely the same bytes -- it shares
    # the paper row, which is what makes it a sibling.
    paper = db.create_paper(source_path="/tmp/p.pdf", sha256="same-sha")
    db.create_episode("EORIG", "/tmp/p.pdf", "same-sha", papers=[paper])
    db.create_episode("EOTHER", "/tmp/q.pdf", "other-sha")
    db.create_episode("ECLONE", "/tmp/p.pdf", "same-sha", papers=[paper])

    assert db.assign_episode_number("EORIG") == 1
    assert db.assign_episode_number("EOTHER") == 2
    assert db.assign_episode_number("ECLONE") == 1, "the same episode, re-voiced"


# ------------------------------------------------ naming the quota that failed

# The real body Gemini returns when a free-tier daily allowance runs out.
_QUOTA_BODY = {
    "error": {
        "code": 429,
        "message": ("You exceeded your current quota, please check your plan and "
                    "billing details. For more information on this error, head to: "
                    "https://ai.google.dev/gemini-api/docs/rate-limits. To monitor "
                    "your current usage, head to: https://ai.dev/usage"),
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{
                 "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                 "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                 "quotaDimensions": {"model": "gemini-3.1-flash-tts", "location": "global"},
                 "quotaValue": "20"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "41s"},
        ],
    }
}

# The real body from a paid project that had spent its day on TTS. Note what it
# is NOT: the quotaId carries no -FreeTier, and the retry window is under two
# hours rather than the time to midnight -- the allowance rolls rather than
# resetting on a calendar day.
_PAID_DAILY_BODY = {
    "error": {
        "code": 429,
        "message": ("You exceeded your current quota. * Quota exceeded for metric: "
                    "generativelanguage.googleapis.com/generate_requests_per_model_per_day, "
                    "limit: 100, model: gemini-3.1-flash-tts. "
                    "Please retry in 1h52m56.409571576s."),
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{
                 "quotaMetric": "generativelanguage.googleapis.com/generate_requests_per_model_per_day",
                 "quotaId": "GenerateRequestsPerDayPerProjectPerModel",
                 "quotaDimensions": {"location": "global", "model": "gemini-3.1-flash-tts"},
                 "quotaValue": "100"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "6776s"},
        ],
    }
}

_PER_MINUTE_BODY = {
    "error": {"code": 429, "message": "You exceeded your current quota", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel",
                         "quotaValue": "10"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"},
    ]}
}


def test_a_daily_quota_is_not_retried_through(monkeypatch):
    """A daily allowance resets on Google's clock, not on ours. Backing off
    through it burns half an hour to arrive at the same 429."""
    from pipeline import QuotaUnavailable, gemini

    monkeypatch.setattr("pipeline.gemini.time.sleep", lambda s: None)
    calls = []

    def always():
        calls.append(1)
        raise _api_error(429, _QUOTA_BODY)

    with pytest.raises(QuotaUnavailable) as caught:
        gemini.call_with_retry(always, RETRY_CFG, "tts-model", "chunk")

    assert len(calls) == 1, "every retry today would fail identically"
    msg = str(caught.value)
    assert "daily quota" in msg
    assert "limit 20" in msg
    assert "retry in about" in msg, (
        "the server names the window; guessing at it got the answer wrong"
    )
    assert "count against the allowance" in msg, (
        "retrying into an exhausted quota spends more of it"
    )
    assert "per Cloud project, not per API key" in msg, (
        "budget on another project does not raise this one, and a new key in "
        "the same project changes nothing"
    )
    assert "already synthesized are kept" in msg


def test_a_per_minute_quota_is_still_waited_out(monkeypatch):
    """The distinction is the whole point: this one does clear."""
    from pipeline import gemini

    monkeypatch.setattr("pipeline.gemini.time.sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise _api_error(429, _PER_MINUTE_BODY)
        return "audio"

    assert gemini.call_with_retry(flaky, RETRY_CFG, "m", "chunk") == "audio"
    assert len(calls) == 2


def test_the_quota_that_failed_is_named_rather_than_the_boilerplate():
    """Raw 429 text is ~400 characters of the same sentence and two URLs, so
    storing it truncates away the only part that says what to do."""
    from pipeline import gemini

    reason = gemini.describe(_api_error(429, _QUOTA_BODY))
    assert "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in reason
    assert "limit 20" in reason and "per day" in reason
    assert len(reason) < 200, "short enough to survive being stored and shown"
    assert "head to: https" not in reason, "the boilerplate is what crowded it out"


def test_a_plain_error_still_describes_itself():
    from pipeline import gemini

    assert gemini.describe(ValueError("bad argument")) == "ValueError: bad argument"


def test_the_shipped_intro_speed_is_applied_only_to_the_intro():
    """A speed on the whole episode would be a different feature entirely."""
    from config import load_config
    from pipeline.assemble import _intro_speed

    cfg = load_config()
    assert cfg["intro"]["speed"] > 1.0, "the disclosure is shipped shortened"
    assert _intro_speed(cfg) == cfg["intro"]["speed"]
    assert "speed" not in cfg["audio"], (
        "the hosts read at their own pace; this knob is the disclosure's alone"
    )


def test_the_script_prompt_asks_for_contractions_concretely():
    """It already said "contractions" -- one word in a list, and scripts came
    back with "it is" anyway. A rule the model can act on needs examples and
    the exception, not a label."""
    body = (Path(__file__).resolve().parents[1] / "prompts" / "script_system.md").read_text()
    style = body[body.index("STYLE:"):]
    assert "it's" in style and "doesn't" in style, "name the forms wanted"
    assert "emphasis" in style, (
        "'it is significant' is right when the sentence leans on the word; a "
        "rule with no exception invites the model to contract that too"
    )

    # A style note near the top of a long prompt is read and then buried under
    # seven segment rules. The forms actually observed slipping are named again
    # as a check to run against the finished draft.
    check = body[body.index("BEFORE YOU RETURN"):]
    for form in ("it is", "that is", "we are", "there is", "does not"):
        assert form in check, f"{form!r} was seen in a real script; name it"
    assert "what it's" in check, (
        "a contraction cannot end a clause -- a blanket rule would produce "
        "'that is what it's', which is not English"
    )


def test_a_revision_still_inherits_the_style_rules():
    """Revisions go through the same system prompt, so the rule cannot be
    stated only in the generation path."""
    body = (Path(__file__).resolve().parents[1] / "prompts" / "script_revise.md").read_text()
    assert "Every constraint in your system instructions still applies" in body


def test_the_script_prompt_asks_for_a_conversation_not_a_relay():
    """Scripts came back in near-even turns, strictly alternating, with the
    second host agreeing every time. The prompt described segments and register
    and said nothing about the shape of the exchange, so the model chose the
    flattest one available."""
    body = (Path(__file__).resolve().parents[1] / "prompts" / "script_system.md").read_text()
    rhythm = body[body.index("RHYTHM"):body.index("HARD CONSTRAINTS")]

    assert "60 to 100 words" in rhythm, (
        "teaching happens in a long turn and cannot happen in a run of "
        "twenty-word exchanges"
    )
    assert "Two turns in a row" in rhythm, "strict alternation is the tell"
    assert "disagree with each other" in rhythm, (
        "hosts who only ever agree are one host in two voices"
    )
    for filler in ("Exactly", "Right", "Absolutely"):
        assert filler in rhythm, f"{filler!r} opened turns in a real script"


def test_short_sentences_are_not_confused_with_short_turns():
    """The two rules pull opposite ways if the distinction is left implicit."""
    body = (Path(__file__).resolve().parents[1] / "prompts" / "script_system.md").read_text()
    assert "not the same as short turns" in body


def test_the_script_prompt_bans_notation_that_cannot_be_read_aloud():
    """Everything here goes to a speech model. "~15%" and "R²" are written
    forms, and a TTS model reading them aloud is at best odd."""
    body = (Path(__file__).resolve().parents[1] / "prompts" / "script_system.md").read_text()
    fmt = body[body.index("OUTPUT FORMAT"):body.index("BEFORE YOU RETURN")]
    assert "read aloud by a speech model" in fmt
    for written_only in ("~15%", "et al.", "vs.", "R²"):
        assert written_only in fmt, f"name {written_only!r} rather than gesturing at symbols"
    assert "p-value" in fmt, "printing one is the common case in an empirical paper"


def test_the_web_artwork_is_web_sized():
    """The feed's 3000px artwork is a megabyte. Serving that to draw an 88px
    square on every page load is the kind of thing nobody notices until the
    site is slow on a phone."""
    import struct

    static = Path(__file__).resolve().parents[1] / "static"
    web, feed = static / "cover-web.png", static / "cover.png"
    assert web.exists(), "rendered by tools/make_cover.py and committed"

    w, h, _, colour = struct.unpack(">IIBB", web.read_bytes()[16:26])
    assert w == h, "square, like the artwork it comes from"
    assert w <= 1024, "a header tile, not a print master"
    assert colour == 2, "RGB, matching the feed artwork"
    assert web.stat().st_size < feed.stat().st_size / 5


def test_the_home_page_does_not_load_the_print_master():
    body = (Path(__file__).resolve().parents[1] / "templates" / "library.html").read_text()
    assert "cover-web.png" in body
    assert "static_url('cover-web.png')" in body, (
        "cache-busted like the other assets, or a re-render serves stale art"
    )
    assert "'cover.png'" not in body and '"cover.png"' not in body


# ------------------------------------------------------ wizard preferences

_PREF_CFG = {
    "models": {"metadata": "flash", "script": "pro", "tts": "tts-1"},
    "voices": {"host_a": "Puck", "host_b": "Kore",
               "choices": ["Puck", "Kore", "Charon"]},
    "script": {"models": ["pro", "flash"], "fallback_model": "flash"},
    "tts": {"models": ["tts-1", "tts-2"]},
}


def test_preferences_start_at_the_configured_defaults(_isolated_db):
    import prefs

    assert prefs.current(_PREF_CFG) == prefs.defaults(_PREF_CFG)
    assert prefs.current(_PREF_CFG)["voice_a"] == "Puck"


def test_a_saved_preference_overrides_the_config(_isolated_db):
    import prefs

    prefs.save({"voice_a": "Charon", "tts_model": "tts-2"}, _PREF_CFG)
    now = prefs.current(_PREF_CFG)
    assert now["voice_a"] == "Charon" and now["tts_model"] == "tts-2"
    assert now["voice_b"] == "Kore", "untouched fields stay on the default"


def test_restoring_defaults_forgets_rather_than_freezes(_isolated_db):
    """Writing the defaults into the database on reset would freeze them: a
    later edit to config.toml would then be silently ignored."""
    import prefs

    prefs.save({"voice_a": "Charon"}, _PREF_CFG)
    prefs.reset()
    assert prefs.current(_PREF_CFG)["voice_a"] == "Puck"

    moved = {**_PREF_CFG, "voices": {**_PREF_CFG["voices"], "host_a": "Kore"}}
    assert prefs.current(moved)["voice_a"] == "Kore", "the config still leads"


def test_a_preference_for_something_retired_is_dropped(_isolated_db):
    """Models get retired and voices get renamed. Carrying a dead one forward
    would fail the episode at its first API call, days after it was chosen."""
    import prefs

    prefs.save({"tts_model": "tts-2"}, _PREF_CFG)
    shrunk = {**_PREF_CFG, "tts": {"models": ["tts-1"]}}
    assert prefs.current(shrunk)["tts_model"] == "tts-1"


def test_an_invalid_choice_is_refused_not_substituted(_isolated_db):
    import prefs

    with pytest.raises(ValueError, match="not one of the choices"):
        prefs.validate({"voice_a": "Gandalf"}, _PREF_CFG)


def test_the_configured_default_is_always_offered(_isolated_db):
    """A config naming a model missing from the lists must still produce a
    usable wizard, not one that cannot express the current setting."""
    import prefs

    odd = {**_PREF_CFG, "models": {**_PREF_CFG["models"], "tts": "tts-9"}}
    assert "tts-9" in prefs.choices(odd)["tts_model"]
    assert prefs.choices(odd)["tts_model"][0] == "tts-9", "default first"


def test_an_episode_keeps_what_it_was_built_with(_isolated_db):
    import db
    import prefs

    db.create_episode("EPIN", "/tmp/p.pdf", "sha-pin")
    prefs.apply_to_episode("EPIN", {"voice_a": "Charon", "tts_model": "tts-2"})
    prefs.save({"voice_a": "Kore"}, _PREF_CFG)   # preferences move on

    row = db.get_episode("EPIN")
    assert prefs.for_episode(row, _PREF_CFG)["voice_a"] == "Charon"
    assert tts.voices_for("EPIN", _PREF_CFG)[0] == "Charon"


def test_the_shipped_voice_list_is_real_ids(_isolated_db):
    """The API rejects a voice name it does not know, and a typo here would
    fail every chunk of every episode built with it."""
    from config import load_config

    cfg = load_config()
    choices = cfg["voices"]["choices"]
    assert len(choices) == 30, "Gemini ships 30 prebuilt voices"
    assert cfg["voices"]["host_a"] in choices
    assert cfg["voices"]["host_b"] in choices
    assert len(set(choices)) == len(choices)


def test_a_paid_daily_cap_is_told_apart_from_a_free_tier_one(monkeypatch):
    """They produce identical 429 prose and need opposite responses: one means
    fix your billing, the other means you are simply out for today."""
    from pipeline import QuotaUnavailable, gemini

    monkeypatch.setattr("pipeline.gemini.time.sleep", lambda s: None)

    def always():
        raise _api_error(429, _PAID_DAILY_BODY)

    with pytest.raises(QuotaUnavailable) as caught:
        gemini.call_with_retry(always, RETRY_CFG, "gemini-3.1-flash-tts", "chunk")

    msg = str(caught.value)
    assert "limit 100" in msg
    assert "billing account" not in msg, "this project is billed; do not send them there"
    assert "paid-tier cap" in msg
    assert "per model" in msg, "another voice model is a bucket that is still full"


def test_the_wait_comes_from_the_server_not_from_a_guess(monkeypatch):
    """An earlier version asserted "resets at midnight Pacific". The real
    payload answers a spent daily quota with under two hours, so the allowance
    rolls rather than resetting on a calendar day."""
    from pipeline import QuotaUnavailable, gemini

    monkeypatch.setattr("pipeline.gemini.time.sleep", lambda s: None)

    def always():
        raise _api_error(429, _PAID_DAILY_BODY)

    with pytest.raises(QuotaUnavailable) as caught:
        gemini.call_with_retry(always, RETRY_CFG, "m", "chunk")

    msg = str(caught.value)
    assert "retry in about 1h52m" in msg
    assert "midnight" not in msg


def test_a_retry_window_reads_as_a_duration():
    from pipeline.gemini import human_delay

    assert human_delay(6776) == "1h52m"
    assert human_delay(7200) == "2h"
    assert human_delay(90) == "1m"
    assert human_delay(0) == ""
    assert human_delay(None) == ""


# ------------------------------------------------- the outline stage

_OUT_CFG = {
    "models": {"script": "pro"},
    "script": {"target_words": 1600, "words_per_minute": 160,
               "fallback_model": "flash",
               "lengths": {"auto": [5, 30], "short": [5, 10], "long": [18, 30]}},
}


def _beats(*pairs):
    return {"why": "because", "beats": [
        {"segment": seg, "covers": "something", "facts": [], "words": words}
        for seg, words in pairs]}


def test_length_is_what_the_beats_add_up_to(_isolated_db):
    """The whole point: nobody picks a duration, so a thin paper is short and
    a rich one is long without either being padded or compressed."""
    from pipeline import outline

    thin = _beats(("Cold open", 200), ("Findings", 700), ("Pressure", 300))
    rich = _beats(("Cold open", 300), ("Setup", 700), ("Identification", 900),
                  ("Findings", 1200), ("Pressure", 700), ("Context", 500),
                  ("So what", 400))

    assert outline.resolve_length(thin, _OUT_CFG, "auto")[0] == 1200   # ~8 min
    assert outline.resolve_length(rich, _OUT_CFG, "auto")[0] == 4700   # ~29 min


def test_a_length_outside_the_range_is_clamped_and_said_so(_isolated_db):
    """A model asked for a number will occasionally return one nobody
    budgeted for, and an episode's length is real money and real daily quota."""
    from pipeline import outline

    absurd = _beats(("Findings", 20000))
    words, note = outline.resolve_length(absurd, _OUT_CFG, "auto")
    assert words == 30 * 160
    assert "above the 30 minute ceiling" in note

    tiny = _beats(("Cold open", 100))
    words, note = outline.resolve_length(tiny, _OUT_CFG, "auto")
    assert words == 5 * 160
    assert "below the 5 minute floor" in note


def test_a_policy_narrows_the_range_rather_than_fixing_a_number(_isolated_db):
    """"short" is not "exactly eight minutes" -- the outline still decides the
    shape, inside a band."""
    from pipeline import outline

    assert outline.policy_range(_OUT_CFG, "short") == (5, 10)
    assert outline.policy_range(_OUT_CFG, "long") == (18, 30)
    assert outline.policy_range(_OUT_CFG, "nonsense") == (5, 30), "falls back to auto"

    middling = _beats(("Findings", 2400))          # 15 min
    assert outline.resolve_length(middling, _OUT_CFG, "short")[0] == 10 * 160
    assert outline.resolve_length(middling, _OUT_CFG, "long")[0] == 18 * 160
    assert outline.resolve_length(middling, _OUT_CFG, "auto")[0] == 2400


_SEGS = ["Cold open", "Setup", "Identification", "Findings", "Pressure",
         "Context", "So what"]


def test_a_malformed_outline_is_rejected_not_half_used(_isolated_db):
    """Half a beat sheet would silently produce half an episode."""
    from pipeline import outline

    with pytest.raises(ValueError):
        outline._parse('{"beats": []}', _SEGS)
    with pytest.raises(ValueError):
        outline._parse('"not an object"', _SEGS)
    with pytest.raises(ValueError, match="none were usable"):
        outline._parse('{"beats": [{"segment": "Nonsense", "covers": "x"}]}', _SEGS)


def test_segment_names_are_canonicalised(_isolated_db):
    """"Cold Open" and "Cold open" grouping separately would quietly make two
    segments out of one."""
    from pipeline import outline

    parsed = outline._parse(
        '{"beats": [{"segment": "cold OPEN", "covers": "x", "words": 200}]}', _SEGS)
    assert parsed["beats"][0]["segment"] == "Cold open"


def test_the_brief_carries_the_facts_that_must_land(_isolated_db):
    """This is what stops the script wandering: not "cover the findings" but
    the specific number it has to reach."""
    from pipeline import outline

    plan = {"why": "", "beats": [
        {"segment": "Findings", "covers": "The employment effect",
         "facts": ["about three percent, a fifth of the raw gap"], "words": 400}]}
    brief = outline.as_brief(plan)
    assert "Findings:" in brief
    assert "(400 words)" in brief
    assert "must land: about three percent" in brief


def test_a_script_without_an_outline_still_has_a_length(_isolated_db):
    """Episodes built before this stage existed, and any retry that skips it,
    must not fall through to zero."""
    import db
    from pipeline import script

    db.create_episode("ENOOUT", "/tmp/n.pdf", "sha-noout")
    row = db.get_episode("ENOOUT")
    assert script._target_words(row, _OUT_CFG) == 1600
    assert "No beat sheet" in script._brief(row)

    db.update_episode("ENOOUT", target_words=2400)
    assert script._target_words(db.get_episode("ENOOUT"), _OUT_CFG) == 2400


def test_the_two_prompts_agree_on_the_arc():
    """The segment names live in three places -- the outline prompt, the script
    prompt and outline.SEGMENTS. Drift between them would have the writer
    ignoring beats it does not recognise."""
    from pipeline import arc

    # Both prompts splice the same file, so the only way they can disagree is
    # if one stops splicing it.
    root = Path(__file__).resolve().parents[1] / "prompts"
    for name in ("outline.md", "script_system.md"):
        assert "$ARC" in root.joinpath(name).read_text(), f"{name} lost its arc"
    for kind in arc.KINDS:
        names = arc.segments(kind)
        assert len(names) == 7, f"{kind} arc parsed as {names}"
        assert names[0] == "Cold open"


# ------------------------------------------------- research dossier and arcs

def test_an_entry_without_a_source_is_dropped(_isolated_db):
    """The rule the whole stage rests on. An unsourced entry reads exactly like
    a sourced one once it is in the dossier, so it cannot be flagged for later
    -- it has to not be there."""
    from pipeline import dossier

    data = dossier._parse(json.dumps({"reception": "mixed", "entries": [
        {"who": "Karl Popper", "what": "objected to irrationalism",
         "kind": "critic", "source": "https://example.org/popper"},
        {"who": "Somebody", "what": "said a thing", "kind": "critic"},
        {"who": "Nobody", "what": "said another", "kind": "critic", "source": ""},
    ]}))

    assert [e["who"] for e in data["entries"]] == ["Karl Popper"]
    assert data["dropped"] == 2


def test_a_source_must_be_a_real_link(_isolated_db):
    """This value is rendered as an href on an admin page, so a javascript:
    entry would be script injection by way of a research note."""
    from pipeline import dossier

    data = dossier._parse(json.dumps({"entries": [
        {"who": "A", "what": "x", "source": "javascript:alert(1)"},
        {"who": "B", "what": "y", "source": "not a url"},
        {"who": "C", "what": "z", "source": "https://ok.example/page"},
    ]}))
    assert [e["who"] for e in data["entries"]] == ["C"]


def test_the_dossier_corroborates_names_the_paper_never_mentions(_isolated_db):
    """Otherwise every critic the research turned up flags as a possible
    fabrication, and a flag list that is mostly noise is one nobody reads."""
    from pipeline import dossier, script

    data = {"reception": "", "entries": [
        {"who": "Imre Lakatos", "what": "proposed research programmes",
         "kind": "critic", "source": "https://example.org/l"}]}
    corpus = dossier.corroboration(data)

    flags = script.citation_flags(
        "HOST_A: Lakatos (1970) pushed back on exactly that.",
        paper_text="nothing about that here", grounding_text=corpus)
    assert flags and all(f["in_paper"] for f in flags)
    assert flags[0]["source"] == "web"


def test_research_only_runs_when_it_was_asked_for(_isolated_db):
    """Off by default: a search-grounded call on top of an already expensive
    pipeline, and most papers do not need it."""
    import db
    from pipeline import dossier

    db.create_episode("ENORES", "/tmp/n.pdf", "sha-nores")
    assert not dossier.wanted(db.get_episode("ENORES"))
    dossier.research("ENORES", {})          # no PDF, no client: proves it returned

    db.update_episode("ENORES", research="on")
    assert dossier.wanted(db.get_episode("ENORES"))


def test_the_two_arcs_differ_where_it_matters(_isolated_db):
    """A book of philosophy has no identification strategy and no effect sizes.
    Asking it for them produces an episode about nothing."""
    from pipeline import arc

    empirical = arc.segments(arc.EMPIRICAL)
    theoretical = arc.segments(arc.THEORETICAL)

    assert "Identification" in empirical and "Findings" in empirical
    assert "Identification" not in theoretical and "Findings" not in theoretical
    assert "The move" in theoretical and "The case" in theoretical
    # Both keep the beats that are about the listener rather than the method.
    for shared in ("Cold open", "Pressure", "Context", "So what"):
        assert shared in empirical and shared in theoretical

    assert "magnitudes" in arc.text(arc.EMPIRICAL)
    assert "magnitudes" not in arc.text(arc.THEORETICAL)


def test_an_unknown_kind_falls_back_to_the_empirical_arc(_isolated_db):
    """Most uploads are papers, and that is the arc with mileage on it."""
    import db
    from pipeline import arc

    db.create_episode("EKIND", "/tmp/k.pdf", "sha-kind")
    assert arc.kind_of(db.get_episode("EKIND")) == arc.EMPIRICAL

    db.update_principal("EKIND", work_kind="philosophy")
    assert arc.kind_of(db.get_episode("EKIND")) == arc.EMPIRICAL
    assert arc.clean_kind("THEORETICAL") == arc.THEORETICAL

    db.update_principal("EKIND", work_kind="theoretical")
    assert arc.kind_of(db.get_episode("EKIND")) == arc.THEORETICAL


def test_the_dossier_prompt_forbids_quotation():
    """Putting invented words in the mouth of a named, often living, academic
    is the highest-risk thing this system could do, and it ships as audio."""
    body = (Path(__file__).resolve().parents[1] / "prompts" / "dossier.md").read_text()
    assert "Never invent a quotation" in body
    assert "source" in body.casefold()


def test_a_script_with_no_research_is_told_to_hedge(_isolated_db):
    import db
    from pipeline import script

    db.create_episode("ENOD", "/tmp/d.pdf", "sha-nod")
    brief = script._dossier_brief(db.get_episode("ENOD"))
    assert "No research was done" in brief and "Hedge" in brief


# ------------------------------------------------- papers, split from episodes

def test_an_episode_always_has_a_paper(_isolated_db):
    """The two tables are only safe to read separately if the join is never
    missing. Every route into create_episode has to produce one."""
    import db

    db.create_episode("EPAP", "/tmp/p.pdf", "sha-pap")
    papers = db.papers_for("EPAP")
    assert len(papers) == 1
    assert papers[0]["sha256"] == "sha-pap"
    assert papers[0]["role"] == db.PRINCIPAL
    assert db.principal_paper("EPAP")["id"] == papers[0]["id"]


def test_paper_facts_are_refused_by_update_episode(_isolated_db):
    """The pre-split columns are still on the episode table, so a stray write
    would land somewhere real, succeed, and never be read again. It has to
    fail loudly or it fails silently."""
    import db
    import pytest

    db.create_episode("EGUARD", "/tmp/g.pdf", "sha-guard")
    with pytest.raises(ValueError) as exc:
        db.update_episode("EGUARD", title="Wrong Door")
    assert "update_principal" in str(exc.value)
    # And the episode's own fields still go through.
    db.update_episode("EGUARD", status="done")
    assert db.get_episode("EGUARD")["status"] == "done"


def test_the_episode_reads_its_principals_facts(_isolated_db):
    import db

    db.create_episode("EREAD", "/tmp/r.pdf", "sha-read")
    db.update_principal("EREAD", title="A Paper", year=1994)
    row = db.get_episode("EREAD")
    assert row["title"] == "A Paper" and row["year"] == 1994
    assert row["paper_id"] == db.principal_paper("EREAD")["id"]


def test_a_correction_reaches_every_episode_on_that_paper(_isolated_db):
    """The point of the split. Two renderings of one work share the row the
    title lives in, so fixing a botched extraction fixes both."""
    import db

    paper = db.create_paper(source_path="/tmp/s.pdf", sha256="sha-shared")
    db.create_episode("EA", "/tmp/s.pdf", "sha-shared", papers=[paper])
    db.create_episode("EB", "/tmp/s.pdf", "sha-shared", papers=[paper])
    db.update_principal("EA", title="Corrected Title")
    assert db.get_episode("EB")["title"] == "Corrected Title"


def test_siblings_need_the_whole_paper_set_to_match(_isolated_db):
    """Overlap is not sameness. A comparison of two papers and a solo episode
    about one of them share a paper, but publishing one must not unpublish the
    other -- they are different episodes, not two takes on the same one."""
    import db

    one = db.create_paper(source_path="/tmp/1.pdf", sha256="sha-1")
    two = db.create_paper(source_path="/tmp/2.pdf", sha256="sha-2")
    db.create_episode("SOLO", "/tmp/1.pdf", "sha-1", papers=[one])
    db.create_episode("PAIR", "/tmp/1.pdf", "sha-1", papers=[one, two])
    db.create_episode("PAIR2", "/tmp/1.pdf", "sha-1", papers=[one, two])

    assert [s["id"] for s in db.siblings("SOLO")] == []
    assert [s["id"] for s in db.siblings("PAIR")] == ["PAIR2"]

    db.update_episode("PAIR2", published=1)
    assert db.demote_siblings("PAIR") == ["PAIR2"]
    assert db.get_episode("PAIR2")["published"] == 0


def test_deleting_an_episode_keeps_a_paper_something_else_uses(_isolated_db):
    """A shared paper is one file on disk. Deleting one of the episodes that
    point at it must not take the bytes out from under the others."""
    import db

    paper = db.create_paper(source_path="/tmp/k.pdf", sha256="sha-keep")
    db.create_episode("EK1", "/tmp/k.pdf", "sha-keep", papers=[paper])
    db.create_episode("EK2", "/tmp/k.pdf", "sha-keep", papers=[paper])

    assert db.delete_episode("EK1") == [], "still referenced, so not an orphan"
    assert db.get_paper(paper) is not None
    assert db.delete_episode("EK2") == [paper], "now nothing points at it"
    assert db.get_paper(paper) is None


def test_the_migration_gives_old_episodes_the_paper_they_implied(tmp_path,
                                                                monkeypatch):
    """An existing library has to survive the upgrade, and it has to survive it
    without moving any files: the new paper takes the episode's own id, which
    is what the stored PDF is already named."""
    import sqlite3
    import db

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "old.db")
    db._local.__dict__.clear()

    # A pre-split database: episode table only, paper facts on the row.
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("CREATE TABLE episode (id TEXT PRIMARY KEY, created_at TEXT NOT NULL,"
                 " source_path TEXT NOT NULL, sha256 TEXT, title TEXT, status TEXT NOT NULL)")
    conn.execute("INSERT INTO episode VALUES ('OLD1', '2026-01-01T00:00:00+00:00',"
                 " '/tmp/old.pdf', 'sha-old', 'An Older Paper', 'done')")
    conn.commit()
    conn.close()
    db._local.__dict__.clear()

    db.init_db()
    row = db.get_episode("OLD1")
    assert row["title"] == "An Older Paper", "the facts came across"
    assert row["paper_id"] == "OLD1", "and the stored PDF keeps its filename"
    assert db.paper_pdf("OLD1").name == "OLD1.pdf"

    # Idempotent: a second start must not deal it a second paper.
    db.init_db()
    assert len(db.papers_for("OLD1")) == 1


def test_every_attached_paper_is_sent_to_the_script_model(tmp_path, monkeypatch):
    """The comparison episodes this split exists for are worthless if only the
    first paper reaches the writer -- and a summary of the second is not the
    same thing as the second, which is the whole reason the PDFs go natively."""
    import db
    from pipeline import script as script_mod

    papers = tmp_path / "papers"
    papers.mkdir()
    monkeypatch.setattr(db, "PAPERS_DIR", papers)

    first = db.create_paper(source_path="/tmp/1.pdf", sha256="sha-m1")
    second = db.create_paper(source_path="/tmp/2.pdf", sha256="sha-m2")
    db.create_episode("EMULTI", "/tmp/1.pdf", "sha-m1", papers=[first, second])
    for paper_id in (first, second):
        db.paper_pdf(paper_id).write_bytes(b"%PDF-1.4 fake")

    sent = []

    class Resp:
        text = "HOST_A: Hello there everyone.\nHOST_B: Good to be here."
        usage_metadata = None
        candidates = []

    def fake_generate_content(*, model, contents, config):
        sent.append([c for c in contents if str(c).startswith("PART:")])
        return Resp()

    monkeypatch.setattr(script_mod, "client",
                        lambda: type("C", (), {"models": type("M", (), {
                            "generate_content": staticmethod(fake_generate_content)})()})())
    monkeypatch.setattr(script_mod, "pdf_part", lambda p: f"PART:{p.name}")

    cfg = {"models": {"script": "m"}, "script": {"target_words": 1600},
           "retry": {"attempts": 1, "base_delay_s": 0, "max_delay_s": 0}}
    script_mod.generate_script("EMULTI", cfg)

    assert sent == [[f"PART:{first}.pdf", f"PART:{second}.pdf"]], (
        "both papers, in running order")
