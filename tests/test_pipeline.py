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
    db.update_episode("EP1", title="Minimum Wages and Employment", year=1994)
    db.update_episode("EP2", title="Minimum Wages and Employment", year=2019)

    assert db.find_by_sha("hash-aaa")["id"] == "EP1"
    assert db.find_by_sha("hash-bbb")["id"] == "EP2"
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
    (papers / "ERET.pdf").write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(script_mod, "PAPERS_DIR", papers)

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
    db.update_episode("ECITE", title="A Paper", cited_by=12,
                      cited_by_source="entered by hand")

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
    db.update_episode("EINTRO", title="Minimum Wages and Employment",
                      authors=json.dumps(["David Card", "Alan Krueger"]))

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
    db.update_episode("EINTRO3", title="A Paper With No Byline")

    text = intro.intro_text(db.get_episode("EINTRO3"), _intro_cfg())
    assert "uncredited" not in text
    assert text.endswith("A Paper With No Byline.")


def test_intro_does_not_double_the_full_stop_after_et_al(_isolated_db):
    import db
    from pipeline import intro

    db.create_episode("EINTRO4", "/tmp/i.pdf", "sha-intro4")
    db.update_episode("EINTRO4", title="Attention Is All You Need",
                      authors=json.dumps(["A", "B", "C", "D", "E"]))

    text = intro.intro_text(db.get_episode("EINTRO4"), _intro_cfg())
    assert text.endswith("by A et al.")
    assert ".." not in text


def test_intro_uses_one_voice_that_is_neither_host(_isolated_db, monkeypatch):
    """The multi-speaker API takes exactly two speakers, so a third voice has
    to come from a separate single-voice call."""
    import db
    from pipeline import intro

    db.create_episode("EINTRO5", "/tmp/i.pdf", "sha-intro5")
    db.update_episode("EINTRO5", title="A Paper", authors=json.dumps(["Solo"]))
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
    assert "This is a Paperpod" in captured["contents"]
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
    db.update_episode("EINTRO6", title="First Title", authors=json.dumps(["Solo"]))
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

    db.update_episode("EINTRO6", title="Second Title")
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
    db.update_episode("EINTRO7", title="Some Important Paper",
                      authors=json.dumps(["Ada Lovelace"]))

    cfg = load_config()
    text = intro.intro_text(db.get_episode("EINTRO7"), cfg)
    assert text.startswith("This is a Paperpod, an AI generated podcast")
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
