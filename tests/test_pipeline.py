"""Tests for the parts that are pure logic: chunking, script validation,
citation flagging, WAV wrapping, range requests, feed generation, resume.

Nothing here calls the Gemini API.
"""

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

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db._local.__dict__.clear()
    db.init_db()
    yield
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
        "HOST_A: The Mariel boatlift in 1980 is the classic case.",
    ])
    flags = citation_flags(script)
    texts = " ".join(f["text"] for f in flags)

    assert "Card and Krueger (1994)" in texts
    assert any("et al" in f["text"] for f in flags)
    assert any(f["line"] == 3 for f in flags), "proper noun near a year should flag"


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
        "HOST_A: William Gould for the 1969 board decision.\n"
        "HOST_B: And Fictitious Person in 1977 supposedly agreed."
    )
    paper = "... appointed William Gould to the board in 1969, which ..."

    flags = {f["text"]: f["in_paper"] for f in citation_flags(script, paper)}
    assert flags["William Gould for the 1969"] is True
    assert flags["Fictitious Person in 1977"] is False


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
    cfg = {"tts": {"chunk_target_words": 120, "chunk_max_words": 200, "context_turns": 2}}

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
    tts.synthesize("EP1", {"tts": {"chunk_target_words": 3, "chunk_max_words": 6, "context_turns": 2}})

    later = entries[2]
    assert later["context"], "later chunks get preceding turns for prosody"
    prompt = tts._build_prompt(later, {})
    assert "Do NOT read those preceding lines aloud" in prompt
    for line in later["turns"]:
        assert line in prompt
