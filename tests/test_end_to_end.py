"""End-to-end pipeline and web-app tests.

Only the three Gemini calls are stubbed. ffmpeg assembly, loudness
normalization, MP3 encoding, HTTP range serving, and RSS generation all run for
real, so these cover acceptance criteria 1, 2, 3, 5 and 6.
"""

import json
import math
import queue
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

# Captured before any test monkeypatches it onto the module.
from pipeline import tts as _tts_module  # noqa: E402

_ORIGINAL_SYNTH_CHUNK = _tts_module._synthesize_chunk

SAMPLE_SCRIPT = "\n".join([
    "HOST_A: Here's a question worth caring about: does raising the minimum wage cost people jobs?",
    "HOST_B: " + " ".join(["Everyone assumed it did, and the theory is clean."] * 12),
    "HOST_A: " + " ".join(["So how did they actually get leverage on that question?"] * 12),
    "HOST_B: " + " ".join(["They compared neighboring counties across a state border."] * 12),
    "HOST_A: " + " ".join(["The effect is about three percent, which is small."] * 12),
    "HOST_B: I'd read this as suggestive rather than settled, honestly.",
])


def _write_sine_wav(path: Path, seconds: float = 1.5, rate: int = 24000, freq: float = 220.0):
    frames = bytearray()
    for i in range(int(rate * seconds)):
        v = int(12000 * math.sin(2 * math.pi * freq * i / rate))
        frames += struct.pack("<h", v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def _make_pdf(path: Path, pages: int = 3, text: str | None = None):
    import fitz

    body = text if text is not None else (
        "Minimum Wages and Employment: A Case Study\n\n"
        + ("We study the effect of minimum wage increases on employment using a "
           "border-discontinuity design across contiguous county pairs. " * 12)
    )
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 750), body if i == 0 else "Further analysis. " * 40, fontsize=9)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated data dirs + DB, with all Gemini calls stubbed."""
    import db
    from pipeline import assemble, ingest, script as script_mod, tts

    papers = tmp_path / "papers"
    chunks = tmp_path / "chunks"
    final = tmp_path / "final"
    for d in (papers, chunks, final):
        d.mkdir()

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db._local.__dict__.clear()
    db.init_db()

    # WORK_Q is module-level and outlives a test; a leftover entry makes the
    # next test's queue assertions read someone else's episode.
    import app as app_mod
    while not app_mod.WORK_Q.empty():
        try:
            app_mod.WORK_Q.get_nowait()
        except queue.Empty:
            break

    monkeypatch.setattr(ingest, "PAPERS_DIR", papers)
    monkeypatch.setattr(script_mod, "PAPERS_DIR", papers)
    monkeypatch.setattr(tts, "CHUNKS_DIR", chunks)
    monkeypatch.setattr(assemble, "CHUNKS_DIR", chunks)
    monkeypatch.setattr(assemble, "FINAL_DIR", final)

    calls = {"metadata": 0, "script": 0, "tts": 0}

    def fake_metadata(episode_id, cfg):
        calls["metadata"] += 1
        db.update_episode(
            episode_id,
            title="Minimum Wages and Employment",
            authors=json.dumps(["David Card", "Alan Krueger"]),
            year=1994,
            abstract="We study the effect of minimum wage increases on employment.",
            venue="American Economic Review",
        )

    def fake_script(episode_id, cfg):
        calls["script"] += 1
        return SAMPLE_SCRIPT

    def fake_chunk(episode_id, entry, wav_path, cfg):
        calls["tts"] += 1
        _write_sine_wav(wav_path, freq=200 + 40 * entry["seq"])

    monkeypatch.setattr(ingest, "extract_metadata", fake_metadata)
    monkeypatch.setattr(script_mod, "generate_script", fake_script)
    monkeypatch.setattr(tts, "_synthesize_chunk", fake_chunk)

    # run.py bound ingest.extract_metadata into STAGES at import time; rebind.
    from pipeline import run as run_mod
    monkeypatch.setattr(
        run_mod, "STAGES",
        [
            ("extracting", fake_metadata),
            ("scripting", run_mod._run_scripting),
            ("synthesizing", tts.synthesize),
            ("assembling", assemble.assemble),
        ],
    )

    cfg = {
        "models": {"metadata": "m", "script": "s", "tts": "t"},
        "voices": {"host_a": "Puck", "host_b": "Kore"},
        "script": {"target_words": 1600, "max_pages": 120},
        "tts": {"chunk_target_words": 60, "chunk_max_words": 120, "context_turns": 2},
        "audio": {"seam_silence_ms": 250, "lufs_target": -16.0, "true_peak": -1.5,
                  "lra": 11.0, "bitrate": "96k"},
        "server": {"base_url": "http://paperpod.test:8000", "port": 8000},
        "feed": {"title": "Paperpod", "description": "Test feed", "author": "Paperpod"},
        "costs": {},
    }
    return {"tmp": tmp_path, "cfg": cfg, "calls": calls, "papers": papers,
            "chunks": chunks, "final": final}


# ---------------------------------------------------------------- ingest

def test_ingest_rejects_scanned_pdf(env, tmp_path):
    from pipeline import PipelineError, ingest

    pdf = tmp_path / "scan.pdf"
    _make_pdf(pdf, pages=2, text="Fig 1.")  # almost no text layer
    with pytest.raises(PipelineError, match="scanned"):
        ingest.ingest_pdf(pdf, env["cfg"])


def test_ingest_rejects_overlong_pdf(env, tmp_path):
    from pipeline import PipelineError, ingest

    pdf = tmp_path / "long.pdf"
    _make_pdf(pdf, pages=8)
    cfg = {**env["cfg"], "script": {"target_words": 1600, "max_pages": 5}}
    with pytest.raises(PipelineError, match="pages"):
        ingest.ingest_pdf(pdf, cfg)


def test_rejected_pdf_is_visible_as_failed_episode(env, tmp_path):
    import db
    from pipeline import PipelineError, ingest

    pdf = tmp_path / "scan.pdf"
    _make_pdf(pdf, pages=1, text="x")
    with pytest.raises(PipelineError):
        ingest.ingest_pdf(pdf, env["cfg"])

    rows = db.list_episodes()
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert "scanned" in rows[0]["error"]


def test_duplicate_pdf_is_skipped(env, tmp_path):
    import db
    from pipeline import ingest

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    first = ingest.ingest_pdf(pdf, env["cfg"])
    assert first is not None

    copy = tmp_path / "paper-renamed.pdf"
    shutil.copy2(pdf, copy)
    from pipeline import DuplicateEpisode

    with pytest.raises(DuplicateEpisode) as exc:
        ingest.ingest_pdf(copy, env["cfg"])
    assert exc.value.episode_id == first, "points at the episode it already is"
    assert len(db.list_episodes()) == 1, "same bytes = same episode"


def test_ingest_never_mutates_source(env, tmp_path):
    from pipeline import ingest

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    before = pdf.read_bytes()
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])

    assert pdf.exists() and pdf.read_bytes() == before
    assert (env["papers"] / f"{episode_id}.pdf").exists()


# ------------------------------------------------------------- full run

@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_full_pipeline_produces_playable_mp3(env, tmp_path):
    """Acceptance criterion 1: a PDF in produces a playable MP3, no hands."""
    import db
    from pipeline import ingest, run

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])
    run.run_episode(episode_id, env["cfg"])

    row = db.get_episode(episode_id)
    assert row["status"] == "done", row["error"]
    assert row["script_md"] == SAMPLE_SCRIPT

    mp3 = env["final"] / f"{episode_id}.mp3"
    assert mp3.exists() and mp3.stat().st_size > 1000
    assert row["duration_s"] and row["duration_s"] > 1

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_name,channels,sample_rate", "-of", "json", str(mp3)],
        capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "mp3"
    assert stream["channels"] == 1
    assert stream["sample_rate"] == "44100"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_id3_tags_are_written(env, tmp_path):
    import db
    from pipeline import ingest, run

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])
    run.run_episode(episode_id, env["cfg"])

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json",
         str(env["final"] / f"{episode_id}.mp3")],
        capture_output=True, text=True,
    )
    tags = {k.lower(): v for k, v in json.loads(probe.stdout)["format"].get("tags", {}).items()}
    assert tags.get("title") == "Minimum Wages and Employment"
    assert "Card" in tags.get("artist", "")


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_seam_silence_is_inserted_between_chunks(env, tmp_path):
    """Chunk seams get configurable silence; total duration reflects it."""
    import db
    from pipeline import ingest, run

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])

    cfg = {**env["cfg"], "audio": {**env["cfg"]["audio"], "seam_silence_ms": 400}}
    run.run_episode(episode_id, cfg)

    n_chunks = len(list((env["chunks"] / episode_id).glob("[0-9][0-9][0-9].wav")))
    assert n_chunks > 1
    duration = db.get_episode(episode_id)["duration_s"]
    expected_audio = 1.5 * n_chunks
    expected_silence = 0.4 * (n_chunks - 1)
    assert duration == pytest.approx(expected_audio + expected_silence, abs=0.5)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_failed_chunk_becomes_a_gap_not_a_failed_episode(env, tmp_path):
    """Failure mode 3: one bad chunk must not sink the whole episode."""
    import db
    from pipeline import ingest, run, tts

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])

    original = tts._synthesize_chunk

    def flaky(eid, entry, wav_path, cfg):
        if entry["seq"] == 1:
            raise RuntimeError("500: model returned text tokens")
        original(eid, entry, wav_path, cfg)

    tts._synthesize_chunk = flaky
    try:
        run.run_episode(episode_id, env["cfg"])
    finally:
        tts._synthesize_chunk = original

    row = db.get_episode(episode_id)
    assert row["status"] == "done", "a single bad chunk must not fail the episode"
    assert (env["final"] / f"{episode_id}.mp3").exists()

    details = " ".join(s["detail"] or "" for s in db.get_stage_log(episode_id))
    assert "1" in details and "gap" in details.lower(), "the gap must be logged"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_truncated_tail_of_chunks_is_detected(env, tmp_path):
    """The failure that shipped a 3-minute episode: when the LAST chunks fail,
    the surviving sequence numbers look like a complete script, so a short
    episode assembled with nothing reported."""
    import db
    from pipeline import ingest, run, tts

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])

    original = tts._synthesize_chunk

    def only_first_two(eid, entry, wav_path, cfg):
        if entry["seq"] >= 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        original(eid, entry, wav_path, cfg)

    tts._synthesize_chunk = only_first_two
    try:
        run.run_episode(episode_id, env["cfg"])
    finally:
        tts._synthesize_chunk = _ORIGINAL_SYNTH_CHUNK

    row = db.get_episode(episode_id)
    assert row["status"] == "done", "the surviving audio still assembles"

    details = " ".join(s["detail"] or "" for s in db.get_stage_log(episode_id))
    assert "INCOMPLETE" in details, "a truncated tail must be reported"
    assert "of" in details and "chunks" in details

    n_on_disk = len(list((env["chunks"] / episode_id).glob("[0-9][0-9][0-9].wav")))
    assert n_on_disk == 2
    manifest = json.loads((env["chunks"] / episode_id / "manifest.json").read_text())
    assert len(manifest) > 2, "the script needed more chunks than were produced"


def test_expected_chunk_count_falls_back_without_a_manifest(tmp_path):
    from pipeline.assemble import _expected_chunks

    assert _expected_chunks(tmp_path, fallback=3) == 3
    (tmp_path / "manifest.json").write_text("not json")
    assert _expected_chunks(tmp_path, fallback=3) == 3
    (tmp_path / "manifest.json").write_text(json.dumps([{"seq": i} for i in range(7)]))
    assert _expected_chunks(tmp_path, fallback=3) == 7


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_incomplete_audio_is_flagged_on_the_episode_page(client, env, tmp_path):
    import db
    from pipeline import ingest, run, tts

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])
    original = tts._synthesize_chunk
    tts._synthesize_chunk = (
        lambda eid, entry, wav_path, cfg:
        original(eid, entry, wav_path, cfg) if entry["seq"] < 2
        else (_ for _ in ()).throw(RuntimeError("boom"))
    )
    try:
        run.run_episode(episode_id, env["cfg"])
    finally:
        tts._synthesize_chunk = _ORIGINAL_SYNTH_CHUNK

    html = client.get(f"/episode/{episode_id}").text
    assert "Audio is incomplete" in html
    assert "INCOMPLETE" in html


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_resume_after_mid_tts_crash_does_not_regenerate_script(env, tmp_path):
    """Acceptance criterion 2: kill mid-TTS, restart, and the script is not
    regenerated while completed chunks are reused."""
    import db
    from pipeline import ingest, run, tts

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])

    original = tts._synthesize_chunk
    synthesized = []

    def crash_on_third(eid, entry, wav_path, cfg):
        if len(synthesized) >= 2:
            raise KeyboardInterrupt("simulated kill -9 mid-TTS")
        synthesized.append(entry["seq"])
        original(eid, entry, wav_path, cfg)

    tts._synthesize_chunk = crash_on_third
    try:
        with pytest.raises(KeyboardInterrupt):
            run.run_episode(episode_id, env["cfg"])
    finally:
        tts._synthesize_chunk = original

    assert env["calls"]["script"] == 1
    partial = sorted((env["chunks"] / episode_id).glob("[0-9][0-9][0-9].wav"))
    assert len(partial) == 2

    # Restart: run.py resumes at the stage that was in flight.
    row = db.get_episode(episode_id)
    resume_from = run.resume_stage_for(row["status"])
    assert resume_from == "synthesizing"

    before = {p.name: p.read_bytes() for p in partial}
    run.run_episode(episode_id, env["cfg"], from_stage=resume_from)

    assert env["calls"]["script"] == 1, "scripting must not re-run on resume"
    assert db.get_episode(episode_id)["status"] == "done"
    for name, data in before.items():
        assert (env["chunks"] / episode_id / name).read_bytes() == data, (
            f"chunk {name} was regenerated instead of reused"
        )


def test_zero_quota_aborts_instead_of_gapping_every_chunk(env, tmp_path):
    """A plan with no allowance fails identically on every chunk, so the run
    must stop with an actionable message rather than emit an empty episode."""
    import db
    from pipeline import QuotaUnavailable, ingest, run, tts

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])

    attempts = []

    def no_quota(eid, entry, wav_path, cfg):
        attempts.append(entry["seq"])
        raise QuotaUnavailable("gemini-3.1-flash-tts-preview has no quota on this plan")

    tts._synthesize_chunk = no_quota
    try:
        run.run_episode(episode_id, env["cfg"])
    finally:
        tts._synthesize_chunk = _ORIGINAL_SYNTH_CHUNK

    row = db.get_episode(episode_id)
    assert row["status"] == "failed"
    assert "no quota" in row["error"]
    assert attempts == [0], "must stop after the first chunk, not try them all"
    assert not (env["final"] / f"{episode_id}.mp3").exists()


def test_rate_limited_stage_recovers_instead_of_failing(env, tmp_path, monkeypatch):
    """A 429 during scripting used to kill the episode outright."""
    import db
    from google.genai import errors

    from pipeline import ingest, run, script as script_mod
    from pipeline.gemini import call_with_retry

    monkeypatch.setattr("pipeline.gemini.time.sleep", lambda s: None)

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])

    calls = []
    body = {"error": {"code": 429, "message": "slow down",
                      "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                   "retryDelay": "12s"}]}}

    def flaky_script(eid, cfg):
        def once():
            calls.append(1)
            if len(calls) < 2:
                raise errors.APIError(429, body)
            return SAMPLE_SCRIPT
        return call_with_retry(once, cfg, "m", "script")

    monkeypatch.setattr(script_mod, "generate_script", flaky_script)
    monkeypatch.setattr(script_mod, "generate_title", lambda *a, **k: "A Title")
    run.run_episode(episode_id, env["cfg"], from_stage="scripting")

    row = db.get_episode(episode_id)
    assert row["status"] == "done", row["error"]
    assert len(calls) == 2, "the rate limit was ridden out, not surfaced"


def test_cost_is_accumulated_and_visible(env, tmp_path):
    """Acceptance criterion 6."""
    import db
    from pipeline import ingest

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])
    db.add_cost(episode_id, 0.0021)
    db.add_cost(episode_id, 0.3400)

    import app as app_mod
    view = app_mod._episode_view(db.get_episode(episode_id))
    assert view["cost_usd"] == pytest.approx(0.3421)


# ------------------------------------------------------------ web routes

@pytest.fixture
def client(env, tmp_path, monkeypatch):
    """TestClient with the worker/watcher threads disabled."""
    from fastapi.testclient import TestClient

    import app as app_mod

    monkeypatch.setattr(app_mod, "CFG", env["cfg"])
    monkeypatch.setattr(app_mod, "FINAL_DIR", env["final"])
    monkeypatch.setattr(app_mod, "PAPERS_DIR", env["papers"])
    monkeypatch.setattr(app_mod, "CHUNKS_DIR", env["chunks"])
    monkeypatch.setattr(app_mod, "_worker", lambda: None)
    monkeypatch.setattr(app_mod, "_watch_inbox", lambda: None)
    with TestClient(app_mod.app) as c:
        yield c


def _done_episode(env, tmp_path, title="Minimum Wages and Employment"):
    import db
    from pipeline import ingest, run

    pdf = tmp_path / f"{title[:8].replace(' ', '_')}.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])
    run.run_episode(episode_id, env["cfg"])
    return episode_id


def test_static_assets_are_cache_busted(client):
    """A stale cached app.js looks identical to a broken feature, so asset
    URLs must change when the file does."""
    import re as _re

    html = client.get("/").text
    css = _re.search(r'href="(/static/style\.css\?v=\d+)"', html)
    js = _re.search(r'src="(/static/app\.js\?v=\d+)"', html)
    assert css and js, "style.css and app.js must carry a version query"
    assert client.get(css.group(1)).status_code == 200
    assert client.get(js.group(1)).status_code == 200


def test_static_version_changes_with_the_file(tmp_path, monkeypatch):
    import app as app_mod

    first = app_mod.static_url("app.js")
    (app_mod.ROOT / "static" / "app.js").touch()
    assert app_mod.static_url("app.js") != first or first.endswith("v=0")


def test_favicon_is_declared(client):
    assert 'rel="icon"' in client.get("/").text


def test_health_route(client):
    body = client.get("/health").json()
    assert "queue_depth" in body and "worker_alive" in body


def _publish(episode_id):
    import db
    db.update_episode(episode_id, published=1, status="done", flags_reviewed=1)


def test_library_lists_episodes(client, env, tmp_path):
    import db
    from pipeline import ingest

    episode_id = ingest.ingest_pdf((lambda p: (_make_pdf(p), p)[1])(tmp_path / "a.pdf"), env["cfg"])
    db.update_episode(episode_id, title="A Paper About Wages",
                      authors=json.dumps(["Jane Roe"]), status="done")

    html = client.get("/admin").text
    assert "A Paper About Wages" in html
    assert "Jane Roe" in html


def test_library_shows_summary_and_play_button(client, env, tmp_path):
    import db
    from pipeline import ingest

    pdf = tmp_path / "a.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])
    db.update_episode(
        episode_id, title="Border Counties", status="done",
        summary="Raising the minimum wage barely moved employment.",
        audio_path=str(pdf), duration_s=612.0,
    )

    _publish(episode_id)
    html = client.get("/").text
    assert "Raising the minimum wage barely moved employment." in html
    assert f'data-src="/episode/{episode_id}/audio"' in html, "inline play button"
    assert "10:12" in html


def test_library_falls_back_to_abstract_when_no_summary(client, env, tmp_path):
    """Episodes made before summaries existed still get a blurb."""
    import db
    from pipeline import ingest

    pdf = tmp_path / "a.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])
    db.update_episode(
        episode_id, title="Old Episode", status="done",
        abstract="We study minimum wages. We use a border design. A third sentence "
                 "that should not appear in the listing at all.",
    )

    _publish(episode_id)
    html = client.get("/").text
    assert "We study minimum wages. We use a border design." in html
    assert "A third sentence" not in html


def test_blurb_truncation():
    import app as app_mod

    long_abstract = "word " * 200
    row = {"summary": None, "abstract": long_abstract}
    out = app_mod._blurb(row)
    assert len(out) <= 261 and out.endswith("…")

    assert app_mod._blurb({"summary": "  A teaser.  ", "abstract": "ignored"}) == "A teaser."
    assert app_mod._blurb({"summary": None, "abstract": None}) == ""


def test_no_play_button_before_audio_exists(client, env, tmp_path):
    import db
    from pipeline import ingest

    pdf = tmp_path / "a.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])
    db.update_episode(episode_id, title="Still Cooking", status="scripting")

    html = client.get("/admin").text
    assert "Still Cooking" in html
    assert 'class="play"' not in html
    assert "scripting" in html, "in-flight status still visible while browsing"


def test_episode_title_and_attribution(client, env, tmp_path):
    import db
    from pipeline import ingest

    pdf = tmp_path / "a.pdf"
    _make_pdf(pdf)
    eid = ingest.ingest_pdf(pdf, env["cfg"])
    db.update_episode(
        eid, status="done", summary="A blurb.",
        title="AMBIGUOUS ATTRIBUTION: THEORY AND EVIDENCE",
        authors=json.dumps(["Ricardo Alonso", "Monica Martinez-Bravo",
                            "Gerard Padró I Miquel", "Carlos Sanz"]),
        episode_title="Who gets blamed when nobody knows who decided",
    )

    _publish(eid)
    for url in ("/", f"/episode/{eid}"):
        html = client.get(url).text
        assert "Who gets blamed when nobody knows who decided" in html
        assert "This is an AI generated podcast drawing from" in html
        assert "Ambiguous Attribution: Theory and Evidence" in html, "all-caps title normalized"
        assert "Ricardo Alonso et al." in html, "long author list collapsed"
        assert "AMBIGUOUS ATTRIBUTION" not in html


def test_falls_back_to_paper_title_before_scripting(client, env, tmp_path):
    import db
    from pipeline import ingest

    pdf = tmp_path / "a.pdf"
    _make_pdf(pdf)
    eid = ingest.ingest_pdf(pdf, env["cfg"])
    db.update_episode(eid, title="A Perfectly Fine Paper Title", status="scripting",
                      authors=json.dumps(["Jane Roe"]))

    html = client.get("/admin").text
    assert "A Perfectly Fine Paper Title" in html
    assert "by Jane Roe" in html


def test_attribution_helpers():
    import app as app_mod

    assert app_mod._decaps("ALL CAPS TITLE OF THE PAPER") == "All Caps Title of the Paper"
    assert app_mod._decaps("A Mixed Case RCT Title") == "A Mixed Case RCT Title"
    # Acronyms must not be mangled into "Nber".
    assert app_mod._decaps("NBER WORKING PAPER SERIES") == "NBER Working Paper Series"
    assert app_mod._decaps("THE EFFECT OF GDP ON US WAGES") == "The Effect of GDP on US Wages"
    assert app_mod._author_credit(["Solo Author"]) == "Solo Author"
    assert app_mod._author_credit(["A B", "C D"]) == "A B and C D"
    assert app_mod._author_credit(["A", "B", "C", "D"]) == "A et al."
    assert "uncredited" in app_mod._attribution("T", [])
    # "et al." already carries a full stop; do not end up with "et al..".
    assert app_mod._attribution("T", ["A", "B", "C", "D"]).endswith("by A et al.")
    assert app_mod._attribution("T", ["Solo Author"]).endswith("by Solo Author.")


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_feed_carries_episode_title_and_disclosure(client, env, tmp_path):
    import xml.etree.ElementTree as ET

    import db

    eid = _done_episode(env, tmp_path)
    db.update_episode(eid, episode_title="The Blame Nobody Claims")
    _publish(eid)

    root = ET.fromstring(client.get("/feed.xml").content)
    item = root.find("channel/item")
    assert item.find("title").text == "The Blame Nobody Claims"
    assert "AI generated podcast" in item.find("description").text


def test_failures_are_collapsed_out_of_the_reading_list(client, env, tmp_path):
    import db
    from pipeline import ingest

    for name, title, status, err in (
        ("good", "A Finished Episode", "done", None),
        ("bad", "A Broken Episode", "failed", "paper is 370 pages; limit is 120."),
    ):
        pdf = tmp_path / f"{name}.pdf"
        _make_pdf(pdf)
        eid = ingest.ingest_pdf(pdf, env["cfg"])
        db.update_episode(eid, title=title, status=status, error=err,
                          summary=f"Summary of {title}.")

    html = client.get("/admin").text
    assert "A Finished Episode" in html
    assert "Summary of A Finished Episode." in html
    # The failure is reachable but out of the main list, with no summary line.
    assert "A Broken Episode" in html
    assert "Summary of A Broken Episode." not in html
    assert "1 failed" in html


def test_long_error_is_truncated_in_the_library(client, env, tmp_path):
    """A full ffmpeg stderr dump must not land in the listing."""
    import db
    from pipeline import ingest

    pdf = tmp_path / "a.pdf"
    _make_pdf(pdf)
    eid = ingest.ingest_pdf(pdf, env["cfg"])
    dump = "loudnorm measurement pass produced no JSON: " + ("size=N/A time=00:12:34 " * 80)
    db.update_episode(eid, title="Noisy Failure", status="failed", error=dump)

    html = client.get("/admin").text
    assert "loudnorm measurement pass produced no JSON" in html
    assert dump not in html, "the whole dump must not reach the library"
    # ...but the episode page keeps it in full.
    assert dump.strip() in client.get(f"/episode/{eid}").text


def test_short_error_helper():
    import app as app_mod

    assert app_mod._short_error(None) == ""
    assert app_mod._short_error("Boom. Details follow here.") == "Boom"
    long = "x" * 400
    assert len(app_mod._short_error(long)) <= 151


def test_episode_page_shows_script_and_flags(client, env, tmp_path):
    import db
    from pipeline import ingest

    pdf = tmp_path / "a.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])
    db.update_episode(
        episode_id, title="Flagged Paper", status="done",
        script_md="HOST_A: As Card and Krueger (1994) showed, wages rose.\nHOST_B: Fair.",
    )

    html = client.get(f"/episode/{episode_id}").text
    assert "Flagged Paper" in html
    assert "Card and Krueger (1994)" in html, "acceptance criterion 3: flags shown inline"
    assert "wages rose" in html


def test_episode_404(client):
    assert client.get("/episode/NOPE").status_code == 404


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_audio_supports_range_requests(client, env, tmp_path):
    """Seeking in a browser or podcast app depends on this."""
    episode_id = _done_episode(env, tmp_path)

    full = client.get(f"/episode/{episode_id}/audio")
    assert full.status_code == 200
    assert full.headers["accept-ranges"] == "bytes"
    size = len(full.content)

    part = client.get(f"/episode/{episode_id}/audio", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 0-99/{size}"
    assert len(part.content) == 100
    assert part.content == full.content[:100]

    tail = client.get(f"/episode/{episode_id}/audio", headers={"Range": "bytes=10-"})
    assert tail.status_code == 206
    assert tail.content == full.content[10:]

    bad = client.get(f"/episode/{episode_id}/audio", headers={"Range": f"bytes={size + 5}-"})
    assert bad.status_code == 416


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_feed_is_valid_rss_with_itunes_tags(client, env, tmp_path):
    """Acceptance criterion 5: the feed must parse and carry real enclosures."""
    import xml.etree.ElementTree as ET

    episode_id = _done_episode(env, tmp_path)
    _publish(episode_id)
    resp = client.get("/feed.xml")
    assert resp.status_code == 200
    assert "rss" in resp.headers["content-type"]

    root = ET.fromstring(resp.content)
    itunes = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
    channel = root.find("channel")
    assert channel.find("title").text == "Paperpod"
    assert channel.find("{http://www.itunes.com/dtds/podcast-1.0.dtd}image") is not None
    assert channel.find("itunes:author", itunes) is not None

    items = channel.findall("item")
    assert len(items) == 1
    item = items[0]
    assert item.find("itunes:duration", itunes).text
    enclosure = item.find("enclosure")
    assert enclosure.get("type") == "audio/mpeg"
    assert enclosure.get("url").startswith("http://paperpod.test:8000/")
    assert enclosure.get("url").endswith(f"/episode/{episode_id}/audio")

    mp3 = env["final"] / f"{episode_id}.mp3"
    assert int(enclosure.get("length")) == mp3.stat().st_size, "enclosure length must be exact"


def test_feed_excludes_unfinished_episodes(client, env, tmp_path):
    import xml.etree.ElementTree as ET

    import db
    from pipeline import ingest

    pdf = tmp_path / "a.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])
    db.update_episode(episode_id, title="Still Cooking", status="synthesizing")

    root = ET.fromstring(client.get("/feed.xml").content)
    assert root.find("channel").findall("item") == []


def test_feed_escapes_xml_in_titles(client, env, tmp_path):
    import xml.etree.ElementTree as ET

    import db
    from pipeline import ingest, run

    pdf = tmp_path / "a.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])
    db.update_episode(episode_id, status="done", audio_path=str(pdf), published=1,
                      title="Wages & Jobs: <Reconsidered>", duration_s=610.0)

    root = ET.fromstring(client.get("/feed.xml").content)  # would raise if unescaped
    assert root.find("channel/item/title").text == "Wages & Jobs: <Reconsidered>"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_delete_removes_row_and_files(client, env, tmp_path):
    import db

    episode_id = _done_episode(env, tmp_path)
    assert (env["final"] / f"{episode_id}.mp3").exists()

    assert client.delete(f"/episode/{episode_id}").status_code == 200
    assert db.get_episode(episode_id) is None
    assert not (env["final"] / f"{episode_id}.mp3").exists()
    assert not (env["papers"] / f"{episode_id}.pdf").exists()
    assert not (env["chunks"] / episode_id).exists()


def test_retry_rejects_unknown_stage(client, env, tmp_path):
    from pipeline import ingest

    pdf = tmp_path / "a.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])
    assert client.post(f"/episode/{episode_id}/retry", data={"stage": "bogus"}).status_code == 400


def test_retry_enqueues_named_stage(client, env, tmp_path):
    import db
    from pipeline import ingest

    pdf = tmp_path / "a.pdf"
    _make_pdf(pdf)
    episode_id = ingest.ingest_pdf(pdf, env["cfg"])
    db.update_episode(episode_id, status="failed", error="boom")

    import app as app_mod
    resp = client.post(f"/episode/{episode_id}/retry", data={"stage": "synthesizing"},
                       follow_redirects=False)
    assert resp.status_code == 303
    assert app_mod.WORK_Q.get_nowait() == (episode_id, "synthesizing")
    assert db.get_episode(episode_id)["status"] == "queued"


def test_upload_rejects_non_pdf(client):
    resp = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


# ------------------------------------------------------- public/admin split

@pytest.fixture
def public_client(env, tmp_path, monkeypatch):
    """A client with a password configured and no session: i.e. the internet."""
    from fastapi.testclient import TestClient

    import app as app_mod

    monkeypatch.setenv("PAPERPOD_ADMIN_PASSWORD", "hunter2")
    monkeypatch.setattr(app_mod, "CFG", env["cfg"])
    monkeypatch.setattr(app_mod, "FINAL_DIR", env["final"])
    monkeypatch.setattr(app_mod, "PAPERS_DIR", env["papers"])
    monkeypatch.setattr(app_mod, "CHUNKS_DIR", env["chunks"])
    monkeypatch.setattr(app_mod, "_worker", lambda: None)
    monkeypatch.setattr(app_mod, "_watch_inbox", lambda: None)
    with TestClient(app_mod.app) as c:
        yield c


def _episode(env, tmp_path, name="a", **fields):
    import db
    from pipeline import ingest

    pdf = tmp_path / f"{name}.pdf"
    _make_pdf(pdf)
    eid = ingest.ingest_pdf(pdf, env["cfg"])
    db.update_episode(eid, **fields)
    return eid


def test_unpublished_episodes_are_invisible_to_the_public(public_client, env, tmp_path):
    eid = _episode(env, tmp_path, title="Secret Draft", status="done",
                   audio_path=str(tmp_path / "a.pdf"), published=0)

    assert "Secret Draft" not in public_client.get("/").text
    assert public_client.get(f"/episode/{eid}").status_code == 404
    assert public_client.get(f"/episode/{eid}/audio").status_code == 404
    assert eid not in public_client.get("/feed.xml").text


def test_mutating_routes_are_closed_to_the_public(public_client, env, tmp_path):
    """These cost money or destroy data; an open internet must not reach them."""
    eid = _episode(env, tmp_path, title="Live One", status="done", published=1)

    assert public_client.post(f"/episode/{eid}/retry",
                              data={"stage": "scripting"}).status_code == 401
    assert public_client.delete(f"/episode/{eid}").status_code == 401
    assert public_client.post(f"/episode/{eid}/publish",
                              data={"published": "0"}).status_code == 401
    assert public_client.post(
        "/upload", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}
    ).status_code == 401
    assert public_client.get("/health").status_code == 401

    import db
    assert db.get_episode(eid)["published"] == 1, "nothing was mutated"


def test_admin_area_requires_login(public_client):
    resp = public_client.get("/admin", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/admin/login"


def test_login_grants_admin_then_logout_revokes_it(public_client, env, tmp_path):
    _episode(env, tmp_path, title="Secret Draft", status="done", published=0)

    assert public_client.post("/admin/login", data={"password": "wrong"},
                              follow_redirects=False).headers["location"].endswith("error=1")

    resp = public_client.post("/admin/login", data={"password": "hunter2"},
                              follow_redirects=False)
    assert resp.status_code == 303
    assert "Secret Draft" in public_client.get("/admin").text

    public_client.post("/admin/logout", follow_redirects=False)
    assert public_client.get("/admin", follow_redirects=False).status_code == 303


def test_forged_session_cookie_is_rejected(public_client, env, tmp_path):
    import time as _t

    _episode(env, tmp_path, title="Secret Draft", status="done", published=0)
    public_client.cookies.set("paperpod_admin", f"{int(_t.time()) + 9999}.deadbeef")
    assert public_client.get("/admin", follow_redirects=False).status_code == 303


def test_expired_session_is_rejected():
    import auth

    assert not auth.token_is_valid("1.abc")
    assert not auth.token_is_valid("")
    assert not auth.token_is_valid("garbage")


def test_admin_open_locally_when_no_password_is_set(client, env, tmp_path):
    """Convenience for local use -- and it fails closed, because a deploy that
    forgets the password locks admin out rather than exposing it."""
    import os

    import auth

    assert os.environ.get("PAPERPOD_ADMIN_PASSWORD") is None
    _episode(env, tmp_path, title="Local Draft", status="done", published=0)
    assert "Local Draft" in client.get("/admin").text
    assert auth.admin_password() is None


def test_publish_requires_reviewed_flags(public_client, env, tmp_path):
    import db

    public_client.post("/admin/login", data={"password": "hunter2"})
    eid = _episode(
        env, tmp_path, title="Flagged", status="done",
        audio_path=str(tmp_path / "a.pdf"),
        script_md="HOST_A: As Fabricated Author (1999) showed, things happened.",
    )

    blocked = public_client.post(f"/episode/{eid}/publish", data={"published": "1"})
    assert blocked.status_code == 400
    assert "flags not reviewed" in blocked.text
    assert db.get_episode(eid)["published"] == 0

    ok = public_client.post(f"/episode/{eid}/publish",
                            data={"published": "1", "reviewed": "1"})
    assert ok.status_code == 200
    assert db.get_episode(eid)["published"] == 1


def test_cannot_publish_an_unfinished_episode(public_client, env, tmp_path):
    public_client.post("/admin/login", data={"password": "hunter2"})
    eid = _episode(env, tmp_path, title="Cooking", status="scripting")

    resp = public_client.post(f"/episode/{eid}/publish", data={"published": "1"})
    assert resp.status_code == 400
    assert "not done" in resp.text


def test_admin_can_edit_title_and_summary(public_client, env, tmp_path):
    import db

    public_client.post("/admin/login", data={"password": "hunter2"})
    eid = _episode(env, tmp_path, title="The Original Paper Title", status="done",
                   published=1, episode_title="Machine title", summary="Machine summary.")

    resp = public_client.post(f"/episode/{eid}/edit", data={
        "episode_title": "  A better hand-written title  ",
        "summary": "A sharper summary,\n  written by a person.",
    }, follow_redirects=False)
    assert resp.status_code == 303

    row = db.get_episode(eid)
    assert row["episode_title"] == "A better hand-written title", "whitespace collapsed"
    assert row["summary"] == "A sharper summary, written by a person."

    public = public_client.get("/").text
    assert "A better hand-written title" in public
    assert "A sharper summary, written by a person." in public
    assert "Machine title" not in public


def test_admin_can_correct_the_attribution(public_client, env, tmp_path):
    """Paper title and authors are model-extracted and public, so an error
    here misnames a real person."""
    import db

    public_client.post("/admin/login", data={"password": "hunter2"})
    eid = _episode(env, tmp_path, title="MISREAD TITLE", status="done", published=1,
                   authors=json.dumps(["Wrong Name"]), year=1999,
                   episode_title="Ep", summary="S.")

    public_client.post(f"/episode/{eid}/edit", data={
        "episode_title": "Ep", "summary": "S.",
        "paper_title": "Ambiguous Attribution: Theory and Evidence",
        "authors": "Ricardo Alonso,  Monica Martinez-Bravo , Carlos Sanz",
        "year": "2026",
    })

    row = db.get_episode(eid)
    assert row["title"] == "Ambiguous Attribution: Theory and Evidence"
    assert db.episode_authors(row) == [
        "Ricardo Alonso", "Monica Martinez-Bravo", "Carlos Sanz"]
    assert row["year"] == 2026

    html = public_client.get("/").text
    assert "Ambiguous Attribution: Theory and Evidence" in html
    assert "Ricardo Alonso, Monica Martinez-Bravo and Carlos Sanz" in html
    assert "Wrong Name" not in html and "MISREAD TITLE" not in html


def test_partial_edit_leaves_omitted_fields_alone(public_client, env, tmp_path):
    """Submitting a field empty clears it; not sending it at all must not."""
    import db

    public_client.post("/admin/login", data={"password": "hunter2"})
    eid = _episode(env, tmp_path, title="Keep This Title", status="done",
                   authors=json.dumps(["Keep Me"]), year=2020,
                   episode_title="Old title", summary="Old summary.")

    public_client.post(f"/episode/{eid}/edit", data={"episode_title": "New title"})

    row = db.get_episode(eid)
    assert row["episode_title"] == "New title"
    assert row["summary"] == "Old summary.", "untouched field survives"
    assert row["title"] == "Keep This Title"
    assert db.episode_authors(row) == ["Keep Me"]
    assert row["year"] == 2020


def test_bad_year_does_not_break_the_edit(public_client, env, tmp_path):
    import db

    public_client.post("/admin/login", data={"password": "hunter2"})
    eid = _episode(env, tmp_path, title="T", status="done", year=2020)

    public_client.post(f"/episode/{eid}/edit",
                       data={"paper_title": "T", "authors": "A B", "year": "not a year"})
    assert db.get_episode(eid)["year"] is None


def test_clearing_an_edited_field_restores_the_fallback(public_client, env, tmp_path):
    import db

    public_client.post("/admin/login", data={"password": "hunter2"})
    eid = _episode(env, tmp_path, title="The Original Paper Title", status="done",
                   published=1, episode_title="Machine title",
                   abstract="First sentence of the abstract. Second one here. Third.")

    public_client.post(f"/episode/{eid}/edit", data={"episode_title": "", "summary": ""})

    row = db.get_episode(eid)
    assert row["episode_title"] is None and row["summary"] is None
    html = public_client.get("/").text
    assert "The Original Paper Title" in html, "falls back to the paper title"
    assert "First sentence of the abstract." in html, "falls back to the abstract"


def test_edit_is_admin_only(public_client, env, tmp_path):
    import db

    eid = _episode(env, tmp_path, title="Live One", status="done", published=1,
                   episode_title="Untouched")
    assert public_client.post(f"/episode/{eid}/edit",
                              data={"episode_title": "Hacked"}).status_code == 401
    assert db.get_episode(eid)["episode_title"] == "Untouched"


def test_edit_form_prefills_the_stored_title_not_the_fallback(public_client, env, tmp_path):
    """An empty episode title must not prefill with the paper's title, or
    saving would silently promote it."""
    public_client.post("/admin/login", data={"password": "hunter2"})
    eid = _episode(env, tmp_path, title="The Original Paper Title", status="done")

    html = " ".join(public_client.get(f"/episode/{eid}").text.split())
    assert 'name="episode_title" maxlength="120" value=""' in html, "no stored title"
    assert 'placeholder="The Original Paper Title"' in html, "fallback shown as a hint"


def test_terms_page_is_public_and_carries_the_disclosures(public_client, env, monkeypatch):
    import app as app_mod

    monkeypatch.setitem(app_mod.CFG, "site",
                        {"owner_name": "Josh Paul", "contact_email": "jbpaul275@gmail.com"})
    resp = public_client.get("/terms")
    assert resp.status_code == 200
    # Prose wraps across source lines; compare on collapsed whitespace.
    html = " ".join(resp.text.split())

    # (a) AI-generated, as-is, not fact checked, may contain errors
    assert "generated by artificial intelligence" in html
    assert "as-is basis" in html
    assert "not been independently fact-checked" in html
    # (b) takedown contact
    assert "jbpaul275@gmail.com" in html
    assert "mailto:jbpaul275@gmail.com" in html
    assert "Josh Paul" in html
    # (c) the rest
    for clause in ("no liability", "endorsed", "as is and as available",
                   "legal, medical, or investment advice", "no user accounts",
                   "rights in the original works remain"):
        assert clause.lower() in html.lower(), f"missing {clause!r}"


def test_footer_links_to_terms_everywhere(public_client, env, tmp_path):
    eid = _episode(env, tmp_path, title="Live One", status="done", published=1,
                   audio_path=str(tmp_path / "a.pdf"))
    for url in ("/", f"/episode/{eid}", "/terms", "/admin/login"):
        html = public_client.get(url).text
        assert 'href="/terms"' in html, f"no terms link on {url}"
        assert "AI-generated and may contain errors" in html


def test_feed_declares_an_owner_contact(public_client, env, tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET

    import app as app_mod

    monkeypatch.setitem(app_mod.CFG, "site",
                        {"owner_name": "Josh Paul", "contact_email": "jbpaul275@gmail.com"})
    root = ET.fromstring(public_client.get("/feed.xml").content)
    itunes = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
    owner = root.find("channel/itunes:owner", itunes)
    assert owner is not None
    assert owner.find("itunes:email", itunes).text == "jbpaul275@gmail.com"


def test_empty_state_does_not_tell_the_public_to_upload(public_client, env, tmp_path):
    """The public has no upload panel, so "drop a paper in below" points at
    nothing that exists for them."""
    _episode(env, tmp_path, title="Private Draft", status="done", published=0)

    public = public_client.get("/").text
    assert "No episodes published yet." in public
    for upload_cue in ("Drop a paper", "Drop a PDF", "choose a file", "Queue"):
        assert upload_cue not in public, f"{upload_cue!r} points at nothing public"

    public_client.post("/admin/login", data={"password": "hunter2"})
    admin_html = public_client.get("/admin").text
    assert "Drop a PDF" in admin_html and "Queue" in admin_html


def test_admin_library_marks_public_and_private(public_client, env, tmp_path):
    public_client.post("/admin/login", data={"password": "hunter2"})
    _episode(env, tmp_path, name="live", title="Live One", status="done", published=1)
    _episode(env, tmp_path, name="draft", title="Draft One", status="done", published=0)

    html = public_client.get("/admin").text
    live = html.index("Live One")
    draft = html.index("Draft One")
    lo, hi = (min(live, draft), max(live, draft))
    assert "public" in html[lo:hi] or "public" in html[hi:]
    assert html.count("private") >= 1 and html.count("public") >= 1


def test_public_page_hides_operator_detail(public_client, env, tmp_path):
    eid = _episode(
        env, tmp_path, title="Live One", status="done", published=1,
        audio_path=str(tmp_path / "a.pdf"), cost_usd=0.64, flags_reviewed=1,
        script_md="HOST_A: As Fabricated Author (1999) showed, things happened.",
    )
    html = public_client.get(f"/episode/{eid}").text

    assert "Live One" in html
    for leak in ("API cost", "0.64", "Stage log", "Delete episode",
                 "not found in the paper", "Visibility"):
        assert leak not in html, f"{leak!r} must not reach the public page"


def test_public_page_is_audio_only(public_client, env, tmp_path):
    """The public gets the title, the disclosure, a summary and the player --
    not the transcript, the paper's abstract, or citation bookkeeping."""
    import db

    eid = _episode(
        env, tmp_path, title="Live One", status="done", published=1,
        episode_title="A Listenable Title", summary="One sentence of teaser.",
        audio_path=str(tmp_path / "a.pdf"), flags_reviewed=1,
        abstract="We study the effect of minimum wage increases on employment.",
        script_md="HOST_A: A spoken line that must not appear as text.\n"
                  "HOST_B: Card and Krueger (1994) is in the paper.",
    )
    html = public_client.get(f"/episode/{eid}").text

    # Present: what a listener needs.
    assert "A Listenable Title" in html
    assert "One sentence of teaser." in html
    assert "AI generated podcast" in html
    assert f"/episode/{eid}/audio" in html

    # Absent: everything else.
    assert "A spoken line that must not appear as text" not in html, "no transcript"
    assert "Abstract" not in html and "minimum wage increases" not in html
    assert "traced back to the paper" not in html
    assert "Card and Krueger" not in html


def test_admin_still_sees_script_and_abstract(public_client, env, tmp_path):
    public_client.post("/admin/login", data={"password": "hunter2"})
    eid = _episode(
        env, tmp_path, title="Live One", status="done", published=1,
        abstract="We study the effect of minimum wage increases.",
        script_md="HOST_A: A spoken line that must not appear as text.",
    )
    html = public_client.get(f"/episode/{eid}").text

    assert "A spoken line that must not appear as text" in html, "review needs the script"
    assert "minimum wage increases" in html


# ------------------------------------------------------- upload feedback

def test_duplicate_upload_points_at_the_existing_episode(public_client, env, tmp_path):
    """A duplicate used to redirect to the public library, which shows only
    published episodes — so it looked exactly like a silent failure."""
    import db

    public_client.post("/admin/login", data={"password": "hunter2"})
    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    first = ingest_module().ingest_pdf(pdf, env["cfg"])

    resp = public_client.post(
        "/upload", files={"file": ("paper.pdf", pdf.read_bytes(), "application/pdf")},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/episode/{first}?dup=1"
    assert len(db.list_episodes()) == 1


def test_rejected_upload_returns_a_readable_message(public_client, env, tmp_path):
    """A scanned or over-long PDF used to dump raw JSON at the browser."""
    public_client.post("/admin/login", data={"password": "hunter2"})
    scan = tmp_path / "scan.pdf"
    _make_pdf(scan, pages=1, text="Fig 1.")

    resp = public_client.post(
        "/upload", files={"file": ("scan.pdf", scan.read_bytes(), "application/pdf")},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/admin?error=")
    assert "scanned" in resp.headers["location"]

    assert "scanned" in public_client.get(resp.headers["location"]).text


def test_upload_succeeds_while_another_episode_is_processing(public_client, env, tmp_path):
    """The suspicion behind the sidebar: a second upload must still enqueue."""
    import db

    import app as app_mod

    public_client.post("/admin/login", data={"password": "hunter2"})
    busy = _episode(env, tmp_path, name="busy", title="Busy One", status="synthesizing")

    second = tmp_path / "second.pdf"
    _make_pdf(second, text="A different paper entirely. " * 40)
    resp = public_client.post(
        "/upload", files={"file": ("second.pdf", second.read_bytes(), "application/pdf")},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].endswith("?queued=1")

    new_id = resp.headers["location"].split("/")[2].split("?")[0]
    assert new_id != busy
    assert db.get_episode(new_id)["status"] == "queued"
    assert app_mod.WORK_Q.qsize() >= 1, "it really is on the queue"


def test_sidebar_shows_the_queue(public_client, env, tmp_path):
    public_client.post("/admin/login", data={"password": "hunter2"})
    _episode(env, tmp_path, name="a", title="Now Running", status="synthesizing")
    _episode(env, tmp_path, name="b", title="Waiting Behind", status="queued")
    _episode(env, tmp_path, name="c", title="All Finished", status="done", published=1)

    html = public_client.get("/admin").text
    queue_block = html.split('id="queue"')[1].split("</section>")[0]
    assert "Now Running" in queue_block
    assert "Waiting Behind" in queue_block
    assert "All Finished" not in queue_block
    assert 'data-count="2"' in html


def ingest_module():
    from pipeline import ingest
    return ingest


# ------------------------------------------------------------- source URL

def test_source_url_hyperlinks_the_paper_title(public_client, env, tmp_path):
    public_client.post("/admin/login", data={"password": "hunter2"})
    eid = _episode(env, tmp_path, title="Elite Capture and Social Stability",
                   status="done", published=1, authors=json.dumps(["Jingjing Chen"]),
                   audio_path=str(tmp_path / "a.pdf"))

    public_client.post(f"/episode/{eid}/edit",
                       data={"source_url": "https://www.nber.org/papers/w12345"})

    import db
    assert db.get_episode(eid)["source_url"] == "https://www.nber.org/papers/w12345"

    for url in ("/", f"/episode/{eid}"):
        html = public_client.get(url).text
        assert 'href="https://www.nber.org/papers/w12345"' in html, f"no link on {url}"
        assert 'rel="noopener nofollow"' in html


def test_no_link_without_a_source_url(public_client, env, tmp_path):
    _episode(env, tmp_path, title="Paywalled Paper", status="done", published=1,
             authors=json.dumps(["A Person"]), audio_path=str(tmp_path / "a.pdf"))
    html = public_client.get("/").text
    assert "Paywalled Paper" in html
    assert "<a href=\"http" not in html.split('class="attribution"')[1].split("</p>")[0]


def test_only_http_urls_are_accepted():
    """This value is rendered into an href on a public page."""
    import app as app_mod

    assert app_mod.safe_url("https://example.org/p.pdf") == "https://example.org/p.pdf"
    assert app_mod.safe_url("http://example.org") == "http://example.org"
    for bad in ("javascript:alert(1)", "data:text/html;base64,x", "  ", None,
                "ftp://example.org", "example.org", "//evil.example"):
        assert app_mod.safe_url(bad) is None, f"{bad!r} must not become an href"


def test_dangerous_url_is_rejected_on_save(public_client, env, tmp_path):
    import db

    public_client.post("/admin/login", data={"password": "hunter2"})
    eid = _episode(env, tmp_path, title="T", status="done")
    public_client.post(f"/episode/{eid}/edit",
                       data={"source_url": "javascript:alert(document.cookie)"})
    assert db.get_episode(eid)["source_url"] is None


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_feed_includes_the_source_paper_link(public_client, env, tmp_path):
    import xml.etree.ElementTree as ET

    import db

    eid = _done_episode(env, tmp_path)
    db.update_episode(eid, published=1, source_url="https://example.org/paper.pdf")

    root = ET.fromstring(public_client.get("/feed.xml").content)
    assert "https://example.org/paper.pdf" in root.find("channel/item/description").text


# --------------------------------------------------- per-episode TTS model

def test_upload_pins_the_chosen_tts_model(public_client, env, tmp_path, monkeypatch):
    import db

    import app as app_mod
    monkeypatch.setattr(app_mod, "tts_choices",
                        lambda: ["model-a", "model-b"])
    public_client.post("/admin/login", data={"password": "hunter2"})

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    resp = public_client.post(
        "/upload",
        files={"file": ("paper.pdf", pdf.read_bytes(), "application/pdf")},
        data={"tts_model": "model-b"}, follow_redirects=False,
    )
    eid = resp.headers["location"].split("/")[2].split("?")[0]
    assert db.get_episode(eid)["tts_model"] == "model-b"


def test_upload_rejects_an_unlisted_model(public_client, env, tmp_path, monkeypatch):
    import app as app_mod
    monkeypatch.setattr(app_mod, "tts_choices", lambda: ["model-a"])
    public_client.post("/admin/login", data={"password": "hunter2"})

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    resp = public_client.post(
        "/upload", files={"file": ("paper.pdf", pdf.read_bytes(), "application/pdf")},
        data={"tts_model": "gemini-does-not-exist"},
    )
    assert resp.status_code == 400


def test_synthesis_uses_the_episodes_model_not_the_config(env, tmp_path):
    """A config change mid-library must not switch voice model partway through
    an episode already part-synthesized."""
    import db
    from pipeline import ingest, tts

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    eid = ingest.ingest_pdf(pdf, env["cfg"])

    assert tts.model_for(eid, env["cfg"]) == "t", "falls back to config"
    db.update_episode(eid, tts_model="pinned-model")
    assert tts.model_for(eid, env["cfg"]) == "pinned-model"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_clone_reuses_the_script_with_a_different_model(public_client, env, tmp_path, monkeypatch):
    """Comparing voice models needs identical words in both episodes."""
    import db
    from pipeline import ingest, run

    import app as app_mod
    monkeypatch.setattr(app_mod, "tts_choices", lambda: ["model-a", "model-b"])
    monkeypatch.setattr(app_mod, "PAPERS_DIR", env["papers"])
    public_client.post("/admin/login", data={"password": "hunter2"})

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    src = ingest.ingest_pdf(pdf, env["cfg"])
    run.run_episode(src, env["cfg"])
    db.update_episode(src, source_url="https://example.org/p.pdf")

    resp = public_client.post(f"/episode/{src}/clone", data={"tts_model": "model-b"},
                              follow_redirects=False)
    assert resp.status_code == 303
    new_id = resp.headers["location"].split("/")[2].split("?")[0]
    assert new_id != src

    original, clone = db.get_episode(src), db.get_episode(new_id)
    assert clone["script_md"] == original["script_md"], "same words, or it is not a comparison"
    assert clone["title"] == original["title"]
    assert clone["source_url"] == original["source_url"]
    assert clone["tts_model"] == "model-b"
    assert original["tts_model"] is None, "the source is left alone"
    assert (env["papers"] / f"{new_id}.pdf").exists(), "PDF copied for flag checking"
    assert app_mod.WORK_Q.get_nowait() == (new_id, "synthesizing"), "skips scripting"


def test_clone_refuses_without_a_script(public_client, env, tmp_path, monkeypatch):
    import app as app_mod
    monkeypatch.setattr(app_mod, "tts_choices", lambda: ["model-a", "model-b"])
    public_client.post("/admin/login", data={"password": "hunter2"})
    eid = _episode(env, tmp_path, title="No Script Yet", status="queued")

    resp = public_client.post(f"/episode/{eid}/clone", data={"tts_model": "model-b"})
    assert resp.status_code == 400
    assert "no script" in resp.text


def test_clone_is_admin_only(public_client, env, tmp_path):
    eid = _episode(env, tmp_path, title="T", status="done", script_md="HOST_A: Hi.")
    assert public_client.post(f"/episode/{eid}/clone",
                              data={"tts_model": "model-b"}).status_code == 401


def test_tts_choices_always_include_the_configured_default(monkeypatch):
    import app as app_mod

    monkeypatch.setitem(app_mod.CFG, "tts", {"models": ["extra-model"]})
    monkeypatch.setitem(app_mod.CFG, "models", {**app_mod.CFG["models"], "tts": "default-model"})
    assert app_mod.tts_choices() == ["extra-model", "default-model"]

    monkeypatch.setitem(app_mod.CFG, "tts", {})
    assert app_mod.tts_choices() == ["default-model"], "never empty"


def test_models_page_lists_what_the_key_offers(public_client, monkeypatch):
    import pipeline.gemini as g

    class FakeModel:
        def __init__(self, n): self.name, self.display_name, self.supported_actions = n, "", ["generateContent"]

    monkeypatch.setattr(g, "client", lambda: type("C", (), {
        "models": type("M", (), {"list": staticmethod(
            lambda: [FakeModel("models/gemini-3.1-flash-tts-preview"),
                     FakeModel("models/gemini-3-flash-preview")])})()})())
    public_client.post("/admin/login", data={"password": "hunter2"})

    html = public_client.get("/admin/models").text
    assert "gemini-3.1-flash-tts-preview" in html
    assert "gemini-3-flash-preview" in html


def test_models_page_is_admin_only(public_client):
    resp = public_client.get("/admin/models", follow_redirects=False)
    assert resp.status_code == 303


# ------------------------------------------- multiple renderings of a paper

@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_publishing_one_rendering_demotes_the_others(public_client, env, tmp_path, monkeypatch):
    """Exactly one rendering of a paper may be public, or the feed would carry
    the same episode twice in different voices."""
    import db
    from pipeline import ingest, run

    import app as app_mod
    monkeypatch.setattr(app_mod, "tts_choices", lambda: ["model-a", "model-b"])
    monkeypatch.setattr(app_mod, "PAPERS_DIR", env["papers"])
    public_client.post("/admin/login", data={"password": "hunter2"})

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    first = ingest.ingest_pdf(pdf, env["cfg"])
    run.run_episode(first, env["cfg"])
    db.update_episode(first, flags_reviewed=1)
    public_client.post(f"/episode/{first}/publish", data={"published": "1"})
    assert db.get_episode(first)["published"] == 1

    resp = public_client.post(f"/episode/{first}/clone", data={"tts_model": "model-b"},
                              follow_redirects=False)
    second = resp.headers["location"].split("/")[2].split("?")[0]
    run.run_episode(second, env["cfg"], from_stage="synthesizing")
    db.update_episode(second, flags_reviewed=1)

    public_client.post(f"/episode/{second}/publish", data={"published": "1"})
    assert db.get_episode(second)["published"] == 1
    assert db.get_episode(first)["published"] == 0, "the previous canonical steps down"

    # And the public library carries one, not both.
    assert len([r for r in db.list_episodes(published_only=True)]) == 1


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg required")
def test_episode_page_lists_every_rendering(public_client, env, tmp_path, monkeypatch):
    import db
    from pipeline import ingest, run

    import app as app_mod
    monkeypatch.setattr(app_mod, "tts_choices", lambda: ["model-a", "model-b"])
    monkeypatch.setattr(app_mod, "PAPERS_DIR", env["papers"])
    public_client.post("/admin/login", data={"password": "hunter2"})

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    first = ingest.ingest_pdf(pdf, env["cfg"])
    run.run_episode(first, env["cfg"])
    resp = public_client.post(f"/episode/{first}/clone", data={"tts_model": "model-b"},
                              follow_redirects=False)
    second = resp.headers["location"].split("/")[2].split("?")[0]
    run.run_episode(second, env["cfg"], from_stage="synthesizing")

    html = public_client.get(f"/episode/{first}").text
    assert "2 renderings of this paper" in html
    assert "model-b" in html, "the sibling's model is named"
    assert db.get_episode(second)["audio_built_at"], "build time recorded"
    assert db.get_episode(second)["audio_built_at"] in html


def test_single_rendering_shows_no_versions_table(public_client, env, tmp_path):
    public_client.post("/admin/login", data={"password": "hunter2"})
    eid = _episode(env, tmp_path, title="Only One", status="done")
    assert "renderings of this paper" not in public_client.get(f"/episode/{eid}").text


def test_versions_are_admin_only(public_client, env, tmp_path, monkeypatch):
    import db

    eid = _episode(env, tmp_path, title="Live", status="done", published=1,
                   audio_path=str(tmp_path / "a.pdf"))
    row = db.get_episode(eid)
    db.create_episode("SIB", "/tmp/s.pdf", row["sha256"], status="done")
    db.update_episode("SIB", title="Sibling", status="done", tts_model="model-b")

    html = public_client.get(f"/episode/{eid}").text
    assert "renderings of this paper" not in html
    assert "model-b" not in html


def test_total_chunk_failure_names_the_reason(env, tmp_path, monkeypatch):
    """"every TTS chunk failed" alone sends you to the logs to find out why."""
    import db
    from pipeline import PipelineError, ingest, tts

    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    eid = ingest.ingest_pdf(pdf, env["cfg"])
    db.update_episode(eid, script_md=SAMPLE_SCRIPT, tts_model="gemini-does-not-exist")

    def always_404(episode_id, entry, wav_path, cfg):
        raise RuntimeError(
            "404 NOT_FOUND. models/gemini-does-not-exist is not found for API version v1beta")

    monkeypatch.setattr(tts, "_synthesize_chunk", always_404)
    with pytest.raises(PipelineError) as exc:
        tts.synthesize(eid, env["cfg"])

    message = str(exc.value)
    assert "gemini-does-not-exist" in message, "names the model that failed"
    assert "404 NOT_FOUND" in message, "carries the underlying error"
