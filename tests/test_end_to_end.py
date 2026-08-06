"""End-to-end pipeline and web-app tests.

Only the three Gemini calls are stubbed. ffmpeg assembly, loudness
normalization, MP3 encoding, HTTP range serving, and RSS generation all run for
real, so these cover acceptance criteria 1, 2, 3, 5 and 6.
"""

import json
import math
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

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
    assert ingest.ingest_pdf(copy, env["cfg"]) is None, "same bytes = same episode"
    assert len(db.list_episodes()) == 1


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


def test_health_route(client):
    body = client.get("/health").json()
    assert "queue_depth" in body and "worker_alive" in body


def test_library_lists_episodes(client, env, tmp_path):
    import db
    from pipeline import ingest

    episode_id = ingest.ingest_pdf((lambda p: (_make_pdf(p), p)[1])(tmp_path / "a.pdf"), env["cfg"])
    db.update_episode(episode_id, title="A Paper About Wages",
                      authors=json.dumps(["Jane Roe"]), status="done")

    html = client.get("/").text
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

    html = client.get("/").text
    assert "Still Cooking" in html
    assert 'class="play"' not in html
    assert "scripting" in html, "in-flight status still visible while browsing"


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

    html = client.get("/").text
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

    html = client.get("/").text
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
    db.update_episode(episode_id, status="done", audio_path=str(pdf),
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
