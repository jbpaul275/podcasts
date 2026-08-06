# Paperpod

Drop a PDF of an academic paper into a folder or a browser drop zone. Ten minutes later there is a finished MP3 in a local library, playable in the browser and subscribable from a phone.

Local-only. No public hosting, no accounts, no publishing to Apple or Spotify, no scraping — you supply PDFs you already have.

## How it works

```
PDF → ingest → script → TTS → assemble → MP3
```

| Stage | What it does |
|---|---|
| `ingest` | SHA-256 dedupe, page-count and text-layer validation, copy into `data/papers/`, native-PDF metadata extraction |
| `script` | Sends the PDF natively (so tables and figures survive) and returns speaker-tagged dialogue |
| `tts` | Chunks the script on speaker-turn boundaries, synthesizes each chunk with two-speaker TTS |
| `assemble` | ffmpeg concat with seam silence, two-pass loudness normalization, 96k mono MP3 with ID3 tags |

Every stage writes its output to disk and its status to SQLite before the next one starts, so a crash is resumable from the last completed stage. TTS in particular resumes at the chunk level — killing the process mid-synthesis and restarting will not regenerate the script.

## Setup

Requires Python 3.11+ and the `ffmpeg` binary (`ffmpeg` and `ffprobe` on `PATH`).

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...        # the only place the key is read from
python app.py                    # serves on 0.0.0.0:8000
```

The API key is read from the environment only. It is never written to `config.toml` or the database.

## Using it

Open <http://localhost:8000>. Drop a PDF on the page, or copy one into `data/inbox/` — a watcher picks it up automatically (2-second debounce so partially-written files are ignored). Originals in the inbox are never mutated or deleted; they are moved to `data/inbox/processed/` once registered.

The episode page shows the player, the full script with per-speaker styling, the stage log, and the accumulated API cost. Failed runs can be re-run from any named stage.

### Citation flags

The script prompt forbids fabricated citations, but models fabricate anyway. After generation, the script is regex-scanned for citation-shaped strings — `Name (Year)`, `et al.`, and proper nouns sitting near a bare four-digit year — and every hit is surfaced on the episode page, both as a summary box and inline on the offending line.

These are **flags, not failures**. Most hits are innocent ("the Mariel boatlift in 1980"). The point is to put anything that *could* be an invented citation in front of a person before you publish it to your own ears. Check each against the paper.

## Listening on a phone

The feed at `/feed.xml` is a valid RSS 2.0 feed with the iTunes namespace and absolute enclosure URLs. There is no authentication — the tailnet is the security boundary, so do not expose this to the public internet.

1. Install [Tailscale](https://tailscale.com/) on both the machine running Paperpod and your phone, signed into the same tailnet.
2. Find the host's Tailscale name (`tailscale status`), e.g. `mac-mini.tail1234.ts.net`.
3. Set `base_url` in `config.toml` to that host so enclosure URLs resolve from the phone:
   ```toml
   [server]
   base_url = "http://mac-mini.tail1234.ts.net:8000"
   ```
4. Restart Paperpod. It binds `0.0.0.0`, so it is reachable from anywhere on the tailnet.
5. In Overcast or Pocket Casts, use **add URL by hand** (Overcast: `+` → *Add URL*; Pocket Casts: *Profile* → *Add Podcast* → *Add by URL*) and paste `http://mac-mini.tail1234.ts.net:8000/feed.xml`.

Only episodes with status `done` and an audio file on disk appear in the feed.

## Configuration

All knobs live in `config.toml`, loaded once at startup.

- `[models]` — model IDs for metadata extraction, scripting, and TTS.
- `[voices]` — the two prebuilt voice IDs. Two speakers is the documented maximum; do not add a third host.
- `[script]` — `target_words` (1600 ≈ ten minutes) and `max_pages` rejection threshold.
- `[audio]` — `seam_silence_ms` is the main quality tell. It ships at 250ms; try 150 and 400 and pick by ear.
- `[server]` — `base_url` (see Tailscale above) and `port`.
- `[costs]` — per-model token prices used to compute the per-episode cost shown in the UI.

Prompts live in `prompts/` as plain Markdown, loaded at call time rather than baked into Python. `script_system.md` is the main quality lever — edit it freely without touching code.

## Tests

```bash
python -m pytest tests/ -q
```

44 tests. Only the three Gemini calls are stubbed; ffmpeg assembly, loudness normalization, MP3 encoding, HTTP range serving, and RSS generation all run for real. Tests requiring ffmpeg skip cleanly if it is not installed.

## Model IDs and the upstream docs

`ai.google.dev` is unreachable from the sandbox this was built in (HTTP 403 through the egress proxy), so the two documentation pages could not be read directly. The API surface was instead verified against the installed `google-genai` SDK (2.17.0), whose type definitions are the binding contract, and the TTS model details cross-checked against published sources.

Confirmed in use:

- `gemini-3.1-flash-tts-preview` supports **at most two speakers** and returns **24 kHz 16-bit mono PCM** — which is why `tts.py` parses the sample rate out of the response MIME type and wraps the PCM in a WAV header rather than guessing.
- Every type used (`SpeechConfig`, `MultiSpeakerVoiceConfig`, `SpeakerVoiceConfig`, `VoiceConfig`, `PrebuiltVoiceConfig`) exists in the SDK with the field names used here.
- `gemini-3-flash-preview` works for metadata and scripting. `gemini-3-pro-preview` returned `limit: 0` on a paid key, so preview Pro appears to need separate access.

### Which TTS models work

Three, all callable through `generateContent` with an audio response:

| Model | Character | Output rate |
|---|---|---|
| `gemini-3.1-flash-tts-preview` | Expressive, audio tags | $20 / 1M tokens |
| `gemini-2.5-pro-preview-tts` | High fidelity, aimed at podcasts and audiobooks | $20 / 1M tokens |
| `gemini-2.5-flash-preview-tts` | Fastest, half the output rate | $10 / 1M tokens |

Input is $1 / 1M tokens for all three, and is a rounding error — audio output is
roughly 97% of an episode's cost. So 3.1 Flash and 2.5 Pro cost the same and the
choice between them is purely which sounds better; 2.5 Flash is about half price.

**The Live models are not usable here.** `gemini-3.1-flash-live-preview` and `gemini-2.5-flash-native-audio-preview-12-2025` are bidirectional streaming models for real-time dialogue, reached through the Live API rather than `generateContent`. Putting one in `[tts] models` would fail every chunk.

Visit `/admin/models` to see what your key can actually call — hardcoded IDs go stale, and a wrong one 404s an entire episode before anything surfaces.

**Every model in the picker needs a `[costs]` entry**, or its spend is reported as $0.00; the app logs a warning at startup for any that are missing.

The prices in `[costs]` are documented for TTS and **estimates for the two text models** — correct them against current Gemini pricing so the per-episode cost figure means something.

## Layout

```
config.toml          config, loaded once at startup
app.py               FastAPI routes, background worker, inbox watcher
db.py                schema + queries (sqlite3 stdlib, no ORM)
pipeline/
  ingest.py          PDF in, metadata + validation out
  script.py          paper in, dialogue script out, citation flagging
  tts.py             script in, audio chunks out
  assemble.py        chunks in, normalized MP3 out
  run.py             orchestrator, stage state machine
prompts/             editable without touching code
templates/ static/   Jinja2 + vanilla JS, no build step
tools/make_cover.py  regenerates static/cover.png
data/                inbox/, papers/, audio/chunks/, audio/final/, paperpod.db
```
