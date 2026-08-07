# Paperpod

Drop a PDF of an academic paper into a folder or a browser drop zone. Ten minutes later there is a finished MP3 in a local library, playable in the browser and subscribable from a phone.

Local-only. No public hosting, no accounts, no publishing to Apple or Spotify, no scraping — you supply PDFs you already have.

## How it works

```
PDF → ingest → script → TTS → assemble → MP3
```

| Stage | What it does |
|---|---|
| `ingest` | SHA-256 dedupe, size/page/text-layer validation, copy into `data/papers/`, native-PDF metadata extraction |
| `script` | Sends the PDF natively (so tables and figures survive) and returns speaker-tagged dialogue |
| `tts` | Records the spoken AI disclosure in its own voice, then chunks the script on speaker-turn boundaries and synthesizes each chunk with two-speaker TTS |
| `assemble` | ffmpeg concat with seam silence, disclosure first, two-pass loudness normalization, 96k mono MP3 with ID3 tags |

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

### Which PDFs are accepted

Rejections are checked before anything is spent, and every message says what to do about it.

The text-layer check samples the **first ten pages**, not the first one. Working-paper series open on a cover page carrying almost nothing — BLS, NBER and IZA all do — and looking only at page one calls those a scan. A genuine scan has no text on any page, so a handful is plenty to tell them apart.

No text anywhere reads as a scan and says to OCR it. A little text but not enough is a different problem and says so, because "run OCR" is the wrong advice for a document that is simply thin.

**A rejection is a verdict under the limits in force at the time**, and it is stored as flat text on the episode. Raise `max_pages` and every paper refused under the old ceiling would otherwise keep quoting that ceiling forever — re-uploading matches on SHA and never reaches the validator again, so the message reads as a live decision by code that no longer exists.

So re-uploading a paper that was turned away at ingest re-runs validation. If it now passes, the existing episode is accepted and queued rather than reported as a duplicate; if it still fails, its stored reason is rewritten to state today's limit. Only episodes that never entered the pipeline qualify — an empty stage log is exactly what an ingest-time rejection looks like, and quietly restarting a scripting or TTS failure would re-spend real money.

A re-accepted episode is **re-dated**. The row is reused, and `created_at` is both what the library sorts on and what the feed sends as `pubDate` — so without this a paper you just re-uploaded would appear wherever it sat when it first bounced, days down a newest-first list. Re-uploading is a new submission and needs to surface like one.

### Where failures go

A failed episode leaves the main list for a collapsed box at the bottom of the admin page. That is right for a failure from last week and wrong for one from ten minutes ago: the episode you were watching disappears from where you were watching it, with a shut disclosure element the only trace.

So `failed_at` is stamped whenever an episode fails — `db.mark_failed()` is the single path, rather than three call sites each remembering — and a failure inside the last six hours opens the box and says "one just now". Older ones settle back down.

### Is it stuck, or just slow?

A running episode shows what it is doing (`synthesizing chunk 3 of 12`) and how long it has been doing it, on both the episode page and the admin queue. TTS is the long stage: chunks land every minute or two, so the timestamp is the signal, not the stage name. Past 15 minutes with no movement the line turns red and reads *stalled* — retry the stage from the episode page, which keeps every chunk already on disk and re-synthesizes only what is missing.

### When chunks fail

A chunk that cannot be synthesized becomes a logged gap rather than sinking the episode — the rest assembles and the episode page says which chunks are missing and **why**. The reason matters: which chunks died is the symptom, and on its own it sends you to the process log to find the cause. Identical reasons are grouped, because eight copies of one sentence hide that it is one problem.

Nearly every failure here is a rate limit, and rate limits are a fact about the account rather than about one request. Three things follow from that:

- **A 429 holds back every other call.** When the server names a retry window, `gemini.THROTTLE` closes it for the whole process — both workers, every remaining chunk. Without this, each chunk discovered the closed window separately: one gives up, the next starts with a fresh retry budget and no idea the API just said slow down, and sprints back into the same wall. That is how one rate limit becomes a contiguous tail of failed chunks. The thread that *got* the 429 backs off on its own schedule and is exempt, so nothing waits twice for one window.
- **Failed chunks get a second pass.** After the first pass the stage waits `[tts] retry_pass_delay_s` (60s) and retries just the ones that failed. A chunk gives up within seconds of hitting a limit; a minute later the same call usually works. Set it to 0 to retry immediately, or negative to disable the pass.
- **A dropped connection is retried.** Transport failures arrive with no HTTP status, so a retry check that keys on the status code classified the *most* transient failure there is as permanent and burned the chunk on its first attempt. `is_transport_error` catches those by exception type and message — but only when no status is present, so a real 400 is never talked out of being a 400.

**A daily quota is not retried through.** The prose in a 429 is the same sentence every time — "You exceeded your current quota, please check your plan and billing details" — followed by two long documentation URLs, and none of it distinguishes a per-minute limit you should wait out from a per-day one that will not move until tomorrow. The structured `QuotaFailure` detail does, and Google spells the window into the quota id (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`), so it is readable rather than guessed at.

A per-day violation stops the stage immediately instead of backing off through it, because the reset is on Google's clock and half an hour of retries arrives at the same 429. Three facts the error message now carries, because each one changes what you should do:

- **RPD quotas reset at midnight Pacific.** There is no shorter wait to find.
- **Failed requests still count against the day's allowance.** Retrying into an exhausted quota spends more of it, which is why grinding through is worse than stopping.
- **Limits are per Cloud project, not per API key.** So a `-FreeTier` quota id means the project *this key belongs to* is unbilled — having budget on a different project does not raise it, and minting a new key in the same project changes nothing. That is a different problem from going too fast, and it needs saying out loud because the 429 text is identical either way.

Budget remaining and rate limit are separate systems: a paid project still caps preview models well below the headline numbers, so money in the account is not evidence that a quota is wrong.

Reasons stored on an episode are distilled rather than dumped — `gemini.describe()` keeps the quota name, limit and window and drops the boilerplate, because the boilerplate is what pushes the useful part past any sensible truncation.

Retrying the stage by hand is still there and still cheap: chunks already on disk are skipped, so you only pay for the holes.

### Rewriting a script

An episode with a script has a **Rewrite the script** panel. Two buttons:

- **Revise this script** takes plain-English notes ("cut the instrument, spend more time on section 5") and edits what is there, leaving passages the notes do not mention intact. The paper goes along with the request, so asking for more on a section it covers actually gets you more rather than padding.
- **Start over from the paper** ignores the current script and writes a fresh one. Your notes still apply as extra direction. This is also how you try a different model on the same paper — pick from the model dropdown, which is fed by `[script] models`.

Both run the scripting stage **and stop**. Audio is ~97% of an episode's cost, so re-synthesizing on every wording change would make iteration unaffordable; the existing audio is left alone and the episode goes to `needs_review`. When the script reads right, re-run from `synthesizing` — the retry picker already defaults there.

The previous script is kept for a one-step undo. An episode whose script is newer than its audio says so at the top of the page, because otherwise the player quietly serves words nobody approved.

### Titles and episode numbers

Episode titles are **Title Case**, and that is enforced in code rather than only asked for in the prompt. `prompts/episode_title.md` used to say "sentence case or title case", which produced exactly what you would expect — a list where every other entry was capitalized differently. A model follows a capitalization instruction most of the time, and "most of the time" is the failure mode, so `prose.title_case()` normalizes what comes back.

It is conservative in one specific way: a word that already contains a capital is left completely alone. Upper-casing first letters blindly turns "iPhone" into "IPhone" and "eBay" into "EBay", and those are precisely the words a reader notices. Only all-lowercase words are touched, so the rule can add a capital but never move one. A title you type by hand is never normalized at all.

**Numbers are assigned at first publish**, not at upload, and never reused. Uploading is the wrong moment: papers that failed validation, episodes still private, and the extra renderings created when comparing voice models would all consume numbers the feed never shows, so it would count 1, 2, 5, 9. Unpublishing keeps the number, so a listener's "episode 7" still means the same episode afterwards, and the gap it leaves is not backfilled.

A re-voiced rendering inherits its sibling's number. It is the same paper and the same discussion in a different voice, and publishing it unpublishes the other — calling it a new episode would advertise a duplicate.

The number appears as `itunes:episode` in the feed, and is omitted rather than sent as `0` for anything published before numbering existed, since Apple requires a non-zero integer.

### Categories

`[[categories]]` in `config.toml` is a fixed tag vocabulary — a slug (stored, and what appears in URLs) and a label (displayed). At library scale, free text costs more than it gives: `AI` / `ai` / `Machine Learning` become three tags for one idea and the filter stops meaning anything.

**An episode can carry several.** A paper can be both History and Economics, and forcing one choice hides it from whichever filter someone actually looks under.

There is deliberately **no "classic" tag**. Sorting by citations does that job better: it is measured rather than a judgement call, and it ranks rather than merely including.

The metadata stage suggests tags from the vocabulary while it is already reading the PDF, so new papers arrive tagged. Anything it returns that is not a configured slug is dropped rather than stored — an unknown tag would show up in no filter and silently take the episode out of every list it belongs in. Correct them per-episode in the admin.

The public page filters via `?category=<slug>`: real links, so each filter is shareable, bookmarkable, and works with the back button. Chips appear only for categories that have episodes, with counts taken from the same list being rendered.

Renaming a label is free. Changing a slug orphans every episode already tagged with it.

The admin library adds a **Show** row — All / Not published / Published — for reviewing what is not live yet. It is admin-only: the public page contains nothing but published episodes, and `?visibility=private` there is ignored rather than becoming a way to see drafts.

All three controls compose, and each chip's count is taken over what the *other* two leave — so a count always describes what clicking it gives you rather than a library-wide total the page would then contradict.

### Sorting, and citation counts

The library sorts three ways — newest episodes (the default), the paper's own publication date, and most cited — composable with the category filter and carried in the URL alongside it.

Citation counts come from [OpenAlex](https://openalex.org): free, no key, lookup by DOI where the paper printed one and by exact title match otherwise. A near-miss on the title is rejected, because attaching some other paper's count would then silently reorder the whole site.

**The count is never asked of the language model.** It is not printed in the paper, so a model has nothing to read it off and would supply a plausible number instead — and here that number drives what the public sees first. The DOI *is* printed, so that much is extracted normally.

Lookups fail silently and leave the count unset; nothing about a third-party outage can fail an episode, and a miss never overwrites a number entered by hand. Each episode has a **Look up again** button, since counts only go up. `[citations] enabled = false` turns the whole thing off.

Unknown and zero sort differently: a paper with no count sorts *below* one with a genuine zero, because "not looked up" and "never cited" are different facts.

### Citation flags

The script prompt forbids fabricated citations, but models fabricate anyway. After generation, the script is regex-scanned for citation-shaped strings — `Name (Year)`, `et al.`, and proper nouns sitting near a bare four-digit year — and every hit is surfaced on the episode page, both as a summary box and inline on the offending line.

Each hit is checked against the extracted text of the source PDF. Names that trace back are collapsed into a secondary list; names that do not are shown prominently, because those are the fabrications.

These are **flags, not failures**. Most hits are innocent ("the Mariel boatlift in 1980"). The point is to put anything that *could* be an invented citation in front of a person before publishing it.

### Web grounding

`[script] grounding` lets the script model search while writing. It is off by default, because it deliberately relaxes the rule the citation check depends on: without it, every claim must come from the PDF, and anything else is a suspected fabrication.

With it on:

- The model may name outside work, but only where a search supports it (`prompts/script_grounding.md` states the terms).
- The queries it ran and the pages it used are recorded and shown on the episode.
- A citation absent from the PDF but corroborated by a consulted source counts as traced, not invented. One that matches neither still flags.

Matching is on names rather than whole phrases, because a grounding source is titled after the paper rather than its authors — "Card and Krueger (1994)" never appears verbatim in a source called *Minimum Wages and Employment*, but both surnames do.

Search grounding bills per request on top of tokens, unlike the other script knobs.

## Submitting to Apple Podcasts and Spotify

`/admin/feed` checks the feed against what the directories actually enforce and links out to both submission forms. Worth reading before submitting: each validates once and tells you very little about what failed.

What the feed carries beyond plain RSS 2.0: `itunes:type`, per-item `itunes:explicit` and `itunes:episodeType`, `copyright`, `lastBuildDate`, artwork at channel and item level, and a nested `itunes:category`/subcategory. All configurable under `[feed]`.

Two things that are easy to get wrong and produce an unhelpful rejection:

- **Artwork must be square, 1400×1400 to 3000×3000.** This is the most common rejection. `static/cover.png` ships at 3000×3000 RGB, and a test asserts it stays inside the bounds.

  The source is `static/cover.svg` — edit that and re-render rather than editing the PNG, so the type stays sharp at 3000px. It is drawn to survive being shrunk: podcast apps show artwork at ~55px in the now-playing bar, so the wordmark is set large and the waveform uses few fat bars rather than many thin ones, which merge into a smear at that size.

  The tagline reads *"Difficult papers, explained simply"*. Episodes are **discussions of** papers, not readings of them — the script prompt caps verbatim quotation at fifteen consecutive words for exactly that reason. Wording that implies otherwise is a claim the project cannot make.
- **`itunes:category` must be spelled exactly as in Apple's list.** The readiness check holds the list and rejects anything else.

### The AI disclosure

Apple requires machine-generated audio to be disclosed in the show metadata, in every episode's metadata, **and in the content itself**. All three are covered:

| Where | What carries it |
| --- | --- |
| Show metadata | `[feed] description` |
| Episode metadata | the attribution line, which leads every `<description>` and `<itunes:summary>` |
| The audio | `[intro]` — a spoken sentence at the top of every episode |

The spoken one is `pipeline/intro.py`. Three things about it are deliberate:

- **The wording is a template, not model output.** A compliance statement a model paraphrases is one that can drift, and nothing downstream would notice. `$TITLE` and `$AUTHORS` are the only variable parts.
- **It is a separate single-voice call.** The multi-speaker API takes *exactly two* speaker configs — "Exactly two speaker voice configurations must be provided" — so a third voice cannot come from the same call as the hosts. The intro is synthesized on its own with `[intro] voice`, and assembly puts it in front of the dialogue with the usual seam silence. Pick a voice that is neither host or the handoff does not read as one.
- **It is `intro.wav`, not a numbered chunk.** The gap detection that spots a truncated episode counts `NNN.wav` against the manifest; a chunk that is not dialogue would confuse it.

The rendered sentence is stored beside the WAV as `intro.txt`. Editing a paper's title or authors changes the sentence, and the episode page then says the audio announces the old wording — nothing in the audio itself would reveal that. Retrying from **synthesizing** re-records only the intro; the dialogue chunks on disk are reused.

Episodes built before this existed have no intro and say so on their page. `/admin/feed` has a button that queues all of them at once.

That button only queues episodes whose dialogue WAVs are still on disk, where re-running synthesis costs one short call. An episode whose chunks are gone would pay for its whole script again — a different order of money — so it is listed separately and left for you to retry deliberately. The two look identical from the outside, which is exactly why the distinction is made in code rather than left to whoever presses the button.

`[intro] enabled = false` switches the whole thing off, which is a directory-compliance decision rather than a style one.

The enclosure URL answers `HEAD` as well as `GET`. Both directories probe it before accepting a feed, and FastAPI's `@app.get` alone returns 405 — which reads to them as a broken media URL rather than a missing method.

A missing duration omits `itunes:duration` rather than emitting the `--:--` the web UI uses for "unknown", which is not a duration and fails validation.

Publishing an episode adds it to the feed and unpublishing removes it; directories honour that on their next poll. A client that already downloaded an episode keeps its copy.

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
- `[intro]` — the spoken AI disclosure that opens every episode. See below.
- `[script]` — the content-quality knobs:
  - `target_words` (1600 ≈ ten minutes).
  - `max_pages` — an editorial limit, not a technical one. The API stops at 1000 pages and 50 MB per PDF; each page costs ~258 input tokens and the PDF is sent on both the metadata and the script call. It ships at **775**, the page count at which a single call reaches the 200k input tokens where Pro's rate doubles — a real price cliff rather than a round number, and the point past which `[costs]` would understate what an episode cost. Override with `PAPERPOD_MAX_PAGES` rather than editing the file, so one awkward paper does not need a code change and a redeploy.
  - `thinking_level` — `MINIMAL`/`LOW`/`MEDIUM`/`HIGH`. How hard the model reasons before writing. Scripting is ~2% of an episode's cost, so this is cheap to raise and it is what dense technical papers reward.
  - `grounding` — let the script model search the web. Off by default; see below.
  - `fallback_model` — used if the script model has no quota, rather than failing the episode. The substitution is recorded in the stage log and shown on the episode.
- `[audio]` — `seam_silence_ms` is the main quality tell. It ships at 250ms; try 150 and 400 and pick by ear.
- `[server]` — `base_url` (see Tailscale above), `port`, and `workers`.
  - `workers` is how many episodes process at once. The pipeline spends nearly
    all its wall-clock waiting on Gemini, so extra workers cost almost no CPU.
    The real ceilings are the account's requests-per-minute and disk headroom
    during assembly, which briefly needs ~2× an episode's chunk audio. Ships at
    2; override per-deployment with `PAPERPOD_WORKERS` rather than editing the
    file. Raise it a step at a time and watch for 429s in the logs.
- `[retry]` — `request_timeout_s` is the ceiling on a single API call. Without
  one, a stalled connection blocks its worker indefinitely: the episode sits in
  `synthesizing` with no error and nothing behind it starts. Timeouts count as
  retryable, so one bad connection costs a retry rather than the chunk.
  `max_delay_s` also caps how long a server-named rate-limit window can hold
  the whole process, so an absurd `retryDelay` cannot wedge a worker.
- `[site]` — `owner_name` and `contact_email` (shown on the terms page, and used as the podcast owner contact in the feed), plus `analytics_id`.
  - `analytics_id` is a Google Analytics measurement ID. **Empty disables it entirely** — no script tag, no request to Google — and the terms page changes to match. It is never loaded for a signed-in admin, so editing the library does not turn up in the numbers as reader traffic.
- `[costs]` — per-model token prices used to compute the per-episode cost shown in the UI.

Prompts live in `prompts/` as plain Markdown, loaded at call time rather than baked into Python. `script_system.md` is the main quality lever — edit it freely without touching code.

### Editing prompts from the browser

`/admin/prompts` edits every prompt in place. They are read at call time, so a save takes effect on the next episode with no redeploy and no restart.

Edits are written to `$PAPERPOD_DATA_DIR/prompts/`, not over the files in `prompts/`. Two reasons: the repo copy is baked into the container image, so a redeploy would wipe anything written there; and keeping them separate means the shipped default stays readable as the thing an edit can always be reverted to. Saving the default back verbatim is treated as a revert, so an unmodified copy never freezes against future upstream edits.

Saving is never blocked, but an edit that drops something the pipeline depends on says so:

- **A missing `$PLACEHOLDER`** — whatever it carried simply stops being sent. `$TARGET_WORDS` gone means no length budget; `$SCRIPT` gone means a revision sends no script at all. Neither is visible in the output. The check is derived from the shipped default, so it needs no list to maintain.
- **`script_system.md` losing `HOST_A`** — every script then fails format validation and the episode fails.
- **`script_system.md` losing its no-fabrication rule** — the citation flags only catch what slips past that rule; they are not a substitute for it.

### Is the deploy actually current?

A stale deploy has no symptom of its own. A fixed bug reads as unfixed, a raised limit reads as ignored, and the only tell is noticing that an error message is worded the way it was two releases ago — which is exactly how the 120-page limit went on rejecting papers after it had been raised to 400.

So the image carries a build stamp. The Dockerfile writes `/app/BUILD_STAMP` immediately after copying the source in, so the stamp cannot outlive the code it describes, and it appears in the admin sidebar and in `/health` as `build`. If that date predates the change you are looking for, the deploy is the problem and nothing else is.

## Tests

```bash
python -m pytest tests/ -q
```

150 tests. Only the Gemini calls are stubbed; ffmpeg assembly, loudness normalization, MP3 encoding, HTTP range serving, and RSS generation all run for real. Tests requiring ffmpeg skip cleanly if it is not installed.

## Model IDs and the upstream docs

`ai.google.dev` is unreachable from the sandbox this was built in (HTTP 403 through the egress proxy), so the two documentation pages could not be read directly. The API surface was instead verified against the installed `google-genai` SDK (2.17.0), whose type definitions are the binding contract, and the TTS model details cross-checked against published sources.

Confirmed in use:

- `gemini-3.1-flash-tts-preview` supports **at most two speakers** and returns **24 kHz 16-bit mono PCM** — which is why `tts.py` parses the sample rate out of the response MIME type and wraps the PCM in a WAV header rather than guessing.
- Every type used (`SpeechConfig`, `MultiSpeakerVoiceConfig`, `SpeakerVoiceConfig`, `VoiceConfig`, `PrebuiltVoiceConfig`) exists in the SDK with the field names used here.
- `gemini-3-flash-preview` works for metadata and scripting. `gemini-3-pro-preview` returned `limit: 0` on a paid key, so preview Pro appears to need separate access — and was later withdrawn entirely.

### Why the text models are aliases

`[models] metadata` and `script` point at `gemini-flash-latest` and `gemini-pro-latest` rather than dated preview IDs. `gemini-3-pro-preview` was withdrawn while `models.list()` still advertised it, and every episode 404'd until someone noticed — so the catalogue proves a model *existed*, not that it works. Google repoints the aliases, which turns a retirement into a quality change rather than an outage.

The trade-offs are real: an alias can move under you, so output is not reproducible across time, and the model behind it may be repriced without `[costs]` noticing. Pin a dated ID if you need reproducibility, and accept that you are then responsible for retirements. TTS has no alias, so it stays pinned — check `/admin/models` when it starts 404ing.

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

The two text-model prices in `[costs]` are still estimates — correct them against current Gemini pricing so the per-episode figure means something.

## Layout

```
config.toml          config, loaded once at startup
app.py               FastAPI routes, background worker, inbox watcher
db.py                schema + queries (sqlite3 stdlib, no ORM)
prose.py             paper metadata rendered as English, shared by web + pipeline
pipeline/
  ingest.py          PDF in, metadata + validation out
  script.py          paper in, dialogue script out, citation flagging
  intro.py           the spoken AI disclosure, in its own voice
  tts.py             script in, audio chunks out
  assemble.py        chunks in, normalized MP3 out
  run.py             orchestrator, stage state machine
prompts/             editable without touching code
templates/ static/   Jinja2 + vanilla JS, no build step
tools/make_cover.py  regenerates static/cover.png
data/                inbox/, papers/, audio/chunks/, audio/final/, paperpod.db
```
