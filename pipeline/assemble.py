"""Assembly: chunk WAVs in, loudness-normalized MP3 with ID3 tags out.

Uses the system ffmpeg binary: concat demuxer with configurable inter-chunk
silence, two-pass loudnorm to the mono podcast standard, 96k mono MP3.
"""

import json
import logging
import re
import subprocess
import wave

import db
from config import CHUNKS_DIR, FINAL_DIR
from . import PipelineError

log = logging.getLogger("paperpod.assemble")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PipelineError(
            f"command failed ({' '.join(cmd[:4])}...):\n{proc.stderr[-2000:]}"
        )
    return proc


def _expected_chunks(chunk_dir, fallback: int) -> int:
    """Chunk count from the manifest TTS wrote, which is the only record of how
    long the script actually was once failed chunks are missing from disk."""
    try:
        manifest = json.loads((chunk_dir / "manifest.json").read_text(encoding="utf-8"))
        if isinstance(manifest, list) and manifest:
            return len(manifest)
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return fallback


def assemble(episode_id: str, cfg: dict) -> None:
    chunk_dir = CHUNKS_DIR / episode_id
    wavs = sorted(chunk_dir.glob("[0-9][0-9][0-9].wav"))
    if not wavs:
        raise PipelineError(f"no audio chunks found in {chunk_dir}")

    # How many chunks TTS was supposed to produce. Deriving the expected count
    # from the files on disk cannot see a truncated tail: if the last chunks all
    # failed, the highest surviving sequence number looks like the end of the
    # script, and a half-length episode assembles silently.
    seqs = [int(p.stem) for p in wavs]
    expected = _expected_chunks(chunk_dir, fallback=seqs[-1] + 1)
    missing = sorted(set(range(expected)) - set(seqs))
    if missing:
        log.warning(
            "assembling %d of %d chunks; missing %s", len(seqs), expected, missing
        )

    acfg = cfg["audio"]
    silence_ms = int(acfg.get("seam_silence_ms", 250))

    # Silence file matching the chunk format.
    with wave.open(str(wavs[0]), "rb") as w:
        rate = w.getframerate()
    silence = chunk_dir / "_silence.wav"
    _run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r={rate}:cl=mono",
        "-t", f"{silence_ms / 1000:.3f}",
        "-sample_fmt", "s16", str(silence),
    ])

    concat_list = chunk_dir / "_concat.txt"
    lines = []
    for i, wav in enumerate(wavs):
        if i:
            lines.append(f"file '{silence.name}'")
        lines.append(f"file '{wav.name}'")
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

    joined = chunk_dir / "_joined.wav"
    _run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list), "-c", "copy", str(joined),
    ])

    # Two-pass loudnorm.
    I = acfg.get("lufs_target", -16.0)
    TP = acfg.get("true_peak", -1.5)
    LRA = acfg.get("lra", 11.0)
    measured = _measure_loudnorm(joined, I, TP, LRA)

    ep = db.get_episode(episode_id)
    authors = db.episode_authors(ep) if ep else []
    title = (ep["title"] if ep and ep["title"] else episode_id)
    year = ep["year"] if ep else None

    out_path = FINAL_DIR / f"{episode_id}.mp3"
    filt = (
        f"loudnorm=I={I}:TP={TP}:LRA={LRA}"
        f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(joined),
        "-af", filt,
        "-ar", "44100", "-ac", "1", "-b:a", acfg.get("bitrate", "96k"),
        "-id3v2_version", "3",
        "-metadata", f"title={title}",
        "-metadata", f"artist={', '.join(authors) if authors else 'Unknown'}",
    ]
    if year:
        cmd += ["-metadata", f"date={year}"]
    comment_bits = [b for b in [ep["venue"] if ep else None, ep["source_path"] if ep else None] if b]
    if comment_bits:
        cmd += ["-metadata", f"comment={' | '.join(comment_bits)}"]
    cmd.append(str(out_path))
    _run(cmd)

    duration = _probe_duration(out_path)
    db.update_episode(episode_id, audio_path=str(out_path), duration_s=duration)

    if missing:
        db.stage_start(episode_id, "assembling:gaps")
        db.stage_end(
            episode_id, "assembling:gaps", ok=False,
            detail=(f"INCOMPLETE: assembled {len(seqs)} of {expected} chunks. "
                    f"Missing chunk(s) {missing} — the audio is short by that much. "
                    f"Retry from synthesizing to fill them in."),
        )

    for tmp in (silence, concat_list, joined):
        tmp.unlink(missing_ok=True)


REQUIRED_LOUDNORM_KEYS = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")


def _measure_loudnorm(path, I, TP, LRA) -> dict:
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
            "-af", f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    # ffmpeg prints its stream summary after the loudnorm block, so the JSON is
    # not at the end of stderr. Take the last brace block that actually parses
    # and carries the measurement keys.
    for block in reversed(re.findall(r"\{[^{}]*\}", proc.stderr)):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if all(k in data for k in REQUIRED_LOUDNORM_KEYS):
            return data
    raise PipelineError(
        f"loudnorm measurement pass produced no usable JSON:\n{proc.stderr[-1500:]}"
    )


def _probe_duration(path) -> float | None:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        return round(float(proc.stdout.strip()), 2)
    except ValueError:
        return None
