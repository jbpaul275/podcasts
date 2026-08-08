"""Render static/cover.svg to the PNGs the site and the feed point at.

Two sizes, for two different jobs:

- cover.png at 3000px is what Apple and Spotify ingest. They want square art
  between 1400 and 3000, and directories keep the file rather than re-fetching
  it, so it is worth shipping at the top of the range.
- cover-web.png is what the site header loads. Serving the 3000px file to every
  visitor would be a megabyte to draw a 112px square.

The source of truth is the SVG. Editing a PNG instead means the next re-render
silently throws the edit away.

Usage:  python tools/make_cover.py
Needs Chromium, which is a development dependency only -- the rendered PNGs are
committed, so nothing at runtime or in the container needs a browser.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "static" / "cover.svg"

# (filename, pixels). The feed size first: it is the one with a hard external
# requirement, so a failure there should be the first thing that shows up.
OUTPUTS = [("cover.png", 3000), ("cover-web.png", 512)]

CHROME_CANDIDATES = [
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    "chromium",
    "chromium-browser",
    "google-chrome",
]


def _chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        found = candidate if Path(candidate).exists() else shutil.which(candidate)
        if found:
            return found
    raise SystemExit(
        "No Chromium found. This is a development-only tool; the rendered PNGs "
        "are committed, so nothing else needs a browser.\n"
        "Looked for: " + ", ".join(CHROME_CANDIDATES)
    )


def render(size: int, out: Path) -> None:
    chrome = _chrome()
    with tempfile.TemporaryDirectory() as tmp:
        # Chromium screenshots the viewport, so the page has to be exactly the
        # square wanted -- no margin, no scrollbars, no white gutter.
        page = Path(tmp) / "page.html"
        page.write_text(
            "<style>html,body{margin:0;padding:0;overflow:hidden}"
            f"img{{display:block;width:{size}px;height:{size}px}}</style>"
            f'<img src="{SVG.as_uri()}">',
            encoding="utf-8",
        )
        subprocess.run(
            [chrome, "--headless", "--no-sandbox", "--disable-gpu",
             # No transparent background: that yields RGBA, and directories
             # want plain RGB. The artwork is full-bleed, so nothing shows
             # through anyway.
             "--hide-scrollbars",
             f"--window-size={size},{size}",
             f"--screenshot={out}", page.as_uri()],
            check=True, capture_output=True,
        )


def main() -> int:
    if not SVG.exists():
        raise SystemExit(f"missing {SVG}")
    for name, size in OUTPUTS:
        out = ROOT / "static" / name
        render(size, out)
        print(f"{name}: {size}x{size}, {out.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
