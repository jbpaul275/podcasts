"""Generate static/cover.png, the artwork the RSS feed points at.

Stdlib-only PNG encoder so the repo needs no image dependency. Podcast clients
want square art; 1400x1400 is the smallest size Apple accepts.
"""

import struct
import sys
import zlib
from pathlib import Path

SIZE = 1400


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def make_cover(path: Path, size: int = SIZE) -> None:
    top = (24, 30, 42)
    bottom = (46, 62, 92)
    bar = (208, 214, 226)

    rows = bytearray()
    for y in range(size):
        t = y / (size - 1)
        r = round(top[0] + (bottom[0] - top[0]) * t)
        g = round(top[1] + (bottom[1] - top[1]) * t)
        b = round(top[2] + (bottom[2] - top[2]) * t)
        row = bytearray([0])  # filter byte: none
        # A stack of "paper" lines across the middle, ragged like a page of text.
        line_band = 0.30 <= t <= 0.70
        band_index = int((t - 0.30) / 0.055) if line_band else -1
        widths = [0.62, 0.74, 0.55, 0.80, 0.48, 0.70, 0.36]
        width = widths[band_index % len(widths)] if line_band else 0
        thick = line_band and (int(y * 1.0) % 77) < 34
        x_start = int(size * 0.16)
        x_end = x_start + int(size * width)
        for x in range(size):
            if thick and x_start <= x < x_end:
                row += bytes(bar)
            else:
                row += bytes((r, g, b))
        rows += row

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent / "static" / "cover.png"
    )
    make_cover(out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
