#!/usr/bin/env python3
"""Downscale the Dropbox reference art so it can be read cheaply and repeatedly.

The art is the canon for this series, so every image has to be looked at. Reading
the 1200x1200 originals works but costs context on every pass; a 900px derivative
is enough to hold composition, palette and the small type, and it is regenerated
deterministically so the notes can always be checked against the same pixels.

Usage:
  python3 tools/art_ingest.py          # (re)build work/art/
  python3 tools/art_ingest.py --check  # exit 1 if the manifest is stale
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

DROPBOX = Path.home() / "Dropbox"
OUT = Path(__file__).resolve().parent.parent / "work" / "art"
SIZE = 900


def manifest() -> list[dict]:
    rows = []
    for src in sorted(DROPBOX.glob("Lucid_Origin_*.jpg")):
        with Image.open(src) as im:
            w, h = im.size
        rows.append(
            {
                "name": src.name,
                "src_bytes": src.stat().st_size,
                "w": w,
                "h": h,
                "sha8": hashlib.sha256(src.read_bytes()).hexdigest()[:8],
            }
        )
    return rows


def build(rows: list[dict]) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for row in rows:
        dst = OUT / row["name"]
        with Image.open(DROPBOX / row["name"]) as im:
            im = im.convert("RGB")
            im.thumbnail((SIZE, SIZE), Image.LANCZOS)
            im.save(dst, "JPEG", quality=88, optimize=True)
        written.append(dst.name)
    return written


def main() -> int:
    rows = manifest()
    mf = OUT.parent / "art-manifest.json"
    if "--check" in sys.argv:
        if not mf.exists():
            print("FAIL: work/art-manifest.json missing")
            return 1
        old = json.loads(mf.read_text())
        stale = [
            r["name"]
            for r, o in zip(rows, old)
            if r != o
        ] or (["manifest length changed"] if len(rows) != len(old) else [])
        missing = [
            r["name"] for r in rows if not (OUT / r["name"]).exists()
        ]
        if stale or missing:
            print("FAIL:", stale or missing)
            return 1
        print(f"OK: {len(rows)} images, manifest and derivatives current")
        return 0
    build(rows)
    mf.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} derivatives to {OUT} and {mf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())