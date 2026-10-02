#!/usr/bin/env python3
"""Downscale the Dropbox reference art so it can be read cheaply and repeatedly.

The art is the canon for this series, so every image has to be looked at. Reading
the 1200x1200 originals works but costs context on every pass; a 900px derivative
is enough to hold composition, palette and the small type, and it is regenerated
deterministically so the notes can always be checked against the same pixels.

Usage:
  python3 tools/art_ingest.py                    # (re)build work/art/
  python3 tools/art_ingest.py --check            # exit 1 if the manifest is stale
  python3 tools/art_ingest.py --dir ART --glob '*.jpg' --out work/art
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent

# The default input is the campaign's own corpus and the default glob is that
# corpus' filename convention, so the bare command keeps behaving exactly as it
# did before these flags existed. A buyer pointing this at their own art folder
# passes --dir and --glob; leaving the defaults in place would silently find
# nothing in a directory that is full of images.
DEFAULT_DIR = Path.home() / "Dropbox"
DEFAULT_GLOB = "Lucid_Origin_*.jpg"
DEFAULT_OUT = HERE.parent / "work" / "art"
SIZE = 900


def manifest(src_dir: Path, pattern: str) -> list[dict]:
    rows = []
    for src in sorted(src_dir.glob(pattern)):
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


def build(rows: list[dict], src_dir: Path, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for row in rows:
        dst = out / row["name"]
        with Image.open(src_dir / row["name"]) as im:
            im = im.convert("RGB")
            im.thumbnail((SIZE, SIZE), Image.LANCZOS)
            im.save(dst, "JPEG", quality=88, optimize=True)
        written.append(dst.name)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help="folder of source images")
    ap.add_argument("--glob", default=DEFAULT_GLOB,
                    help="filename glob inside --dir (pass '*.jpg' for your own art)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="where the derivatives and the manifest are written")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the manifest or a derivative is stale")
    args = ap.parse_args()

    src_dir, out = args.dir, args.out
    if not src_dir.is_dir():
        print(f"FAIL: no such directory: {src_dir}")
        return 1
    rows = manifest(src_dir, args.glob)
    if not rows:
        print(f"FAIL: no image matched {args.glob!r} in {src_dir}")
        return 1

    mf = out.parent / "art-manifest.json"
    if args.check:
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
            r["name"] for r in rows if not (out / r["name"]).exists()
        ]
        if stale or missing:
            print("FAIL:", stale or missing)
            return 1
        print(f"OK: {len(rows)} images, manifest and derivatives current")
        return 0
    build(rows, src_dir, out)
    mf.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} derivatives to {out} and {mf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())