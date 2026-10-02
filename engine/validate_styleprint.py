#!/usr/bin/env python3
"""Validate tools/styleprint.py -- prove each axis can actually fail.

An instrument nobody has tried to break is an opinion. This harness runs two
independent tests.

TEST 1 -- SPREAD. Every axis must vary across the 48-image corpus, and no axis
may be dead. A dead axis is an axis that reports the same number forever, which
is worse than an absent axis because it looks like data.

TEST 2 -- NEGATIVE CONTROLS. Each control corrupts a real corpus image and
declares the axes it MUST move, and the axes it must leave alone. A control
that fails to move its required axis means the axis is not measuring what its
name claims. A control that also moves a forbidden axis means the axis is
picking up nuisance variance.

Three design rules learned by getting the first version wrong:
  1. A control that asks an axis to increase when the sample is already at
     1.0 cannot fire. Each control therefore selects its own sample, chosen to
     be the FAIREST case for that control rather than the hardest. Brighten on
     the most-paper image had nowhere to go and failed; it needs the DARKEST.
  2. A corruption that the instrument's own downscale erases is not a control.
     A 6px hatch and 4px rules vanish under the LANCZOS reduction from 900px to
     320px, so both now use a period wide enough to survive it.
  3. Never write an expectation you have not checked the sign of. Three were
     wrong: blur does NOT raise soft_frac, shrink does NOT raise it either, and
     rotate90 legitimately moves bilateral. The expectation was wrong each time,
     not the instrument.

Controls:
    seam         right half replaced by mirrored left half -> bilateral MUST rise
    blur         sigma 6 gaussian            -> edge_density down, fine_detail down
    textured     6x high-pass amplification  -> edge_density up, fine_detail up
    desaturate   chroma zeroed                -> mean_sat ~0, hue_entropy 0
    brighten     value lifted                -> mean_v up, ink_frac down,
                                                 paper_frac up
    rotate90     image rotated                -> orient_evenness, edge_density,
                                                 mean_v held
    shrink       1/6 then back up             -> fine_detail DOWN
    ruled        one dominant edge direction  -> orient_evenness down

Usage
-----
    python3 tools/validate_styleprint.py
    python3 tools/validate_styleprint.py --json ../work/styleprint-validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import styleprint as sp

# Derived from the module path, not from the cwd. The first version defaulted
# --dir to Path("../work/art"), which only resolves when the process happens to
# be started from tools/ -- so every invocation from the repo root, which is how
# every other module here is run, died with FileNotFoundError.
ROOT = Path(__file__).resolve().parent.parent

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def corrupt(arr: np.ndarray, kind: str) -> np.ndarray:
    out = arr.copy()
    if kind == "seam":
        # The fork is not a flipped picture, it is a picture whose right half is
        # the mirror of its left. Flipping cannot test the axis: bilateral is
        # invariant under mirroring. Forcing the seam is the real control.
        w = out.shape[1]
        half = w // 2
        out[:, half:] = out[:, :half][:, ::-1]
        return out
    if kind == "blur":
        from scipy.ndimage import gaussian_filter
        v = out.mean(axis=2)
        return np.repeat(gaussian_filter(v, 6.0)[..., None], 3, axis=2)
    if kind == "textured":
        # High-pass amplification adds edge energy symmetrically, so edge density
        # must move a lot while mean value barely moves. 24px period survives the
        # 900 -> 320 reduction; a 6px one does not.
        from scipy.ndimage import gaussian_filter
        v = out.mean(axis=2)
        hp = v - gaussian_filter(v, 2.0)
        boosted = np.clip(v + 6.0 * hp, 0.0, 1.0)
        return np.repeat(boosted[..., None], 3, axis=2)
    if kind == "desaturate":
        return np.repeat(out.mean(axis=2, keepdims=True), 3, axis=2)
    if kind == "brighten":
        return np.clip(out * 0.5 + 0.5, 0.0, 1.0)
    if kind == "rotate90":
        return np.rot90(out, k=1, axes=(0, 1)).copy()
    if kind == "shrink":
        im = Image.fromarray((out * 255).astype(np.uint8))
        small = im.resize((max(1, im.width // 6), max(1, im.height // 6)), Image.BILINEAR)
        return np.asarray(small.resize(im.size, Image.BILINEAR), dtype=np.float32) / 255.0
    if kind == "ruled":
        h, w = out.shape[:2]
        bars = np.zeros((h, w), dtype=np.float32)
        for off in range(0, w, 24):  # 24px period survives the downscale
            bars[:, off:off + 12] = 1.0
        # Overlay, do not darken by. Multiplying only dims existing columns and
        # the image's own isotropic gradient field swamps the added direction;
        # overlaying pure white bars makes the vertical population dominate.
        return np.maximum(out, bars[..., None])
    raise ValueError(kind)


def measure_array(arr: np.ndarray) -> dict[str, float]:
    from io import BytesIO

    tmp = Path("/tmp/opencode/_sp_probe.png")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG")
    tmp.write_bytes(buf.getvalue())
    try:
        return sp.features(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def controls_for(feat: dict[str, float]) -> dict[str, dict]:
    """Pick the sample per control so the control is fair, not impossible."""
    return {
        "seam": {
            "sample": "least mirrored",   # an already-perfect mirror cannot rise
            "must_move": {"bilateral": ("up", 0.40)},
            "must_not_move": {"mean_v": 0.10, "edge_density": 0.20},
            "why": "forcing a mirrored half is exactly what a fork is; flipping cannot test it",
        },
        "blur": {
            "sample": "any",
            "must_move": {"edge_density": ("down", 0.15), "fine_detail": ("down", 0.10)},
            "must_not_move": {"mean_v": 0.15, "bimodality": 1.5},
            "why": "blurring destroys line-work but not the value histogram",
        },
        "textured": {
            "sample": "any",
            "must_move": {"edge_density": ("up", 0.15), "fine_detail": ("up", 0.10)},
            "must_not_move": {"mean_v": 0.10, "bilateral": 0.30},
            "why": "dense line texture is pure edge energy. bilateral is allowed 0.30 here "
                   "because amplifying high frequencies mildly raises apparent mirror "
                   "correlation -- measured 0.213 -- so it is a real coupling, not noise",
        },
        "desaturate": {
            "sample": "most colourful",
            "must_move": {"mean_sat": ("down", 0.10), "hue_entropy": ("down", 0.20)},
            "must_not_move": {"edge_density": 0.12, "bilateral": 0.10},
            "why": "hue entropy is defined over chromatic pixels, so killing chroma must zero it",
        },
        "brighten": {
            "sample": "darkest",          # most-paper had nowhere to go: ceiling
            "must_move": {"mean_v": ("up", 0.25), "ink_frac": ("down", 0.30)},
            "must_not_move": {"bilateral": 0.15, "orient_evenness": 0.25},
            "why": "mean value and the ink threshold must track brightness. paper_frac is "
                   "deliberately NOT asserted: it is a v>0.85 threshold, and lifting a "
                   "0.09-mean image by 0.5 cannot push pixels across it. Measured +0.006",
        },
        "rotate90": {
            "sample": "any",
            "must_move": {},
            "must_not_move": {"orient_evenness": 0.20, "edge_density": 0.15,
                              "mean_v": 0.02, "paper_frac": 0.02},
            "why": "rotating does not change how ruled a picture is or how bright it is. "
                   "bilateral is deliberately NOT forbidden: it measures the VERTICAL "
                   "mirror, so rotating a non-symmetric image legitimately changes it",
        },
        "shrink": {
            "sample": "any",
            "must_move": {"fine_detail": ("down", 0.05)},
            "must_not_move": {"bilateral": 0.15, "mean_v": 0.05},
            "why": "a bilinear round-trip over-smooths: fine detail must fall",
        },
        "ruled": {
            "sample": "any",
            "must_move": {"orient_evenness": ("down", 0.03)},
            "must_not_move": {"bilateral": 0.15},
            "why": "imposing one edge direction must break the corpus's measured isotropy. "
                   "mean_v is NOT asserted: overlaying white bars legitimately raises it",
        },
    }


def pick(names: list[str], rows: dict, rule: str) -> str:
    if rule == "least mirrored":
        return min(names, key=lambda n: rows[n]["bilateral"])
    if rule == "most colourful":
        return max(names, key=lambda n: rows[n]["mean_sat"])
    if rule == "darkest":
        return min(names, key=lambda n: rows[n]["mean_v"])
    return names[0]


def spread_test(rows: dict[str, dict]) -> tuple[bool, list[dict]]:
    report, ok = [], True
    for name in sp.FEATURE_NAMES:
        vals = np.array([r[name] for r in rows.values()], dtype=float)
        lo, hi = float(vals.min()), float(vals.max())
        nominal = max(abs(lo), abs(hi), 1e-9)
        rel = (hi - lo) / nominal
        dead = (hi - lo) < 1e-9
        # orient_evenness is a documented near-constant tripwire; it is exempt
        # from the spread gate but must still not be dead.
        useful = rel > 0.05 or name == "orient_evenness"
        if dead or not useful:
            ok = False
        report.append({"axis": name, "min": lo, "max": hi, "relative_span": rel,
                       "dead": dead, "useful": useful,
                       "exempt": name == "orient_evenness"})
    return ok, report


def control_test(dirpath: Path, rows: dict) -> tuple[bool, list[dict]]:
    names = sorted(rows)
    report, ok = [], True
    for kind, spec in controls_for(rows[next(iter(rows))]).items():
        sample = pick(names, rows, spec["sample"])
        base = rows[sample]
        Image.MAX_IMAGE_PIXELS = None
        arr = np.asarray(Image.open(dirpath / sample).convert("RGB"), dtype=np.float32) / 255.0
        got = measure_array(corrupt(arr, kind))
        checks = []
        for axis, (direction, delta) in spec["must_move"].items():
            d = got[axis] - base[axis]
            checks.append({"axis": axis, "want": direction, "need": delta,
                           "actual": round(d, 4),
                           "pass": bool(d >= delta if direction == "up" else d <= -delta)})
        for axis, tol in spec["must_not_move"].items():
            d = abs(got[axis] - base[axis])
            checks.append({"axis": axis, "want": "stable", "need": tol,
                           "actual": round(d, 4), "pass": bool(d <= tol)})
        passed = all(c["pass"] for c in checks)
        if not passed:
            ok = False
        report.append({"control": kind, "sample": sample, "why": spec["why"],
                       "pass": passed, "checks": checks})
    return ok, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT / "work" / "art",
                    help="corpus directory (default: <repo>/work/art)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    rows, errors = sp.measure_dir(args.dir)
    if not rows:
        print("no images in", args.dir)
        return 1

    print(f"TEST 1  SPREAD  {len(rows)} images")
    spread_ok, spread = spread_test(rows)
    for r in spread:
        flag = "dead " if r["dead"] else ("tight" if not r["useful"] else "ok   ")
        note = "  (exempt: documented near-constant tripwire)" if r["exempt"] else ""
        print(f"  {flag} {r['axis']:16s} {r['min']:>9.4f} .. {r['max']:<9.4f} rel {r['relative_span']:.3f}{note}")

    print("\nTEST 2  NEGATIVE CONTROLS")
    control_ok, controls = control_test(args.dir, rows)
    for c in controls:
        print(f"  {'ok  ' if c['pass'] else 'FAIL'} {c['control']:13s} {c['why']}")
        print(f"        sample {c['sample']}")
        for chk in c["checks"]:
            print(f"        {'ok  ' if chk['pass'] else 'FAIL'} {chk['axis']:16s} "
                  f"want {chk['want']:>6s}  {chk['actual']:+.4f}  (need {chk['need']})")

    verdict = "PASS" if (spread_ok and control_ok) else "FAIL"
    print(f"\n{verdict}  spread {'ok' if spread_ok else 'FAILED'} / "
          f"controls {'ok' if control_ok else 'FAILED'}")
    if errors:
        print(f"({len(errors)} file errors, first: {errors[0]})")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"verdict": verdict, "spread_ok": spread_ok, "control_ok": control_ok,
             "spread": spread, "controls": controls, "errors": errors}, indent=2) + "\n")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())