#!/usr/bin/env python3
"""rekey.py -- turn a buyer's own reference images into that buyer's own cover prompts.

This is the product. styleprint answers "is this image in the series"; prompts_chatgpt
answers "what do I write for MY 48 issues". Neither helps anybody who has their own art.
rekey closes that gap: point it at a directory of the buyer's reference images and a
JSON list of targets, and it emits one prompt per target in the buyer's own measured
style, plus that buyer's own postgate thresholds.

    python3 tools/rekey.py --ref DIR --targets FILE.json --out DIR
    python3 tools/rekey.py --ref DIR --calibrate          # their thresholds
    python3 tools/rekey.py --ref DIR --validate           # does it actually work
    python3 tools/rekey.py --ref DIR --print 3            # one prompt to stdout

A target file is a JSON list; each entry needs `subject`, and may carry:

    name        label used in the output heading            (default "NN")
    reference   filename inside --ref, matched exactly       (default: nearest by index)
    geometry    one sentence on how the subject is arranged
    counts      an integer fold count, e.g. 8                (emits a rosette reference)
    text        "texture" | "none" | {"allow": ["STRING"]}

--------------------------------------------------------------------------
WHAT rekey MEASURES, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------------------------------
The KEEP clause of each prompt is written from numbers, not from adjectives. For the
chosen reference image we measure the twelve styleprint axes and print both the number
and a plain-language descriptor built from bands calibrated on the 48-image corpus
(bands below, with the measured gap each one sits in). An invented hex palette is the
failure mode this replaces: I could not tell whether "duotone limited palette" matched
anything, and it never did.

THE AXES MEASURE SERIES MEMBERSHIP, NOT FAMILY IDENTITY. Measured, not assumed:
single-linkage clustering of the 48 in z-scored axis space yields ONE cluster of 42
images holding 9 of the 13 visual sets, purity 0.10. At t=3.5 there are 5 clusters and
at t=2.0 there are 17. No cut separates the visual families, because the corpus is
narrow on these axes -- mean_sat q25 0.397 to q75 0.605, bimodality 2.78 to 3.63 -- while
looking wildly different to a person. So this module does NOT try to detect a family and
does NOT cluster anything into one. The buyer names the reference image per target. That
is the reference-anchored premise stated honestly: choosing the reference is a creative
act, and no statistic here is entitled to make it.

The per-axis min/max printed in each prompt is DESCRIPTIVE PROSE, not a gate. A min/max
envelope was tried as the series gate and REFUTED: a sigma-14 blur stayed inside the box
on all twelve axes. The gate is nearest-neighbour distance in the same space, and that
gate catches a blurred image outside the tolerance (8.50 against 7.08 here).

--------------------------------------------------------------------------
DESCRIPTOR BANDS -- every threshold sits in a measured gap
--------------------------------------------------------------------------
    PAPER_PLATE  paper_frac > 0.15   corpus has 5 values >= 0.237 and 48 with the next
                                     at 0.113: a 2.4x gap, and the 5 are exactly the
                                     cream-plate images (Forge_2 pen-and-ink, The_Roses
                                     _3 engraved, markdown_3 drafting table, the Chaos
                                     cream artboard, the white-ground node diagram)
    LIGHT_V      mean_v     > 0.55   3 of 48; next is 0.491
    SAT_HIGH     mean_sat   > 0.40   q25 = 0.397, so roughly the top three quarters
    SAT_LOW      mean_sat   < 0.20   only the paper plates; line-work on paper is grey
    BIMO_STRONG  bimodality > 3.60   q75 = 3.633
    EDGE_FINE    edge_density > 0.80 q75 = 0.804
    SOFT_MUCH    soft_frac  > 0.20   q75 = 0.248
    BILAT_STRONG bilateral  > 0.70   BILAT_SOME > 0.25, min -0.323

--------------------------------------------------------------------------
KNOWN LIMITS
--------------------------------------------------------------------------
1. The descriptors are calibrated on ONE corpus (48 images, one artist). On a different
   corpus the BANDS may be wrong even though the NUMBERS are right. --calibrate reports
   the buyer's own bands so the constants can be seen rather than trusted.
2. Naming a string does not guarantee the renderer spells it. Measured on this corpus:
   issue 42's real word `Amphetamemes` reads at confidence 91.9 while gibberish in the
   same image reads 89-93, and issue 30's real word in an ornate display face reads 67.9,
   below the gibberish. Post-render OCR is the only check that works.
3. A blurred image is caught. The figure depends on the base image: this harness
   puts its own sigma-14 victim at 8.50 against a tolerance of 7.08 (outside);
   postgate's victim lands at 9.50 standalone, 10.00 re-encoded as JPEG.
   A NOISE image is NOT: postgate measures it at 4.10 in-harness / 5.73
   standalone, both inside the tolerance, because a noisy image's axes genuinely resemble
   the series' busiest member. Not tuned away -- it is a real property of the metric.
4. One reference image is a weak description of a series. 4+ references give the kNN
   baseline room to mean something; with 1 the tolerance is degenerate and --validate
   says so instead of printing a confident number.
5. Nothing here reads meaning. All twelve axes are photometric.

--------------------------------------------------------------------------
ENVIRONMENT
--------------------------------------------------------------------------
numpy, Pillow and scipy only. No GPU. tesseract is shelled out to for the OCR gates and
is NOT required by --calibrate, only by --validate and by text decisions that ask for it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import styleprint  # noqa: E402
import postgate  # noqa: E402

AXES = list(styleprint.FEATURE_NAMES)

# --- descriptor bands, each sitting in a measured gap (see module docstring) -----
PAPER_PLATE = 0.15
LIGHT_V = 0.55
SAT_HIGH = 0.40
SAT_LOW = 0.20
BIMO_STRONG = 3.60
EDGE_FINE = 0.80
SOFT_MUCH = 0.20
BILAT_STRONG = 0.70
BILAT_SOME = 0.25

KNOWN_LIMITS = [
    "descriptor bands are calibrated on one 48-image corpus; the numbers are "
    "portable, the bands may not be. --calibrate prints the buyer's own bands.",
    "naming a string does not make the renderer spell it: a real word read at "
    "confidence 91.9 sits inside a 89-93 gibberish band, and an ornate display "
    "face read the same real word at 67.9, below the gibberish.",
    "the series gate catches structural damage but not statistical damage. The "
    "blur figure is measured twice and differs by base image: this harness puts "
    "its own sigma-14 victim at 8.50 against a tolerance of 7.08 (outside), and "
    "postgate's victim lands at 9.50 standalone / 10.00 when re-encoded as JPEG. "
    "A pure-noise image is NOT caught: postgate measures it at 4.10 in-harness "
    "/ 5.73 standalone, both inside the tolerance, because a noisy image's axes "
    "genuinely resemble one of the corpus' busiest images.",
    "one reference image is a weak description of a series; below four references "
    "the kNN tolerance is degenerate and --validate says so rather than printing "
    "a confident number.",
    "every axis here is photometric. none of them reads meaning.",
]

# ----------------------------------------------------------------------------- io
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def list_refs(ref: Path) -> list[Path]:
    if not ref.is_dir():
        raise SystemExit(f"--ref is not a directory: {ref}")
    found = sorted(p for p in ref.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not found:
        raise SystemExit(f"--ref holds no images: {ref}")
    return found


def measure(refs: list[Path]) -> dict:
    rows = {}
    for p in refs:
        try:
            rows[p.name] = styleprint.features(p)
        except Exception as exc:  # one bad file must not kill the batch
            print(f"  skip {p.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if not rows:
        raise SystemExit("every reference image failed to measure")
    return rows


# ------------------------------------------------------------------- series space
def zspace(rows: dict) -> tuple[dict, dict]:
    """z-score every axis across the corpus. Returns ({name: zvector}, stats)."""
    names = sorted(rows)
    mu, sd = {}, {}
    for a in AXES:
        vals = [rows[n][a] for n in names]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        mu[a], sd[a] = m, max(1e-9, var**0.5)
    z = {n: [(rows[n][a] - mu[a]) / sd[a] for a in AXES] for n in names}
    return z, {"mean": mu, "sd": sd}


def _dist(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def project(feat: dict, stats: dict) -> list[float]:
    """Place one feature dict into an existing corpus's z-space.

    Used on images the corpus has never seen -- a blur control, or a buyer's render --
    so the distance is measured against the same mean and spread the tolerance came
    from. A separately-fitted normalisation would silently rescale the verdict.
    """
    return [(feat[a] - stats["mean"][a]) / stats["sd"][a] for a in AXES]


def loo_scale(z: dict) -> dict:
    """Leave-one-out nearest-neighbour distance for every reference.

    This is the same quantity postgate uses as its tolerance, and the same principle:
    a series tolerance should be as loose as the series' own most isolated member.
    """
    names = sorted(z)
    if len(names) < 2:
        return {"n": len(names), "usable": False,
                "reason": "one reference gives no internal gap to measure"}
    ds = []
    for n in names:
        others = [m for m in names if m != n]
        ds.append(min(_dist(z[n], z[m]) for m in others))
    ds.sort()
    mid = len(ds) // 2
    p50 = ds[mid] if len(ds) % 2 else 0.5 * (ds[mid - 1] + ds[mid])
    return {"n": len(names), "usable": True, "loo_min": round(ds[0], 3),
            "loo_p50": round(p50, 3), "loo_max": round(ds[-1], 3),
            "loo_all": [round(d, 3) for d in ds]}


# ------------------------------------------------------------------ descriptors
def descriptors(f: dict) -> list[str]:
    """Turn one measured feature dict into plain-language claims about the image.

    Every clause is a number that was measured, so a wrong descriptor is falsifiable by
    the printed value next to it.
    """
    out: list[str] = []

    if f["paper_frac"] > PAPER_PLATE:
        out.append(f"a light paper plate as the ground (paper {f['paper_frac']:.2f}, "
                   f"mean value {f['mean_v']:.2f})")
    elif f["mean_v"] < 0.22:
        out.append(f"a near-black ground (mean value {f['mean_v']:.2f}, "
                   f"ink {f['ink_frac']:.2f})")
    else:
        out.append(f"a mid-dark ground (mean value {f['mean_v']:.2f}, "
                   f"ink {f['ink_frac']:.2f})")

    if f["mean_sat"] < SAT_LOW:
        out.append(f"near-greyscale ink (saturation {f['mean_sat']:.2f})")
    elif f["mean_sat"] > SAT_HIGH:
        out.append(f"high-chroma colour throughout (saturation {f['mean_sat']:.2f})")
    else:
        out.append(f"coloured but not neon (saturation {f['mean_sat']:.2f})")

    out.append("two separated value groups, dark against hot or bright against flat "
               f"(separation {f['bimodality']:.2f})"
               if f["bimodality"] > BIMO_STRONG
               else f"one continuous value range, no clean split (separation "
                    f"{f['bimodality']:.2f})")

    if f["edge_density"] > EDGE_FINE:
        out.append(f"dense fine line-work, lots of edges (edge density "
                   f"{f['edge_density']:.2f})")
    else:
        out.append(f"sparse open drawing (edge density {f['edge_density']:.2f})")

    out.append("soft airbrushed shading is a real part of it "
               f"({f['soft_frac']:.2f})" if f["soft_frac"] > SOFT_MUCH
               else "hard-edged, almost no soft gradient "
                    f"({f['soft_frac']:.2f})")

    b = f["bilateral"]
    if b > BILAT_STRONG:
        out.append(f"the layout is bilaterally mirrored about the vertical centre "
                   f"(mirror correlation {b:+.2f})")
    elif b > BILAT_SOME:
        out.append(f"loosely balanced about the centre, not strictly mirrored "
                   f"(mirror correlation {b:+.2f})")
    else:
        out.append(f"deliberately asymmetric about the centre "
                   f"(mirror correlation {b:+.2f})")

    out.append(f"colour is spread across many hues (hue entropy "
               f"{f['hue_entropy']:.2f})" if f["hue_entropy"] > 0.55
               else f"colour is concentrated in few hues (hue entropy "
                    f"{f['hue_entropy']:.2f})")
    return out


def text_rule(t: dict) -> str:
    """Build the lettering instruction. Three states, because there is no fourth.

    'texture' is the default because the measured corpus shows lettering everywhere:
    all 48 images returned OCR tokens, from 4 to 145. The renderer does not do
    'no text'; it does word-shaped marks. Asking for none is not the same as getting
    none, so the default asks for marks that spell nothing.
    """
    if t.get("text") == "none":
        return ("ABSOLUTELY NO text, no letters, no numbers, no type block, no "
                "signature, no watermark, nothing that reads as writing anywhere in "
                "the image including the corners.")
    allow = (t.get("text") or {}).get("allow") if isinstance(t.get("text"), dict) else None
    if allow:
        strings = list(allow)
        one = "exactly this one string" if len(strings) == 1 else "exactly these strings"
        listed = "; ".join(f'"{s}"' for s in strings)
        return (f"Set {one}, spelled character for character and in this order: "
                f"{listed}. Nothing else in the image may spell anything. Any other "
                f"text-shaped mark must read as writing that spells nothing -- "
                f"garbled letterforms, numerals, tick labels, a boxed credit, a "
                f"signature. Never invent a word that is not listed above, and never "
                f"misspell a listed one.")
    return ("Lettering appears as texture only: garbled letterforms, tick labels, "
            "numeral strings, a boxed credit, a signature in a corner. None of it "
            "may spell a real word. No legible brand, no real logo, no real word "
            "anywhere in the image.")


# ------------------------------------------------------------------------ prompts
def keep_clause(feat: dict, refname: str) -> str:
    return (
        f"UPLOAD {refname} FIRST and use it as the style reference. Keep from it, "
        f"exactly and without variation:\n"
        + "\n".join(f"  - {d}" for d in descriptors(feat))
        + "\n\nMeasured on that reference, so you can argue with it: "
        + ", ".join(f"{a} {feat[a]:.3f}" for a in AXES)
        + "\nChange only the subject and the arrangement given above. Keep the ground, the "
          "palette temperature, the line quality, the density, the print process and "
          "the degree of symmetry identical."
    )


def series_clause(rows: dict) -> str:
    """Descriptive per-axis range across the whole reference set.

    NOT a gate. A min/max envelope was tried as the gate and refuted -- a blurred image
    stayed inside the box on all twelve axes. This is prose to keep the new image inside
    the same visual neighbourhood; the real check is postgate's kNN distance.
    """
    lines = []
    for a in AXES:
        vals = [r[a] for r in rows.values()]
        lines.append(f"  {a:18s} {min(vals):.3f} to {max(vals):.3f}")
    return (
        "Hold every one of these inside the range the reference set spans. This is a "
        "description of the neighbourhood, not a test -- a blurred image can sit "
        "inside a min/max box, which is why postgate measures nearest-neighbour "
        "distance instead.\n" + "\n".join(lines)
    )


def build_prompt(idx: int, target: dict, feat: dict, refname: str,
                 rows: dict) -> str:
    name = str(target.get("name") or f"{idx:02d}")
    subject = (target.get("subject") or "").strip()
    if not subject:
        raise SystemExit(f"target {name}: 'subject' is required")
    geometry = (target.get("geometry") or "").strip()
    counts = target.get("counts")

    sec = 2  # 1 DELIVERABLE and 2 THE ONE IDEA are always present
    parts = [
        f"ISSUE {name}",
        "",
        "## 1. DELIVERABLE",
        "One cover image, 2:3 portrait, for a comic book cover. One image only, no "
        "contact sheet, no variants, no grid.",
        "",
        "## 2. THE ONE IDEA",
        f"WHAT THE IMAGE SHOWS: {subject}",
    ]
    if geometry:
        sec += 1
        parts += ["", f"## {sec}. HOW IT IS ARRANGED", geometry]

    if counts:
        motif = str(target.get("motif") or "rosette")
        n = int(counts)
        sec += 1
        parts += [
            "",
            f"## {sec}. THE FOLD",
            f"{n}-fold symmetry, and the number is deliberate, not incidental. "
            f"Diffusion models cannot draw exact rotational symmetry, so a second "
            f"image is supplied: generate `rosette-{n}.png` with the rosette "
            f"module, which ships as its own repo `amph-rosette` rather than "
            f"inside this one -- "
            f"`python3 rosette.py --n {n} --motif {motif} "
            f"--out rosette-{n}.png` -- upload THAT alongside the style "
            f"reference, and match its rotational rhythm exactly. Hold the count "
            f"precisely: {n}, not approximately.",
        ]

    sec += 1
    parts += [
        "",
        f"## {sec}. KEEP FROM THE REFERENCE",
        keep_clause(feat, refname),
    ]

    sec += 1
    parts += ["", f"## {sec}. THE SERIES NEIGHBOURHOOD", series_clause(rows)]

    sec += 1
    parts += ["", f"## {sec}. TEXT", text_rule(target)]

    sec += 1
    parts += [
        "",
        f"## {sec}. AFTER THE IMAGE",
        "1. Look at all four corners. A renderer will invent a signature or a "
        "developer handle there. Anything word-shaped in a corner is a reject, not "
        "something to retouch.",
        "2. OCR the result. The gate cannot tell a real word from word-shaped "
        "gibberish -- on the reference corpus a real word read at confidence 91.9 sat "
        "inside a 89-93 gibberish band. If a string was named above and OCR does not "
        "return it character for character, regenerate.",
        "3. Run `python3 tools/postgate.py IMAGE`. Expect the series gate near 0 "
        "z-units from the nearest reference. A blurred or structurally broken render "
        "lands far outside the tolerance.",
        "4. Judge it at 200 px wide. If it mushes, cut the SUBJECT, never the ink "
        "density -- thinning the line-work destroys the style, removing detail from "
        "the subject does not.",
    ]
    return "\n".join(parts)


def resolve_reference(target: dict, refs: list[Path]) -> Path:
    want = target.get("reference")
    if want:
        for p in refs:
            if p.name == want:
                return p
        # allow a unique suffix so a truncated filename still resolves
        hits = [p for p in refs if p.name.endswith(str(want))]
        if len(hits) == 1:
            return hits[0]
        raise SystemExit(f"reference {want!r} not found (matched {len(hits)} by suffix)")
    idx = target.get("index")
    if isinstance(idx, int):
        if not 1 <= idx <= len(refs):
            raise SystemExit(f"index {idx} out of range 1..{len(refs)}")
        return refs[idx - 1]
    raise SystemExit("each target needs 'reference' (filename) or 'index' (1-based)")


# ------------------------------------------------------------------------ outputs
def calibrate(rows: dict, z: dict, scale: dict) -> dict:
    """The buyer's own postgate thresholds, derived from their own corpus.

    postgate's shipped constants were measured on my 48 images. Transferring them to
    someone else's art would be the same mistake the retirement of the hard-coded A-E
    registers was. So: recompute, and print the band the decision sits in.
    """
    out = {
        "n_reference_images": len(rows),
        "axes": list(AXES),
        "series_tolerance_z": scale.get("loo_max"),
        "series_scale": scale,
        "styleprint_constants": styleprint.payload({})["constants"],
        "rekey_bands": {
            "PAPER_PLATE": PAPER_PLATE, "LIGHT_V": LIGHT_V, "SAT_HIGH": SAT_HIGH,
            "SAT_LOW": SAT_LOW, "BIMO_STRONG": BIMO_STRONG, "EDGE_FINE": EDGE_FINE,
            "SOFT_MUCH": SOFT_MUCH, "BILAT_STRONG": BILAT_STRONG,
            "BILAT_SOME": BILAT_SOME,
        },
        "measured_bands": {},
        "frame": {"target_ratio": postgate.FRAME_TARGET, "tol": postgate.FRAME_TOL,
                  "note": "both 1:1 and 2:3 are legal: generate then extend"},
    }
    for a in AXES:
        vals = sorted(rows[n][a] for n in rows)

        def q(p):
            return vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))]
        out["measured_bands"][a] = {"min": round(vals[0], 3), "q25": round(q(.25), 3),
                                    "med": round(q(.5), 3), "q75": round(q(.75), 3),
                                    "max": round(vals[-1], 3)}
    return out


def print_calibrate(rows: dict, z: dict, scale: dict) -> None:
    c = calibrate(rows, z, scale)
    print(f"CALIBRATE  {c['n_reference_images']} reference images")
    if scale.get("usable"):
        print(f"  series tolerance   {c['series_tolerance_z']} z-units "
              f"(loo min {scale['loo_min']} / p50 {scale['loo_p50']} / max "
              f"{scale['loo_max']})")
    else:
        print(f"  series tolerance   UNAVAILABLE -- {scale.get('reason')}")
    print(f"  frame               {c['frame']['note']}")
    print("\n  measured bands      min    q25    med    q75    max")
    for a in AXES:
        b = c["measured_bands"][a]
        print(f"    {a:17s} {b['min']:6.3f} {b['q25']:6.3f} {b['med']:6.3f} "
              f"{b['q75']:6.3f} {b['max']:6.3f}")
    print("\n  rekey descriptor bands (calibrated on a 48-image corpus, not this one)")
    for k, v in c["rekey_bands"].items():
        print(f"    {k:14s} {v}")
    print("\n  postgate text/corner thresholds stay at their shipped values until "
          "--validate\n  measures this corpus; OCR confidence does not transfer between "
          "artists.")


# ---------------------------------------------------------------------- validate
def validate(rows: dict, z: dict, stats: dict, scale: dict,
             refs: list[Path]) -> int:
    """Two tests. Can the series gate fire at all, and does a negative control fire it?

    A gate nobody has watched fail is not a gate. This is the same discipline as
    validate_styleprint.py, applied to the buyer's corpus.
    """
    print(f"TEST 1 spread   can the series gate fire on this corpus?")
    ok1 = False
    if not scale.get("usable"):
        print(f"    n/a   {scale.get('reason')}; tolerance is undefined, not zero")
    else:
        tol = scale["loo_max"]
        print(f"    ok    tolerance {tol} z-units from {scale['n']} images "
              f"(loo min {scale['loo_min']} / p50 {scale['loo_p50']})")
        if scale["n"] < 4:
            print(f"    WARN  {scale['n']} images: below four the tolerance is a single "
                  f"outlier, not a distribution")
        else:
            ok1 = True

    print("\nTEST 2 negative control   a structurally destroyed image must land outside")
    ok2 = False
    if scale.get("usable") and scale["n"] >= 2:
        from PIL import Image, ImageFilter
        victim = refs[0]
        others = [m for m in z if m != victim.name]
        base = min(_dist(z[victim.name], z[m]) for m in others)
        tmp = HERE.parent / "work" / "_rekey_blur.jpg"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(victim) as im:
            im.convert("RGB").filter(ImageFilter.GaussianBlur(14.0)).save(
                tmp, quality=92)
        try:
            bad = styleprint.features(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        far = min(_dist(project(bad, stats), z[m]) for m in others)
        verdict = "ok   " if far > scale["loo_max"] else "FAIL "
        print(f"    {verdict} {victim.name[:44]} sits {base:.2f} z-units from its "
              f"nearest neighbour; a sigma-14 blur of the same image lands "
              f"{far:.2f} against a tolerance of {scale['loo_max']}")
        ok2 = far > scale["loo_max"]
    else:
        print("    n/a   need at least two reference images")

    print("\nKNOWN LIMITS")
    for i, lim in enumerate(KNOWN_LIMITS, 1):
        print(f"  {i}. {lim}")

    good = ok1 and ok2
    print(f"\n{'PASS' if good else 'NOT USABLE YET'}  rekey")
    return 0 if good else 1


# --------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="turn a buyer's own reference images into their own cover prompts")
    ap.add_argument("--ref", type=Path, required=True,
                    help="directory of the buyer's reference images")
    ap.add_argument("--targets", type=Path,
                    help="JSON list of targets (subject, reference, geometry, counts, text)")
    ap.add_argument("--out", type=Path, help="directory for NN.md + ALL.txt")
    ap.add_argument("--print", dest="show", type=int, metavar="N",
                    help="print target N's prompt to stdout and exit")
    ap.add_argument("--calibrate", action="store_true",
                    help="report the thresholds derived from this corpus")
    ap.add_argument("--calibrate-json", type=Path, help="write the calibration as JSON")
    ap.add_argument("--validate", action="store_true",
                    help="negative controls on this corpus")
    ap.add_argument("--known-limits", action="store_true")
    args = ap.parse_args(argv)

    if args.known_limits:
        for i, lim in enumerate(KNOWN_LIMITS, 1):
            print(f"{i}. {lim}")
        return 0

    refs = list_refs(args.ref)
    rows = measure(refs)
    z, stats = zspace(rows)
    scale = loo_scale(z)

    if args.calibrate:
        print_calibrate(rows, z, scale)
    if args.calibrate_json:
        args.calibrate_json.parent.mkdir(parents=True, exist_ok=True)
        args.calibrate_json.write_text(
            json.dumps(calibrate(rows, z, scale), indent=2) + "\n")

    if args.validate:
        rc = validate(rows, z, stats, scale, refs)
        if not (args.targets or args.show is not None or args.out):
            return rc

    if not (args.targets or args.show is not None or args.out
            or args.calibrate or args.calibrate_json or args.validate):
        ap.error("give --targets, --out, --print, --calibrate or --validate")

    if args.targets:
        targets = json.loads(args.targets.read_text())
        if isinstance(targets, dict):
            targets = targets.get("targets", [])
        if not isinstance(targets, list) or not targets:
            raise SystemExit("--targets must be a non-empty JSON list")

        built = []
        for i, t in enumerate(targets, 1):
            ref = resolve_reference(t, refs)
            if ref.name not in rows:
                raise SystemExit(f"reference {ref.name} failed to measure")
            built.append(build_prompt(i, t, rows[ref.name], ref.name, rows))

        if args.show is not None:
            if not 1 <= args.show <= len(built):
                raise SystemExit(f"--print must be 1..{len(built)}")
            print(built[args.show - 1])
            return 0

        bodies = []
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            for i, (t, body) in enumerate(zip(targets, built), 1):
                name = str(t.get("name") or f"{i:02d}")
                (args.out / f"{name}.md").write_text(body + "\n")
                bodies.append(f"===== {name} =====\n{body}")
            (args.out / "ALL.txt").write_text("\n\n".join(bodies) + "\n")
            (args.out / "README.md").write_text(
                "# rekeyed prompts\n\n"
                f"Generated by `tools/rekey.py` from {len(rows)} reference image(s) in "
                f"`{args.ref}` for {len(built)} target(s).\n\n"
                "Each `NN.md` is one paste-whole block. Upload the reference image named "
                "in section 5, paste the rest, then run the four checks in section 8.\n\n"
                "The per-axis ranges in section 6 are descriptive, not a test. The real "
                "check is `postgate.py`'s nearest-neighbour distance.\n")
            print(f"wrote {len(built)} prompts to {args.out}")
        else:
            print("\n\n".join(bodies))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
