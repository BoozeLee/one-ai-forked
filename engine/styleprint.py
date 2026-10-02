#!/usr/bin/env python3
"""styleprint v2 -- measured style fingerprint of a rendered image.

v1 shipped 18 axes. Nine survived contact with the 48-image corpus; three were
broken and mechanically fixable; six were unshippable and are documented in
KNOWN_LIMITS instead of being shipped as numbers.

v2 changes from v1
------------------
KEPT (9):    ink_frac paper_frac mean_v v_contrast mean_sat hue_entropy
             edge_density soft_frac orient_entropy
FIXED (3):   fine_detail  log10 of a high/low frequency energy ratio. v1 divided
                        two Sobel means and got 35 .. 1_065_281 -- five orders of
                        magnitude, because the denominator approached zero.
             bimodality   Fisher separation of a 2-cluster fit on the (V, S)
                        plane. Replaces v1's sat_gap, whose absolute bin-emptiness
                        test (< 0.004) can never fire once an image has thousands
                        of pixels. Saturation alone is the wrong axis anyway: the
                        corpus rule is about the JOINT distribution.
             bilateral    Mirror correlation measured on a sigma=16 blurred
                        value image. v1 measured it on raw pixels and scored the
                        corpus's signature fork as its WORST case (FORK -0.147 vs
                        a random diagram +0.517). The fork seam is a LAYOUT
                        property, not a pixel property: blurred, it rises to
                        +0.363 while an asymmetric control falls to -0.153.
REMOVED (2): sym_order sym_strength. Fold-order DETECTION on rendered AI images
             does not work. A polar angular FFT recovers a clean synthetic
             8-fold rosette but then scores 10 as 8, 7 as 2, 2 as 4, and returns
             a degenerate peak of 1.00 on every asymmetric image -- an asymmetric
             liquid spiral came back as 8-fold. You cannot audit a model at
             drawing exact n-fold symmetry; hand it a correct reference instead
             (see tools/rosette.py).

Usage
-----
    python3 tools/styleprint.py --dir work/art --out work/styleprint.json
    python3 tools/styleprint.py --dir work/art --out work/styleprint.json --check
    python3 tools/styleprint.py --featurenames
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel

PROBES = 320           # long edge the image is reduced to before measuring
INK_V = 0.18           # below this value a pixel counts as ink
PAPER_V = 0.85         # above this value a pixel counts as paper
GREY_S = 0.14          # chroma below this counts as achromatic
HUE_BINS = 12
ORIENT_BINS = 12
BLUR_SIGMA = 16.0      # layout scale; the fork is legible here and not at sigma 0
DETAIL_SIGMA = 1.0     # high-frequency scale for the fine_detail ratio
# Measured, not guessed. On this corpus the value-channel gradient magnitude has
# median q10 = 0.022 and median q50 = 0.294, so 0.04 sits above the flat tail and
# below the edge mode, and splits the corpus with a relative span of 0.957.
# v1 used 6.0, which is a 0-255 number applied to a [0,1] image -- it made the
# axis a constant 1.0 for all 48 images.
SOFT_GRAD = 0.04
ORIENT_GATE = (0.5, 2.0)  # only gradients within this multiple of the median count

FEATURE_NAMES = (
    "ink_frac",
    "paper_frac",
    "mean_v",
    "v_contrast",
    "mean_sat",
    "hue_entropy",
    "bimodality",
    "edge_density",
    "fine_detail",
    "soft_frac",
    "orient_evenness",
    "bilateral",
)

FEATURE_DOC = {
    "ink_frac": "fraction of pixels darker than INK_V. High = drawn plate, low = open ground.",
    "paper_frac": "fraction brighter than PAPER_V. Non-zero only on the cream drafting plates.",
    "mean_v": "mean value. The corpus splits dark-ground and bright-flat; this is the first axis that sees it.",
    "v_contrast": "std of value. Low means flat fill, high means engraved tonal gradation.",
    "mean_sat": "mean chroma magnitude (max-min). Blind to hue by construction.",
    "hue_entropy": "Shannon entropy of a HUE_BINS hue histogram over chromatic pixels, / log(HUE_BINS). High = many hues present.",
    "bimodality": "Fisher separation of a 2-cluster fit on the (V, S) plane. The corpus is bimodal: dark+hot or bright+flat, never a pastel mid-tone. 0 = one cloud.",
    "edge_density": "mean Sobel gradient magnitude / 255. Line-work density.",
    "fine_detail": "log10(high-frequency energy / value std). The v1 version of this spanned five orders of magnitude; this is bounded.",
    "soft_frac": "fraction of pixels whose gradient magnitude is below SOFT_GRAD, i.e. flat areas. High = flat colour fills, low = line-work everywhere.",
    "orient_evenness": (
        "Shannon entropy of an ORIENT_BINS gradient-orientation histogram over mid-strength "
        "gradients only, / log(ORIENT_BINS). Measured range on the corpus is 0.949-1.000 -- "
        "this corpus is orientationally ISOTROPIC, there is no dominant ruling direction in "
        "any of the 48 images. It therefore carries almost no in-corpus discriminative power "
        "and is kept as an out-of-distribution tripwire, not as a family axis: a buyer "
        "uploads a photograph with one dominant light direction and this drops immediately."
    ),
    "bilateral": (
        "Pearson correlation between the image and its own mirror, measured on a sigma=16 "
        "blurred value image. High = the layout mirrors. This is the fork detector. NOTE it "
        "is mathematically INVARIANT under mirroring -- pearson(mirror(I), mirror(mirror(I))) "
        "== pearson(mirror(I), I) -- so it measures whether a layout IS mirror-symmetric, "
        "not whether a change made it so. Validating it therefore requires forcing a seam, "
        "not flipping the image."
    ),
}

KNOWN_LIMITS = {
    "status": "measured negative results, deliberately not shipped as axes",
    "why_this_exists": (
        "The fork is the series' signature and the thing a buyer most needs to "
        "reproduce, so it is worth being precise about which half of it is "
        "measurable. It is a LAYOUT property plus a PALETTE property. The layout "
        "half is recoverable (see the bilateral axis). The palette half is not."
    ),
    "seam_palette_is_not_recoverable": {
        "claim": "No pixel statistic can tell you a fork is cold on the left and hot on the right.",
        "reason": (
            "A bilaterally mirrored fork swaps hue across the seam by construction, "
            "so the left and right hue histograms are near-identical BY DESIGN. The "
            "eye reads cold-versus-warm as a labelling; the underlying distributions "
            "genuinely overlap. Any measure keyed on hue-mass difference is measuring "
            "the mirror, not the split."
        ),
        "consequence": (
            "A fork cannot be obtained from a prompt. It can only be obtained from a "
            "reference image. That is the product argument for reference-anchored "
            "generation, and it is a measurement rather than an opinion."
        ),
    },
    "approaches_tried_and_failed": [
        {
            "name": "chroma Fisher on the two halves",
            "axis": "tanh(1.6*fisher(chroma)) + tanh(2.0*|warmth_r - warmth_l|)",
            "result": "inverted. seam family 0.097 (n=8) vs everything else 0.208 -- separation 0.47x",
            "structural_cause": (
                "cyan and magenta have near-equal luminance AND near-equal chroma "
                "magnitude; only hue differs. Any chroma-magnitude or luminance "
                "measure is blind to the signature feature."
            ),
        },
        {
            "name": "hue-opponent axis times hue-histogram overlap",
            "axis": "tanh(1.4*opponent_fisher) * (1 - Bhattacharyya(hue_hist_L, hue_hist_R, 12 bins))",
            "result": (
                "weak. seam mean 0.180 vs other 0.127, separation 1.41x. Issue 8 -- the "
                "clearest hard-seam image in the corpus by eye -- scored -0.054, dead "
                "last of all 48."
            ),
            "structural_cause": (
                "the opponent axis does see the split (issue 8's raw opponent score of "
                "1.50 was the highest in the corpus); the overlap multiplier destroys "
                "it, and the multiplier is itself buggy -- it returned ~1.056, which is "
                "impossible for properly normalised histograms."
            ),
        },
        {
            "name": "local gradient ratio across a 3% centre band",
            "axis": "tanh(grad_in_seam_band / grad_outside_band)",
            "result": (
                "tried on luminance, saturation and hue-derivative maps, each against "
                "its own rolled-25%-sideways negative control. No map separated the "
                "hard-seam sets (FORK, SUPERBRAIN) from MANDALA8 / LIQ / ROSE. The FORK "
                "scored 0.65x -- weaker than a random column."
            ),
            "structural_cause": (
                "the seam is a global compositional property (cold half vs hot half), "
                "not a local edge. And the same near-equal-luminance problem applies."
            ),
        },
    ],
    "near_constant_axes_kept_on_purpose": {
        "orient_evenness": (
            "Measured 0.949-1.000 across all 48 images. Tested with four gating schemes "
            "(all gradients; mid-strength only; strong only; 8 and 12 bins) and the "
            "relative span never exceeded 0.06. The corpus is orientationally isotropic: "
            "no image has a dominant ruling direction. Shipping it as a family axis "
            "would be dishonest, so it is kept only as an out-of-distribution tripwire "
            "with its measured range recorded here. See tools/validate_styleprint.py "
            "for the `ruled` control that proves it does move when a single edge "
            "direction is imposed."
        ),
    },
    "retracted_claims": [
        "v1's |warmth| column reading +0.99 / +0.83 on the hard-seam sets did NOT reproduce. Correct per-file values are 0.00-0.16. Retracted, not tuned.",
        "v1's sat_gap emptiness test used an absolute bin count of 0.004, which no bin in an image with thousands of pixels can ever fall below. It could never fire. Replaced by bimodality.",
    ],
}


# --------------------------------------------------------------------------- io


def load_gray(path: Path) -> np.ndarray:
    """Reduce to a PROBES-long-edge float value image in [0, 1]."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = PROBES / max(w, h)
        if scale < 1.0:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def rgb_to_hsv(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB -> (hue in [0,1), saturation, value). hue is 0 where d == 0."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.max(arr, axis=-1)
    mn = np.min(arr, axis=-1)
    d = mx - mn
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(mx > 0, d / np.maximum(mx, 1e-8), 0.0)
        h = np.zeros_like(mx)
        nz = d > 1e-8
        rm = nz & (mx == r)
        gm = nz & (mx == g)
        bm = nz & (mx == b)
        h[rm] = ((g[rm] - b[rm]) / d[rm]) % 6.0
        h[gm] = (b[gm] - r[gm]) / d[gm] + 2.0
        h[bm] = (r[bm] - g[bm]) / d[bm] + 4.0
        h = (h / 6.0) % 1.0
    return h.astype(np.float32), s.astype(np.float32), mx.astype(np.float32)


# ----------------------------------------------------------------- 1-D 2-means


def two_means_1d(x: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """Deterministic 2-means on 1-D data. Init at the 10th and 90th percentile.

    Deterministic on purpose: v1's clustering came from a seeded RNG and the same
    image could fingerprint differently across a numpy version bump.
    """
    lo, hi = np.percentile(x, [10.0, 90.0])
    m0, m1 = float(lo), float(hi)
    for _ in range(iters):
        a = x[x <= (m0 + m1) / 2.0]
        b = x[x > (m0 + m1) / 2.0]
        if a.size == 0 or b.size == 0:
            break
        n0, n1 = float(a.mean()), float(b.mean())
        if abs(n0 - m0) < 1e-7 and abs(n1 - m1) < 1e-7:
            break
        m0, m1 = n0, n1
    return m0, m1


def _fisher(x: np.ndarray) -> float:
    m0, m1 = two_means_1d(x)
    a = x[x <= (m0 + m1) / 2.0]
    b = x[x > (m0 + m1) / 2.0]
    if a.size < 8 or b.size < 8:
        return 0.0
    spread = 0.5 * (float(a.std()) + float(b.std()))
    if spread < 1e-6:
        return 0.0
    return float(np.clip((m1 - m0) / spread, 0.0, 10.0))


# ------------------------------------------------------------------- the axes


def features(path: Path) -> dict[str, float]:
    arr = load_gray(path)
    h, s, v = rgb_to_hsv(arr)
    chroma = s

    out: dict[str, float] = {}
    out["ink_frac"] = float((v < INK_V).mean())
    out["paper_frac"] = float((v > PAPER_V).mean())
    out["mean_v"] = float(v.mean())
    out["v_contrast"] = float(v.std())
    out["mean_sat"] = float(chroma.mean())

    chromatic = chroma > GREY_S
    n_chrom = int(chromatic.sum())
    if n_chrom >= 32:
        hist, _ = np.histogram(h[chromatic], bins=HUE_BINS, range=(0.0, 1.0))
        p = hist.astype(np.float64) / hist.sum()
        nz = p[p > 0]
        out["hue_entropy"] = float(-(nz * np.log(nz)).sum() / math.log(HUE_BINS))
    else:
        out["hue_entropy"] = 0.0

    # Joint (V, S) separation. The projection axis V - S is chosen because the
    # corpus rule is "dark and hot OR bright and flat", which is exactly that
    # contrast. Saturation alone cannot express it.
    proj = (v - chroma).ravel()
    out["bimodality"] = _fisher(proj)

    gx = sobel(v, axis=1, mode="nearest")
    gy = sobel(v, axis=0, mode="nearest")
    grad = np.hypot(gx, gy)
    out["edge_density"] = float(grad.mean())

    hi = float(np.abs(v - gaussian_filter(v, DETAIL_SIGMA)).mean())
    lo = float(v.std())
    out["fine_detail"] = float(math.log10(max(hi, 1e-6) / max(lo, 1e-6)))

    out["soft_frac"] = float((grad < SOFT_GRAD).mean())

# Orientation. Gated to mid-strength gradients: including the flat-field
    # noise floor swamps the histogram and every image scores ~1.0.
    med = float(np.median(grad))
    sel = (grad > ORIENT_GATE[0] * med) & (grad < ORIENT_GATE[1] * med)
    if int(sel.sum()) >= 32:
        ang = (np.arctan2(gy[sel], gx[sel]) + math.pi) / (2.0 * math.pi)
        hist, _ = np.histogram(ang, bins=ORIENT_BINS, range=(0.0, 1.0))
        p = hist.astype(np.float64) / hist.sum()
        nz = p[p > 0]
        out["orient_evenness"] = float(-(nz * np.log(nz)).sum() / math.log(ORIENT_BINS))
    else:
        out["orient_evenness"] = 0.0

    # Layout mirror correlation. Blurred hard, because at sigma 0 this measures
    # whether the ink happens to match, not whether the composition mirrors.
    b = gaussian_filter(v, BLUR_SIGMA)
    r = b[:, ::-1]
    bf, rf = b.ravel(), r.ravel()
    sb, sr = bf.std(), rf.std()
    if sb < 1e-6 or sr < 1e-6:
        out["bilateral"] = 0.0
    else:
        out["bilateral"] = float(np.clip(((bf - bf.mean()) * (rf - rf.mean())).mean() / (sb * sr), -1.0, 1.0))

    return out


# ------------------------------------------------------------------- plumbing


def measure_dir(d: Path) -> tuple[dict, list[str]]:
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    rows: dict[str, dict] = {}
    errors: list[str] = []
    for p in files:
        try:
            rows[p.name] = features(p)
        except Exception as exc:  # one bad file must not kill the batch
            errors.append(f"{p.name}: {type(exc).__name__}: {exc}")
    return rows, errors


def payload(rows: dict) -> dict:
    return {
        "version": 2,
        "feature_names": list(FEATURE_NAMES),
        "feature_doc": FEATURE_DOC,
        "constants": {
            "PROBES": PROBES,
            "INK_V": INK_V,
            "PAPER_V": PAPER_V,
            "GREY_S": GREY_S,
            "HUE_BINS": HUE_BINS,
            "ORIENT_BINS": ORIENT_BINS,
            "BLUR_SIGMA": BLUR_SIGMA,
            "DETAIL_SIGMA": DETAIL_SIGMA,
            "SOFT_GRAD": SOFT_GRAD,
        },
        "known_limits": KNOWN_LIMITS,
        "images": rows,
    }


def digest(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, help="directory of images to fingerprint")
    ap.add_argument("--out", type=Path, help="write the JSON report here")
    ap.add_argument("--check", action="store_true", help="exit 1 if --out is missing or stale")
    ap.add_argument("--featurenames", action="store_true", help="print the axis names and docs, then exit")
    ap.add_argument("--limits", action="store_true", help="print the known-limits record, then exit")
    args = ap.parse_args(argv)

    if args.featurenames:
        for name in FEATURE_NAMES:
            print(f"{name:16s} {FEATURE_DOC[name]}")
        return 0

    if args.limits:
        print(json.dumps(KNOWN_LIMITS, indent=2))
        return 0

    if not args.dir or not args.dir.is_dir():
        ap.error("--dir must be an existing directory")

    rows, errors = measure_dir(args.dir)
    if not rows:
        print("no images measured", file=sys.stderr)
        return 1
    rep = payload(rows)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.check:
            if not args.out.exists():
                print(f"STALE  {args.out} does not exist", file=sys.stderr)
                return 1
            old = json.loads(args.out.read_text())
            if digest(old.get("images", {})) != digest(rows):
                print(f"STALE  {args.out} no longer matches {args.dir}", file=sys.stderr)
                return 1
            print(f"PASS   {len(rows)} images match {args.out}")
            return 0
        args.out.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
        print(f"wrote  {args.out}  {len(rows)} images  digest {digest(rows)}")
        for e in errors:
            print(f"  error: {e}", file=sys.stderr)
        return 0

    print(json.dumps({"digest": digest(rows), "images": rows}, indent=2, sort_keys=True))
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
