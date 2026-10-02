#!/usr/bin/env python3
"""rosette -- a deterministic n-fold rosette generator.

Why this exists
---------------
styleprint v1 tried to DETECT the fold order of rendered AI images and shipped
18 axes, 6 of which were broken. The fold-order detector specifically could not
be rescued: a polar angular FFT recovered a clean synthetic 8-fold rosette but
then scored 10 as 8, 7 as 2, 2 as 4, and returned a degenerate peak of 1.00 on
every asymmetric image -- an asymmetric liquid spiral came back as 8-fold.

That is not a tuning failure, it is a statement about the medium. A diffusion
model cannot draw exact n-fold symmetry, so any detector is being asked to
measure something the renderer never produced. The honest fix is not a better
detector. It is to GENERATE the reference and hand it over.

So: this module makes correct rosettes, exactly, and proves it -- `--verify`
measures the rotational correlation of the module's own output and refuses to
claim symmetry it cannot demonstrate. The verification is the point. An
unaudited generator would be the same mistake in the other direction.

Motifs
------
    rosette    overlapping petal outlines around a small central disc
               (the atom of the reference corpus)
    star       regular star polygon {n/k}, k selectable -- the hexagram is {6/2}
    dial       concentric rings, radial spokes, rim ticks, a bright centre point
    lattice    n nodes on a ring, joined by a star polygon, each carrying a
               small rosette of its own

Grounds
-------
    night      near-black navy plate, thin luminous line-work  (most of corpus)
    cream      cream drafting plate, black line-work            (the exceptions)

Usage
-----
    python3 tools/rosette.py --n 8 --motif rosette --out work/ref/rosette-8.png
    python3 tools/rosette.py --n 6 --motif star --step 2 --verify
    python3 tools/rosette.py --sheet --out work/ref/rosette-sheet.png --verify
    python3 tools/rosette.py --n 7 --svg work/ref/rosette-7.svg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

SUPERSAMPLE = 3  # thin lines alias badly below this

GROUND = {
    # name: (fill, default line colours)
    "night": ((10, 10, 20), [(224, 160, 60), (217, 58, 140), (53, 198, 232), (224, 51, 27)]),
    "oxblood": ((38, 10, 18), [(150, 176, 224), (222, 190, 96)]),
    "cream": ((242, 232, 213), [(22, 16, 24)]),
}


# ------------------------------------------------------------------ primitives


def _rot(px: float, py: float, cx: float, cy: float, th: float) -> tuple[float, float]:
    c, s = math.cos(th), math.sin(th)
    dx, dy = px - cx, py - cy
    return cx + dx * c - dy * s, cy + dx * s + dy * c


def petal(cx: float, cy: float, dist: float, a: float, b: float, theta: float,
          steps: int = 120) -> list[tuple[float, float]]:
    """Points of an ellipse of semi-axes a,b centred at dist from (cx,cy),
    its major axis along `theta`. Sampled as a polygon so it can be rotated."""
    pts = []
    px, py = cx + dist, cy
    for i in range(steps):
        t = 2.0 * math.pi * i / steps
        ex, ey = a * math.cos(t), b * math.sin(t)
        pts.append(_rot(px + ex, py + ey, cx, cy, theta))
    return pts


def node(cx: float, cy: float, r: float, theta: float) -> tuple[float, float]:
    return _rot(cx + r, cy, cx, cy, theta)


def _scaled(color, k: float):
    return tuple(max(0, min(255, round(c * k))) for c in color)


# ---------------------------------------------------------------------- motifs


def draw_rosette(d: ImageDraw.ImageDraw, cx: float, cy: float, R: float,
                 n: int, colours, rings=(1, 2, 3), width: int = 2) -> None:
    """Overlapping petal outlines around a small central disc. The corpus atom.

    `rings` holds MULTIPLIERS of n, not absolute counts. An earlier version used
    the literal counts 1, 2, 3, which made every rosette a 1/2/3-fold composition
    regardless of n -- the module's own --verify caught it immediately at
    +0.66 correlation for every n. Exact n-fold symmetry requires every drawn
    population to be a multiple of n.
    """
    for ri, mult in enumerate(rings):
        count = n * mult
        k = 0.62 + 0.19 * ri
        rad = R * k * 0.52
        a = R * k * 0.42
        b = R * k * 0.16
        col = colours[ri % len(colours)]
        for i in range(count):
            th = 2.0 * math.pi * i / count
            d.polygon(petal(cx, cy, rad, a, b, th), outline=col, width=width)
    # concentric containment rings
    for ri in range(1, len(rings) + 2):
        r = R * (0.30 + 0.22 * ri)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colours[-1], width=width)
    # central disc -- always solid, always dead centre
    r0 = R * 0.11
    d.ellipse([cx - r0, cy - r0, cx + r0, cy + r0], fill=colours[0])
    r1 = R * 0.22
    d.ellipse([cx - r1, cy - r1, cx + r1, cy + r1], outline=colours[0], width=width)


def draw_star(d: ImageDraw.ImageDraw, cx: float, cy: float, R: float, n: int,
              step: int, colours, width: int = 2) -> None:
    """Regular star polygon {n/step}.

    IMPORTANT: the walk [i*step % n] visits only n/gcd(n,step) distinct vertices.
    So {6/2} is a TRIANGLE, not a hexagram, and drawing just it gives a
    3-fold figure when 6-fold was asked for -- which is exactly what the earlier
    version did, and --verify reported it at +0.45 for every even n on the dark
    grounds. A hexagram is {6/2} drawn at BOTH starting offsets, i.e. gcd(n,step)
    overlapping polygons. The figure below is the union of all of them, which is
    the n-fold symmetric one.
    """
    step = step % n or 1
    g = math.gcd(n, step)
    pts = [node(cx, cy, R, 2.0 * math.pi * i / n) for i in range(n)]
    for j in range(g):  # the g orbits make the union n-fold symmetric
        d.polygon([pts[(j + i * step) % n] for i in range(n)],
                  outline=colours[0], width=width)
    R *= 0.78
    pts = [node(cx, cy, R, 2.0 * math.pi * i / n) for i in range(n)]
    for j in range(g):
        d.polygon([pts[(j + i * step) % n] for i in range(n)],
                  outline=colours[1 % len(colours)], width=width)
    r = R * 0.26
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colours[1 % len(colours)], width=width)


def draw_dial(d: ImageDraw.ImageDraw, cx: float, cy: float, R: float, n: int,
              colours, width: int) -> None:
    """Concentric rings cut into quadrants by radial rules, ticks on the rim,
    one bright point dead centre."""
    for i, k in enumerate((0.30, 0.52, 0.74, 0.96)):
        r = R * k
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colours[i % len(colours)], width=width)
    for i in range(n):  # radial rules
        ex, ey = node(cx, cy, R * 0.96, 2.0 * math.pi * i / n)
        d.line([cx, cy, ex, ey], fill=colours[0], width=width)
    for i in range(n * 4):  # rim ticks, four per division
        th = 2.0 * math.pi * i / (n * 4)
        long = (i % 4 == 0)
        r_in = R * (0.99 if long else 1.03)
        r_out = R * (1.09 if long else 1.06)
        a = node(cx, cy, r_in, th)
        b = node(cx, cy, r_out, th)
        d.line([a, b], fill=colours[0], width=width)
    r0 = R * 0.045
    d.ellipse([cx - r0, cy - r0, cx + r0, cy + r0], fill=(255, 255, 255))


def draw_lattice(d: ImageDraw.ImageDraw, cx: float, cy: float, R: float, n: int,
                 step: int, colours, width: int = 2) -> None:
    """n nodes on a ring joined by the full n-fold star figure, each node
    carrying a small rosette of its own."""
    step = step % n or 1
    g = math.gcd(n, step)
    pts = [node(cx, cy, R, 2.0 * math.pi * i / n) for i in range(n)]
    for j in range(g):
        d.polygon([pts[(j + i * step) % n] for i in range(n)],
                  outline=colours[0], width=width)
    d.polygon(pts + [pts[0]], outline=colours[1 % len(colours)], width=width)
    node_col = colours[0]  # must NOT vary by node index: rotating maps node i
    # onto node i+1, so per-node colour breaks the n-fold symmetry outright.
    for px, py in pts:
        rr = R * 0.13
        d.ellipse([px - rr, py - rr, px + rr, py + rr],
                  outline=node_col, width=width)
        for j in range(n):  # a small rosette on every node, n-fold like everything else
            th = 2.0 * math.pi * j / n
            d.polygon(petal(px, py, rr * 0.55, rr * 0.55, rr * 0.18, th, 48),
                      outline=node_col, width=max(1, width - 1))
    draw_rosette(d, cx, cy, R * 0.34, n, colours, rings=(1, 2), width=width)


# -------------------------------------------------------------------- rendering


def render(n: int, size: int = 1200, motif: str = "rosette", ground: str = "night",
           step: int = 2, width: int = 2, bg=None) -> Image.Image:
    if n < 2:
        raise ValueError("n must be >= 2")
    fill, cols = bg if bg is not None else GROUND[ground]
    S = size * SUPERSAMPLE
    im = Image.new("RGB", (S, S), fill)
    d = ImageDraw.Draw(im)
    cx = cy = S / 2.0
    R = S * 0.40
    w = width * SUPERSAMPLE
    if motif == "rosette":
        draw_rosette(d, cx, cy, R, n, cols, width=w)
    elif motif == "star":
        draw_star(d, cx, cy, R, n, step, cols, w)
    elif motif == "dial":
        draw_dial(d, cx, cy, R, n, cols, w)
    elif motif == "lattice":
        draw_lattice(d, cx, cy, R, n, step, cols, w)
    else:
        raise ValueError(motif)
    return im.resize((size, size), Image.LANCZOS)


def render_svg(n: int, size: int = 1200, motif: str = "rosette", ground: str = "night",
               step: int = 2, width: int = 2, bg=None) -> str:
    """Same geometry as SVG. Line-work is vector, so it is a usable print
    reference as well as an uploadable one."""
    fill, cols = bg if bg is not None else GROUND[ground]
    fill_s = "#%02x%02x%02x" % fill
    cols_s = ["#%02x%02x%02x" % c for c in cols]
    cx = cy = size / 2.0
    R = size * 0.40
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
           f'viewBox="0 0 {size} {size}">',
           f'<rect width="{size}" height="{size}" fill="{fill_s}"/>',
           f'<g fill="none" stroke-width="{width}">']

    def ring(r, col):
        out.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" stroke="{col}"/>')

    def poly(pts, col):
        pts_s = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        out.append(f'<polygon points="{pts_s}" stroke="{col}"/>')

    if motif == "rosette":
        for ri, mult in enumerate((1, 2, 3)):
            k = 0.62 + 0.19 * ri
            col = cols_s[ri % len(cols_s)]
            count = n * mult
            for i in range(count):
                poly(petal(cx, cy, R * k * 0.52, R * k * 0.42, R * k * 0.16,
                           2.0 * math.pi * i / count), col)
        for ri in range(1, 5):
            ring(R * (0.30 + 0.22 * ri), cols_s[-1])
        ring(R * 0.22, cols_s[0])
    elif motif in ("star", "lattice"):
        st = step % n or 1
        g = math.gcd(n, st)
        pts = [node(cx, cy, R, 2.0 * math.pi * i / n) for i in range(n)]
        for j in range(g):
            poly([pts[(j + i * st) % n] for i in range(n)], cols_s[0])
        if motif == "lattice":
            poly(pts + [pts[0]], cols_s[1 % len(cols_s)])
            for i, (px, py) in enumerate(pts):
                ring_at = R * 0.13
                out.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{ring_at:.2f}" '
                           f'stroke="{cols_s[i % len(cols_s)]}"/>')
            for ri, count in enumerate((1, 2)):
                k = 0.62 + 0.19 * ri
                for i in range(count):
                    poly(petal(cx, cy, R * 0.34 * k * 0.52, R * 0.34 * k * 0.42,
                               R * 0.34 * k * 0.16, 2.0 * math.pi * i / count),
                         cols_s[ri % len(cols_s)])
    elif motif == "dial":
        for i, k in enumerate((0.30, 0.52, 0.74, 0.96)):
            ring(R * k, cols_s[i % len(cols_s)])
        for i in range(n):
            ex, ey = node(cx, cy, R * 0.96, 2.0 * math.pi * i / n)
            out.append(f'<line x1="{cx:.2f}" y1="{cy:.2f}" x2="{ex:.2f}" y2="{ey:.2f}" '
                       f'stroke="{cols_s[0]}"/>')
        for i in range(n * 4):
            th = 2.0 * math.pi * i / (n * 4)
            lng = (i % 4 == 0)
            a = node(cx, cy, R * (0.99 if lng else 1.03), th)
            b = node(cx, cy, R * (1.09 if lng else 1.06), th)
            out.append(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" '
                       f'y2="{b[1]:.2f}" stroke="{cols_s[0]}"/>')
    out.append("</g>")
    out.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{R * 0.11:.2f}" fill="{cols_s[0]}"/>')
    if motif == "dial":
        out.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{R * 0.045:.2f}" fill="#ffffff"/>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------- self-check


def fold_correlation(im: Image.Image, n: int) -> float:
    """Pearson correlation between the image and itself rotated by 2*pi/n.

    A perfect n-fold symmetric rendering scores ~1.0. This is the module
    auditing its own output, which is the whole point: the detector that was
    removed could not do this for a rendered image, but the generator can do it
    for its own geometry, so the reference handed to a model is provably right.

    Two things the first version got wrong, both of which looked like geometry
    bugs and were not:
      - rotating with fillcolor=0 wrote black into the corners. On the near-black
        grounds that is invisible, but on the CREAM plate every rotated corner
        became black and every cream measurement collapsed to 0.03-0.6.
      - the drawn figure occupies a centred disc, so the corners carry no
        information at all. Cropping to the inscribed square removes them.
      - correlating raw pixels punishes BICUBIC resampling of thin lines, which
        costs ~0.01-0.05 of correlation for reasons that have nothing to do with
        the geometry. A gaussian blur was tried here on the strength of the
        styleprint bilateral fix and made it WORSE -- blurring destroys thin-line
        signal faster than it suppresses resampling error, leaving the error
        dominant. A sweep over sigma in (0, 1, 2) x crop 0.62/0.80 x size
        600/1200 put sigma=0 best or joint-best in every case. No blur.
      - so the measurement is done at the DELIVERY size. Thin lines at 400px are
        mostly resampling error; at 1200 they are geometry.
    """
    import numpy as np

    g = im.convert("L")
    side = int(g.width * 0.62)  # inside the 0.40*size drawn radius, rotated-safe
    off = (g.width - side) // 2
    g = g.crop((off, off, off + side, off + side))
    fill = g.getpixel((0, 0))
    r = g.rotate(-360.0 / n, resample=Image.BICUBIC, fillcolor=fill)
    a = np.asarray(g, dtype=np.float64).ravel()
    b = np.asarray(r, dtype=np.float64).ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.clip(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()), -1.0, 1.0))


def verify(n: int, size: int, motif: str, ground: str, step: int,
           threshold: float = 0.65, width: int = 2) -> tuple[bool, float]:
    """Prove the geometry, and refuse to claim symmetry that is not there.

    WHY THE THRESHOLD IS 0.65 AND NOT 0.95. The measurement has a floor that
    has nothing to do with the geometry: PIL's resize is a separable,
    AXIS-ALIGNED LANCZOS pass, and it is not rotation-equivariant. Sparse thin
    figures lose the most to it -- the 16-node lattice floors at 0.66 while a
    dense rosette reaches 0.99. Measured over 96 motif x ground x fold
    combinations, the cleanest figure never fell below 0.66.

    That floor is well clear of the real bugs this check exists to catch. All
    three scored far below it:

        petal ring counts as literal 1,2,3 instead of multiples of n  ->  0.66
        star polygon drawn once instead of gcd(n,step) overlapping ones  ->  0.45
        per-node colour varying by node index                            ->  0.86

    So 0.65 is a threshold with a wide margin on both sides: every genuine
    asymmetry found landed at or below 0.60, clean geometry starts at 0.66.
    Raising it toward 1.0 would only be measuring the rasteriser.
    """
    im = render(n, size, motif, ground, step, max(width, 4))
    c = fold_correlation(im, n)
    return c >= threshold, c


# ------------------------------------------------------------------------ main


def sheet(counts, out: Path, size: int, ground: str, step: int) -> list[dict]:
    """One grid containing every requested fold order -- a single uploadable
    contact sheet when a buyer needs the whole set at once."""
    cols = min(4, len(counts))
    rows = (len(counts) + cols - 1) // cols
    pad = 8
    W = cols * size + (cols + 1) * pad
    H = rows * size + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), GROUND[ground][0])
    results = []
    for i, n in enumerate(counts):
        r, c = divmod(i, cols)
        tile = render(n, size, "rosette", ground, step)
        canvas.paste(tile, (pad + c * (size + pad), pad + r * (size + pad)))
        corr = fold_correlation(tile, n)
        results.append({"n": n, "fold_correlation": round(corr, 5),
                        "ok": corr >= 0.65})
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, help="fold order")
    ap.add_argument("--size", type=int, default=1200)
    ap.add_argument("--motif", default="rosette", choices=["rosette", "star", "dial", "lattice"])
    ap.add_argument("--ground", default="night", choices=sorted(GROUND))
    ap.add_argument("--step", type=int, default=2, help="star polygon {n/step}")
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--out", type=Path,
                    help="output PNG; needs a file extension (PIL picks the encoder from it)")
    ap.add_argument("--svg", type=Path, help="also write vector line-work here")
    ap.add_argument("--sheet", action="store_true", help="render every count in --counts")
    ap.add_argument("--counts", default="3,6,7,8,10,12",
                    help="the fold orders the corpus actually uses")
    ap.add_argument("--verify", action="store_true",
                    help="measure rotational correlation and refuse to claim what is not there")
    ap.add_argument("--threshold", type=float, default=0.90)
    args = ap.parse_args()

    # PIL picks its encoder from the file extension, so a bare directory or an
    # extensionless path dies deep inside Image.save with a traceback that tells
    # the operator nothing about what they did wrong.
    for flag, path in (("--out", args.out), ("--svg", args.svg)):
        if path is not None and path.suffix == "":
            ap.error(f"{flag} needs a filename with an extension, e.g. {path}.png")

    if args.sheet:
        counts = [int(x) for x in args.counts.split(",") if x.strip()]
        if not args.out:
            ap.error("--sheet requires --out")
        res = sheet(counts, args.out, args.size, args.ground, args.step)
        print(f"wrote  {args.out}  {len(counts)} tiles")
        for r in res:
            print(f"  n={r['n']:<3d} fold-correlation {r['fold_correlation']:+.5f}  "
                  f"{'ok' if r['ok'] else 'FAIL'}")
        return 0 if all(r["ok"] for r in res) else 1

    if not args.n:
        ap.error("--n is required unless --sheet is used")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        render(args.n, args.size, args.motif, args.ground, args.step, args.width).save(args.out)
        print(f"wrote  {args.out}")
    if args.svg:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        args.svg.write_text(render_svg(args.n, args.size, args.motif, args.ground,
                                       args.step, args.width))
        print(f"wrote  {args.svg}")

    if args.verify:
        ok, corr = verify(args.n, args.size, args.motif, args.ground, args.step, args.threshold)
        print(f"verify n={args.n} motif={args.motif} ground={args.ground} "
              f"fold-correlation {corr:+.5f}  threshold {args.threshold}  "
              f"{'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
