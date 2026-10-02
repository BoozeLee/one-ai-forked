#!/usr/bin/env python3
"""Compose copy-pasteable ChatGPT cover prompts for every issue in the series.

Reads  issues/table.json          titles, arcs, canon beats, palette names
Reads  tools/issues_chatgpt.json  per-issue ART DIRECTION: register, subject, geometry, text
Reads  tools/issues_tech.json     per-issue ENGINEERING SPEC: bbox, measured math, detail, negatives
Reads  tools/issues_content.json  per-issue CONTENT: the plain-language description of the image
Writes covers/prompts-chatgpt/NN.md   one issue per file
Writes covers/prompts-chatgpt/ALL.md  all 48 in one file

Four inputs, four jobs, no overlap:

  issues_chatgpt.json  WHO and WHAT, in one clause. Hand-written prose.
  issues_tech.json     HOW BIG and WHERE, in numbers. Hand-written. Auditable.
  issues_content.json  WHAT A VIEWER SEES, in sentences. Hand-written. No digits.
  this file            Series identity, registers, the shared physics, the gates.

Ordering is the whole argument. A model reads the first block as what the image
IS and the last block as constraints, so the subject sits at the top and the
print specification sits at the bottom. Leading with process produced a print
shop order that rendered as boring art.

No network, no model calls. Pure text composition.

    python3 tools/prompts_chatgpt.py            # write the .md files
    python3 tools/prompts_chatgpt.py --print 7  # print issue 7's prompt to stdout
    python3 tools/prompts_chatgpt.py --check    # verify the batch, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "issues" / "table.json"
AUTHORED = ROOT / "tools" / "issues_chatgpt.json"
TECH = ROOT / "tools" / "issues_tech.json"
CONTENT = ROOT / "tools" / "issues_content.json"
OUT = ROOT / "covers" / "prompts-chatgpt"

TOL = 1e-9
WIDTH = 74
HANG = "  "

# --- the shared physics. One source of truth; every issue inherits it. --------

DELIVERABLE = """One square image, 1:1, 1200 x 1200 px. It is the cover art for issue {n} of
a 48-issue series. A finished comic book cover, not a study and not a
mockup."""

FRAMING = """Percentages are of the frame. x is measured from the LEFT edge, y from the
TOP edge. Concentricity, symmetry and spacing are exact, not approximate —
where a number is given, hold it to 0.002 of the frame. The 200 px
thumbnail is the acceptance test, not the full-size render."""

OUTPUT_RULE = """Output one image only. No captions, no commentary, no alternative
versions, no contact sheet."""

THUMBNAIL_RULE = """Reads at 200 px: the image must still be one idea when it is 200 px on its
long edge. When it turns to mush, cut the SUBJECT, never the ink density —
the density is the quality."""

CORNER_RULE = """The four corners stay clean. No invented signature, no artist handle, no
@-mention, no watermark, no logo, no page number. If a mark appears there
that this brief did not name, regenerate — do not paint it out."""

PLATE_RULE = """Millimetres are on a 120 mm plate: 0.004 is a hairline, 0.008 a normal
contour, 0.014 a structural line. Every rule is straight, parallel to an edge
or on a stated angle, and ruled rather than sketched."""

NO_TEXT = """Absolutely no text in the image: no words, no letters, no numbers, no
captions, no speech bubbles, no lettering of any kind, anywhere —
including inside the border, the grid, the background notation and the
corners."""

# --- registers --------------------------------------------------------------
#
# Each register carries five things: what it is made of, how it is processed,
# how it is lit, how its line behaves, and what it looks like. Text below is
# verbatim from the corpus that was already generated and approved, so a
# rewrite of this file must not paraphrase it.

REGISTER_SPEC: dict[str, dict[str, object]] = {
    "A": {
        "name": "engraved occult plate",
        "medium": """Steel engraving on a 120 mm plate, printed as a two-plate overprint: a
near-black key plate and a gold ink plate. Every tone is a cut line, not a
wash — the gradations come from the spacing of hatch, never from a
gradient. Gold is laid last, so gold always sits on top of the modelling
and reads as light rather than as colour.""",
        "look": """Near-black ground. An ornate gold Art-Deco double-ruled border with corner
ornaments encloses the whole frame, with a thin gold geometric construction
(a circle, a triangle, a rule) overlaid on the subject. The subject is
monochrome, shaded with near-photographic tonal gradation and cast shadow,
not flat colour. A faint perspective grid recedes behind it. Scattered gold
glyph-shapes, strokes and dashes float across the background as texture:
they may look like real notation, but they spell nothing and they must
never resolve into a readable word, phrase or number.""",
        "light": """A single key at 40 degrees elevation, 25 degrees camera-left, with a fill
ratio of 1:5. Cast shadows fall down and to the right, and they are the
only soft marks in the frame. Every other edge is a cut line.""",
        "line": """Contour 0.008 mm, structural 0.014 mm, hairline 0.004 mm. Hatch sets at 0, 35,
70 and 105 degrees so the four tones never parallel each other. Hatch pitch
tightens with depth: 0.35 mm in the lightest plane, 0.18 mm in the
darkest. Value is carried by pitch alone — a tone is a countable number of
lines per millimetre, not a percentage of grey.""",
        "palette": [
            ["ground, near-black", "#0B0A0D", 0.62],
            ["gold", "#C8A046", 0.13],
            ["warm shadow", "#2A1F1A", 0.11],
            ["subject mid-tone", "#6B5F55", 0.08],
            ["paper highlight", "#EFE6D4", 0.06],
        ],
        "value": [0.34, 0.38, 0.20, 0.08],
    },
    "B": {
        "name": "psychedelic rorschach",
        "medium": """Tattoo-flash screenprint on black, 4 inks: a key plate in near-black,
then magenta, cyan and gold, each laid as hard-edged shapes with no blend.
Where two inks meet they overprint and neither mixes into a third. The
whole image is line-work and flat shape; there is no tonal wash and no
soft gradient anywhere, and the only soft marks are the internal glows
named below.""",
        "look": """Near-black ground. Very dense engraved line-work at screenprint /
tattoo-flash frequency. Hot magenta, red and orange flame-structure on the
outside; cold cyan glow on the inside; small gold and pale-blue accents;
thin filigree filling the corners. Hard-edged shapes, high contrast, no
soft gradients.""",
        "light": """No external key. Light is internal: a cold cyan glow from inside the form
and hot magenta-ember flame-structure on the outside, so the image is lit
by its own structure. A single hairline rim on ONE side only, never on
both.""",
        "line": """Contour 0.006 mm at 85 lines per 10 mm of plate — screenprint frequency,
the densest in the series. Hatch sets at 0, 45, 90 and 135 degrees. Where
the line-work exceeds 120 lines per 10 mm it becomes mush at thumbnail
size, so the frequency is a ceiling and not a target.""",
        "palette": [
            ["ground, near-black", "#0A0709", 0.55],
            ["magenta", "#FF2D95", 0.17],
            ["cyan", "#35C6E8", 0.11],
            ["ember red", "#E0331B", 0.09],
            ["gold", "#E0B451", 0.05],
            ["pale blue", "#A9D3E8", 0.03],
        ],
        "value": [0.52, 0.24, 0.16, 0.08],
    },
    "C": {
        "name": "technical diagram poster",
        "medium": """Plotter output, then engraved over: a dark navy ground carrying a hairline
cyan grid, with a single bright outline colour for the nodes and a warm
core colour. Nothing is filled heavily. The drawing is a document, so
every mark is aligned to the 0.01 grid and nothing is placed by eye.""",
        "look": """Dark navy near-black ground with a faint vignette and a thin grid. A
hub-and-spoke layout: circular nodes outlined in one bright colour at even
radial spacing, thin leader lines running out to small labels, faint
concentric rings behind, and corner tick marks at the edge of the frame.
Typography is tiny, tidy and aligned to the grid, never scattered.""",
        "light": """No modelled light at all. Everything is orthographic: no perspective on the
structures, no cast shadow, no foreshortening. A faint vignette at the
corners is the only non-orthographic effect permitted.""",
        "line": """Rules 0.003 mm, node outlines 0.005 mm, the frame grid 0.002 mm on a 0.0833
of frame-width pitch (a 12 x 12 minor grid, major every fifth). Leader
lines 0.0025 mm. Every rule is parallel to a frame edge and no two cross at
an angle other than 0 or 90 degrees.""",
        "palette": [
            ["ground, navy-black", "#060A14", 0.60],
            ["grid line", "#12263A", 0.12],
            ["node outline cyan", "#3FBFE0", 0.11],
            ["warm core", "#FF6B4A", 0.07],
            ["label white", "#E8EEF2", 0.06],
            ["amber tick", "#F0A93B", 0.04],
        ],
        "value": [0.44, 0.30, 0.18, 0.08],
    },
    "D": {
        "name": "inked comic page",
        "medium": """A real comic page: cream stock, black ink, one flat accent colour, and
screentone dots for every shadow. Drawn with a brush, inked by hand. The
panel grid, the gutters and the margins are real geometry measured in the
frame, not a suggestion of a page layout.""",
        "look": """Cream paper, black ink, a real panel grid with real gutters and margins,
screentone dots for shadow, and exactly one flat accent colour. Confident
brush line, real inking.""",
        "light": """A single key from the upper left in every panel, consistent across the
page. A panel that breaks the key reads as pasted in. Shadows are 20% and
45% screentone dots; the darkest 8% of each panel is solid ink.""",
        "line": """Brush line with real tapering: thick on the shadow side, thin on the light, and
thinner again on the more distant figure. Panel rules 0.014 mm. Screentone
at 65 lines per inch, so the dot pitch is 0.39 mm and the dot is visible
rather than fused. Three tones only: solid, 45%, 20%.""",
        "palette": [
            ["cream paper", "#F2EAD6", 0.55],
            ["black ink", "#14110E", 0.34],
            ["screentone 45%", "#9A9086", 0.04],
            ["accent magenta", "#E21A7A", 0.07],
        ],
        "value": [0.30, 0.34, 0.24, 0.12],
    },
    "E": {
        "name": "one-ink plate",
        "medium": """A single ink on cream stock, printed one colour, no second plate and no
registration. There is no grey ink and no black ink: every tone in the
image is either bare paper or the one colour, struck at three line weights
and four dot densities.""",
        "look": """Cream paper and a single ink, nothing else. No second colour anywhere. All
separation between elements is carried by line weight and dot density.""",
        "light": """No modelled light and no cast shadow. Depth is entirely a function of line
weight and dot density, which is the argument this register exists to
make.""",
        "line": """Exactly three weights: 0.004, 0.008 and 0.014 mm. Exactly four dot densities:
0%, 25%, 50%, 75%, at 65 lines per inch. That is 12 tone values and no
others — a fourth weight or a fifth density breaks the proof.""",
        "palette": [
            ["cream paper", "#F0EAD8", 0.62],
            ["ink", "#C0202E", 0.30],
            ["ink tint", "#E4DCC8", 0.08],
        ],
        "value": [0.26, 0.32, 0.28, 0.14],
    },
}

# --- emitters ---------------------------------------------------------------


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def point(xy: tuple[float, float]) -> str:
    return f"({pct(xy[0])}, {pct(xy[1])})"


def dimensions(w: float, h: float) -> str:
    return f"{pct(w)} of frame width by {pct(h)} of frame height"


def wrap(text: str, width: int = WIDTH) -> str:
    """Fill to `width`, never breaking on a hyphen or inside a word."""
    return "\n".join(
        textwrap.wrap(
            " ".join(text.split()),
            width=width,
            break_on_hyphens=False,
            break_long_words=False,
        )
    )


def bullet(text: str) -> str:
    """A bullet that hangs: continuation lines indent under the text, not the dash."""
    return "\n".join(f"- {line}" if i == 0 else HANG + line
                     for i, line in enumerate(wrap(text, WIDTH - 2).split("\n")))


def sentence(text: str) -> str:
    """Authored strings are stored lowercase and unpunctuated; emitted prose
    needs a capital and a full stop. Strip any trailing punctuation first, or
    a string that already ends in a period gains a second one."""
    text = " ".join(text.split()).rstrip(".;: ")
    return text[:1].upper() + text[1:] + "." if text else text


def spec(reg: str) -> dict[str, object]:
    return REGISTER_SPEC[reg]


def placement_block(entry: dict) -> str:
    cx, cy, w, h = entry["bbox"]
    return wrap(
        f"HOW IT IS ARRANGED\n"
        f"The subject's bounding box: centre {point((cx, cy))}, measuring "
        f"{dimensions(w, h)} — area {pct(w * h)} of the frame. Off-centre "
        "placement is the default; only the issues whose brief says otherwise "
        "are centred, and they are centred deliberately. One dominant idea per "
        "image, and the corners stay clean."
    )


def geometry_block(entry: dict) -> str:
    return "\n".join(
        ["FRAME MATH — measured, and checkable on the finished image"]
        + [bullet(m) for m in entry["math"]]
    )


def colour_block(reg: str, entry: dict) -> str:
    """Palette and value share one heading. `value_block` must therefore return
    a headingless body: emitting both heads made COLOUR AND TONE appear twice
    in every prompt, and a duplicated heading reads to a model as two sections."""
    colours = entry.get("palette")
    if colours is None:
        colours = spec(reg)["palette"]
    total = sum(c[2] for c in colours)
    if abs(total - 1.0) > TOL:
        raise SystemExit(f"palette shares sum to {total}, not 1.0")

    value = entry.get("value")
    if value is None:
        value = spec(reg)["value"]
    total = sum(value)
    if abs(total - 1.0) > TOL:
        raise SystemExit(f"value shares sum to {total}, not 1.0")

    names = ["shadow", "mid-tone", "light", "specular"]
    tone = ", ".join(f"{n} {pct(x)}" for n, x in zip(names, value))

    out = [
        "COLOUR AND TONE",
        wrap(
            f"The {len(colours)} values below are the only colours in the image. "
            "Shares are of the frame area and they sum to 100%."
        ),
        "",
    ]
    for name, hexv, share in colours:
        out += [bullet(f"{hexv}  {name} — {pct(share)} of the frame"), ""]
    out.append(
        wrap(
            f"Tone by area: {tone}. Four bands, no fifth. Hold the split; a "
            "frame that drifts to one band is flat."
        )
    )
    return "\n".join(out)


def text_rule(entry: dict) -> str:
    allowed = entry.get("text", "none")
    if allowed == "none" or not allowed.get("allow"):
        return "TEXT\n" + wrap(NO_TEXT)
    allow = allowed["allow"]
    named = " / ".join(allow)
    one = len(allow) == 1
    total = sum(len(s) for s in allow)
    return "TEXT\n" + wrap(
        f"Render exactly {named} and nothing else — {len(allow)} "
        f"{'string' if one else 'strings'}, {total} characters in total. Spell "
        f"{'this string' if one else 'these strings'} correctly and character "
        "for character. Every other text-shaped mark anywhere in the image must "
        "read as writing that spells nothing: glyphs, ticks and dummy rules are "
        "fine, readable words and numbers are not. Nothing in the border, the "
        "corners or the background may be legible."
    )


def build(row: dict, authored: dict, tech: dict, content: dict) -> str:
    n = row["n"]
    key = str(n)
    a = authored["issues"][key]
    t = tech["issues"][key]
    reg = a["register"]
    if reg not in REGISTER_SPEC:
        raise SystemExit(f"issue {n}: unknown register {reg!r}")
    s = spec(reg)

    parts: list[str] = []

    parts.append("DELIVERABLE\n" + DELIVERABLE.format(n=n))

    parts.append("WHAT THE IMAGE SHOWS\n" + wrap(content["issues"][key]))

    parts.append(
        "THE ONE IDEA\n"
        + wrap(
            f"{sentence(a['subject'])} This is the one thing the eye lands on. "
            "The reason it works: " + " ".join(a["geometry"].split())
        )
    )

    parts.append("FRAME CONVENTION\n" + wrap(FRAMING))
    parts.append(placement_block(t))
    parts.append(geometry_block(t))

    if a.get("ground"):
        parts.append(
            "GROUND — this issue overrides its register\n"
            + wrap(sentence(a["ground"]))
        )

    parts.append(
        "HOW IT IS MADE\n" + wrap(s["medium"]) + "\n" + wrap(s["look"])
    )
    parts.append(colour_block(reg, t))
    parts.append(
        "LIGHT AND LINE\n"
        + wrap(s["light"])
        + "\n"
        + wrap(s["line"])
        + "\n"
        + wrap(PLATE_RULE)
    )
    parts.append(
        "SURFACE AND EDGE\n" + "\n\n".join(bullet(d) for d in t["detail"])
    )
    parts.append(
        "DO NOT\n"
        + "\n\n".join([bullet(x) for x in t["neg"]] + [bullet(CORNER_RULE),
                                                       bullet(THUMBNAIL_RULE)])
    )
    parts.append(text_rule(a))
    parts.append("OUTPUT\n" + wrap(OUTPUT_RULE))

    return "\n\n".join(parts)


# --- file wrapper -----------------------------------------------------------

CHECK_STEPS = [
    "1. Check the corners first for invented signatures.",
    "2. Check for stray text. If any appears that the brief did not name, "
    "regenerate — do not try to paint it out.",
    "3. Judge it at 200 px as well as full size. If it turns to mush small, "
    "the line-work density is too high; cut the subject, not the ink.",
    "4. Then extend vertically to 2:3 for the cover: art in the upper "
    "two-thirds, blank band at the foot for the logo and issue number. Do "
    "not regenerate at 2:3 — the corpus quality is established at square.",
]


def header(row: dict, authored: dict, tech: dict, content: dict) -> str:
    n = row["n"]
    a = authored["issues"][str(n)]
    t = tech["issues"][str(n)]
    ref = a.get("reference")
    lines = [
        f"# {n:02d} — {row['title']}",
        "",
        f"Arc {row['arc']} · register {a['register']} "
        f"({spec(a['register'])['name']}) · palette {row['device']}",
        "",
        f"> {row['beat']}",
        "",
        f"Compositional trick: {row['idea']}",
        "",
        f"Subject bounding box: {point((t['bbox'][0], t['bbox'][1]))}, "
        f"{dimensions(t['bbox'][2], t['bbox'][3])}",
    ]
    if ref:
        lines += ["", f"Closest reference already in the corpus: `{Path(ref).name}`"]
    lines += [
        "",
        "## Paste this, whole, into ChatGPT",
        "",
        "```",
        build(row, authored, tech, content),
        "```",
        "",
        "## After you get the image",
        "",
        *CHECK_STEPS,
        "",
    ]
    return "\n".join(lines)


# --- gates ------------------------------------------------------------------

FENCE = re.compile(r"^```", re.M)

SECTIONS = (
    "DELIVERABLE", "WHAT THE IMAGE SHOWS", "THE ONE IDEA", "FRAME CONVENTION",
    "HOW IT IS ARRANGED", "FRAME MATH", "GROUND", "HOW IT IS MADE",
    "COLOUR AND TONE", "LIGHT AND LINE", "SURFACE AND EDGE", "DO NOT",
    "TEXT", "OUTPUT",
)

# Sections that only appear for some issues.
OPTIONAL = ("GROUND",)

# Sections whose whole job is engineering numbers. A heading with nothing
# numeric under it is a broken prompt, so the gate reads the body, not the
# heading.
MUST_BE_NUMERIC = ("FRAME MATH", "HOW IT IS ARRANGED", "COLOUR AND TONE",
                   "LIGHT AND LINE")

# Minimum measured statements per issue. Below this the "math" is decoration.
MIN_MATH = 3

# The content block is the reason the image is not boring, so it has a floor on
# substance and a ban on numbers: numbers belong in FRAME MATH where they can be
# audited, and a content paragraph stuffed with measurements stops describing an
# image and starts specifying a drawing.
MIN_CONTENT_SENTENCES = 3
MIN_CONTENT_CHARS = 220


def section_body(body: str, sec: str) -> str | None:
    """Text under a section heading, up to the next section heading.

    Bullet lines start with '-', so the stop condition has to be the next
    heading rather than the next non-blank line, or every section reads as one
    bullet long.
    """
    lines = body.split("\n")
    starts = [i for i, ln in enumerate(lines)
              if any(re.match(rf"^{re.escape(s)}\b", ln) for s in SECTIONS)]
    for pos, i in enumerate(starts):
        if not re.match(rf"^{re.escape(sec)}\b", lines[i]):
            continue
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        return "\n".join(lines[i + 1:end])
    return None


def heading_at(lines: list[str], i: int) -> str | None:
    """Longest section name matching this line, or None.

    Matching the first space-delimited word truncates 'WHAT THE IMAGE SHOWS' to
    'WHAT', which silently turned every section check into a miss.
    """
    best = None
    for s in SECTIONS:
        if re.match(rf"^{re.escape(s)}\b", lines[i]) and (best is None
                                                          or len(s) > len(best)):
            best = s
    return best


def strip_numbers(body: str) -> str:
    """Remove the issue number and bare integers so distinctness cannot be
    inherited from 'issue 7 of a 48-issue series' alone."""
    s = re.sub(r"issue \d+", "issue N", body)
    return re.sub(r"\d+", "#", s)


def check(rows: list[dict], authored: dict, tech: dict, content: dict) -> int:
    """Fail loudly rather than ship a batch that collapsed into near-duplicates."""
    problems: list[str] = []
    bodies: dict[str, list[str]] = {}
    masked: dict[str, list[str]] = {}

    for row in rows:
        n = row["n"]
        key = str(n)
        entry = tech["issues"][key]
        bad = False

        if len(entry.get("math", [])) < MIN_MATH:
            problems.append(
                f"{n:02d}: {len(entry.get('math', []))} measured statements in "
                f"issues_tech.json, floor is {MIN_MATH}"
            )
            bad = True
        bbox = entry.get("bbox", [])
        if len(bbox) != 4 or not all(isinstance(v, (int, float)) for v in bbox):
            problems.append(f"{n:02d}: bbox must be four numbers, got {bbox!r}")
            bad = True

        prose = content["issues"].get(key, "")
        sentences = len([s for s in re.split(r"(?<=[.!?])\s+", prose) if s])
        if len(prose) < MIN_CONTENT_CHARS:
            problems.append(
                f"{n:02d}: content paragraph is {len(prose)} characters, floor is "
                f"{MIN_CONTENT_CHARS} — the brief leads with this, so a thin one "
                "renders as a boring image"
            )
            bad = True
        elif sentences < MIN_CONTENT_SENTENCES:
            problems.append(
                f"{n:02d}: content paragraph has {sentences} sentences, floor is "
                f"{MIN_CONTENT_SENTENCES}"
            )
            bad = True
        if re.search(r"\d", prose):
            problems.append(
                f"{n:02d}: content paragraph contains a digit — numbers belong in "
                "FRAME MATH so they stay auditable"
            )
            bad = True
        if bad:
            # build() unpacks bbox; report the defect instead of raising on it.
            continue

        try:
            body = build(row, authored, tech, content)
        except SystemExit as exc:
            # An emitter that refuses bad data must still produce a report, not
            # a traceback: a gate that raises has not graded the batch.
            problems.append(f"{n:02d}: build refused the input: {exc}")
            continue

        bodies.setdefault(hashlib.sha256(body.encode()).hexdigest()[:12],
                          []).append(f"{n:02d}")
        masked.setdefault(
            hashlib.sha256(strip_numbers(body).encode()).hexdigest()[:12], []
        ).append(f"{n:02d}")

        doc = header(row, authored, tech, content)
        if len(FENCE.findall(doc)) != 2:
            problems.append(f"{n:02d}: expected exactly one fenced code block")
        steps = doc.split("## After you get the image", 1)[1]
        if len(re.findall(r"^\d\.", steps, re.M)) != 4:
            problems.append(f"{n:02d}: expected exactly 4 numbered steps")
        if f"issue {n} of" not in body:
            problems.append(f"{n:02d}: prompt body does not name its own issue number")

        for sec in SECTIONS:
            text = section_body(body, sec)
            if text is None:
                if sec in OPTIONAL:
                    continue
                problems.append(f"{n:02d}: missing section {sec}")
            elif not text.strip():
                problems.append(f"{n:02d}: section {sec} is empty")
            elif sec in MUST_BE_NUMERIC and not re.search(r"\d", text):
                problems.append(f"{n:02d}: section {sec} carries no numbers")

        # A renamed heading can still be emitted twice. Count occurrences rather
        # than assuming one: COLOUR AND TONE was emitted by build() and again by
        # value_block(), and a gate that only asks "is it present?" cannot see it.
        lines = body.split("\n")
        for sec in SECTIONS:
            seen = sum(1 for i, ln in enumerate(lines) if heading_at(lines, i) == sec)
            if seen > 1:
                problems.append(
                    f"{n:02d}: heading {sec} appears {seen} times, once is the "
                    "contract"
                )

        top_at = body.find("WHAT THE IMAGE SHOWS")
        first_numbered = min(
            (body.find(sec) for sec in ("FRAME CONVENTION", "FRAME MATH",
                                        "COLOUR AND TONE", "LIGHT AND LINE")
             if sec in body),
            default=len(body),
        )
        if top_at < 0:
            problems.append(f"{n:02d}: WHAT THE IMAGE SHOWS block is absent")
        elif top_at > first_numbered:
            problems.append(f"{n:02d}: WHAT THE IMAGE SHOWS is not in the top block")

        # A doubled period is a defect; "0..23" is a range. Flag the first,
        # never the second.
        if re.search(r"\.{2,}(?!\d)", body):
            problems.append(f"{n:02d}: doubled period in the prompt body")
        # The trick sentence is introduced by a colon, so it stays lowercase.
        if re.search(r"The reason it works: [A-Z]", body):
            problems.append(
                f"{n:02d}: trick sentence capitalised after the colon"
            )
        for bad_line in (ln for ln in lines if len(ln) > WIDTH + 5):
            problems.append(f"{n:02d}: line over {WIDTH + 5} columns: {bad_line[:40]}...")
        stray = [ln for ln in lines
                 if re.search(r"(?<!^)\S*?- (?![\d(+*/-])", ln)
                 and re.match(r"^[A-Za-z0-9]", ln) and " - " in ln
                 and not ln.startswith("- ")
                 and not re.search(r"\d+ - \d", ln)]
        for ln in stray:
            if re.search(r"[a-z)] - [A-Z#]", ln):
                problems.append(f"{n:02d}: inline bullet: {ln[:50]}...")

    for h, ns in sorted(bodies.items()):
        if len(ns) > 1:
            problems.append(f"duplicate prompt body {h} across issues {', '.join(ns)}")
    for h, ns in sorted(masked.items()):
        if len(ns) > 1:
            problems.append(
                f"issue-number-masked body {h} across issues {', '.join(ns)} "
                "— distinct only by the issue number"
            )
    if len(bodies) != len(rows) or len(masked) != len(rows):
        problems.append(
            f"{len(bodies)} distinct bodies / {len(masked)} number-masked, "
            f"for {len(rows)} issues"
        )

    if problems:
        print("FAIL")
        for p in problems:
            print(f"  {p}")
        return 1

    sizes = [len(build(r, authored, tech, content).encode()) for r in rows]
    print(
        f"PASS  {len(rows)} issues, {len(bodies)} distinct bodies, "
        f"{len(masked)} distinct with issue numbers masked, "
        f"{min(sizes)}-{max(sizes)} bytes per prompt"
    )
    return 0


# --- entry ------------------------------------------------------------------


def load() -> tuple[list[dict], dict, dict, dict]:
    for p in (TABLE, AUTHORED, TECH, CONTENT):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    rows = json.loads(TABLE.read_text())
    authored = json.loads(AUTHORED.read_text())
    tech = json.loads(TECH.read_text())
    content = json.loads(CONTENT.read_text())
    for src, name in ((authored, "issues_chatgpt.json"),
                      (tech, "issues_tech.json"),
                      (content, "issues_content.json")):
        missing = [r["n"] for r in rows if str(r["n"]) not in src["issues"]]
        if missing:
            raise SystemExit(f"{name} is missing entries for: {missing}")
    return rows, authored, tech, content


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", type=int, metavar="N", help="print one prompt and exit")
    ap.add_argument("--check", action="store_true", help="verify the batch, write nothing")
    args = ap.parse_args()

    rows, authored, tech, content = load()

    if args.print:
        row = next(r for r in rows if r["n"] == args.print)
        print(build(row, authored, tech, content))
        return 0

    if args.check:
        return check(rows, authored, tech, content)

    OUT.mkdir(parents=True, exist_ok=True)
    for row in rows:
        (OUT / f"{row['n']:02d}.md").write_text(header(row, authored, tech, content))

    parts = [
        "# Amphetamemes x Voidshatter — ChatGPT cover prompts, all 48",
        "",
        "Generated by `tools/prompts_chatgpt.py` from `issues/table.json`, "
        "`tools/issues_chatgpt.json`, `tools/issues_tech.json` and "
        "`tools/issues_content.json`. Do not hand-edit: edit the JSON and "
        "re-run. `python3 tools/prompts_chatgpt.py --check` verifies the batch.",
        "",
        "One code block per issue. Paste a block whole, on its own, into ChatGPT.",
        "",
    ]
    for row in rows:
        parts += [
            f"## {row['n']:02d} — {row['title']}", "",
            "```", build(row, authored, tech, content), "```", "",
        ]
    (OUT / "ALL.md").write_text("\n".join(parts))

    print(f"wrote {len(rows)} files to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
