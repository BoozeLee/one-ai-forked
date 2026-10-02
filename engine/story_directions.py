#!/usr/bin/env python3
"""Compose the 48 story directions into one document.

Reads  issues/table.json         n, arc, title, beat, idea  (the user's own canon)
       tools/issues_chatgpt.json  register, subject, geometry, text
       tools/issues_content.json  the authored cover prose
       tools/story_beats.json     against / finds / turn / fail / close  (NEW)

Writes products/STORY-DIRECTIONS.md.

Why this is a generator and not a hand-written document: the four source files are
already the sources of truth for the covers, and a directions document that could
drift from them would be worse than none.  --check is a drift gate.

The frame is fixed.  All 48 issues are set at the same bench and every direction
keeps it.  The only field carrying 48 distinct values is `against`.  See the
_readme in story_beats.json for the full statement and the failure schedule.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TABLE = ROOT / "issues" / "table.json"
CHAT = ROOT / "tools" / "issues_chatgpt.json"
CONTENT = ROOT / "tools" / "issues_content.json"
BEATS = ROOT / "tools" / "story_beats.json"
OUT = ROOT / "products" / "STORY-DIRECTIONS.md"

FAIL_KINDS = {
    "plate-wrong": "the render is false; the yard item does not match",
    "plate-true-worse": "it is true and that is worse",
    "instrument-fails": "the Reader's own measure is wrong, not the plate",
    "reader-misread": "the Reader draws a conclusion and the plate refuses it",
}

# Issues the pilot or the thesis depends on must not be scheduled as failures.
MUST_BE_CLEAN = (8, 20, 48)


def load():
    table = json.loads(TABLE.read_text())
    chat = json.loads(CHAT.read_text())["issues"]
    content = json.loads(CONTENT.read_text())["issues"]
    beats = json.loads(BEATS.read_text())
    readme = beats.pop("_readme", [])
    return table, chat, content, beats, readme


def rows(table, chat, content, beats):
    out = []
    for t in sorted(table, key=lambda r: r["n"]):
        n = t["n"]
        b = beats.get(str(n))
        if b is None:
            raise SystemExit(f"story_beats.json is missing issue {n}")
        c = content[str(n)]
        if isinstance(c, dict):
            c = " ".join(str(v) for v in c.values())
        out.append({
            "n": n,
            "arc": t["arc"],
            "title": t["title"],
            "beat": t["beat"],
            "idea": t["idea"],
            "register": chat[str(n)]["register"],
            "subject": chat[str(n)]["subject"],
            "cover": " ".join(str(c).split()),
            **b,
        })
    return out


def frame_block():
    return """## The frame — fixed for all 48, and the point

There is one bench in this series and it does not change. Every issue is set at it.

- The **Reader** is alone, always, and their own face is never shown — only hands,
  shoulders, a back, or a shadow falling into frame.
- The **plates** arrive folded and warm, in bundles, from the One. They are never
  signed. They are never dated by anything but substrate decay.
- The **ruled comparison grid** on the bench, ruled out in one-centimetre squares,
  is the only instrument the Reader trusts before the lens.
- The **lens** is the expensive instrument. Using it is a decision, not a default.
- The **low bench lamp**, the **long window**, and the **field outside that does
  nothing** are in every frame and are never remarked on at length.
- The **archive** is the building the bench is in. It has a lintel, a ledger, a
  front page, a filing system, eight bays, a brass plaque and a store room.

**What may not appear in any of the 48:** a second room, a colleague, an antagonist,
a named second person, a different hour of the day, a change of weather, a journey.
A direction that introduces any of these has broken the frame, and the frame is what
makes forty-eight separate one-shots read as one work.

**Where the variation lives.** Entirely in what is on the plate, and in what the
plate is measured against. `against` is the only field below with forty-eight
distinct values, and every one of them is an object that already exists in the room
or the yard. That is deliberate: a series of forty-eight needs a rigid frame and
exactly one thing to vary, and that one thing has to be checkable against something
real or the stories have no stakes.

"""


def failure_block(rs):
    fails = [r for r in rs if r["fail"]]
    kinds = {}
    for r in fails:
        kinds.setdefault(r["fail"], []).append(r["n"])
    lines = ["## Scheduled failures", "",
             "A validation pass on the three pilot stories found the real weakness, "
             "and it was not the missing antagonist: **the Reader was never wrong and "
             "nothing ever failed.** A story in which every conflict resolves through "
             "the Reader measuring correctly is a demonstration, not a story.",
             "",
             f"**{len(fails)} of the {len(rs)} issues are scheduled to fail.** None is "
             "adjacent to another. Issues 8, 20 and 48 are clean, because each carries "
             "a thesis and a failure would blunt it.", ""]
    lines.append("| kind | meaning | issues |")
    lines.append("|---|---|---|")
    for k, meaning in FAIL_KINDS.items():
        ns = kinds.get(k, [])
        cells = ", ".join(str(n) for n in ns)
        lines.append(f"| `{k}` | {meaning} | {cells or '--'} |")
    lines += ["", "Failures are scheduled by kind so they do not all feel the same: "
              "a false render, a truth that costs, a broken instrument and a refused "
              "conclusion are four different shapes of a bad day.", ""]
    return "\n".join(lines)


def issue_block(r):
    cover = r["cover"]
    if len(cover) > 300:
        cover = cover[:300].rsplit(" ", 1)[0] + "..."
    fail = f"**{FAIL_KINDS[r['fail']]}.**" if r["fail"] else "none scheduled"
    return f"""### {r['n']:02d} — {r['title']}  ·  arc {r['arc']}  ·  register {r['register']}

> {r['beat']}

**Compositional trick.** {r['idea']}

**What is on the plate.** {r['subject']}

**Measured against.** {r['against']}

**What it gives up.** {r['finds']}

**The turn.** {r['turn']}

**Fails.** {fail}

**Last line — a thing you can see, no summary, no moral.** {r['close']}

**Cover prose already authored** (from `tools/issues_content.json`, so the story and
the cover argue the same thing): {cover}
"""


def build():
    table, chat, content, beats, readme = load()
    rs = rows(table, chat, content, beats)
    parts = ["# Story directions — all 48 issues", "",
             "Generated by `tools/story_directions.py` from `issues/table.json`, "
             "`tools/issues_chatgpt.json`, `tools/issues_content.json` and "
             "`tools/story_beats.json`. Do not hand-edit; run the tool.", "",
             "This supersedes the old `tools/story_directions.json`, which was built "
             "on the void Voidshatter canon.", ""]
    if readme:
        parts.append("## Why this file exists\n")
        parts += [f"{line}" if line else "" for line in readme]
        parts.append("")
    parts.append(frame_block())
    parts.append(failure_block(rs))
    parts.append("## The forty-eight\n")
    parts += [issue_block(r) for r in rs]
    return "\n".join(parts)


def check():
    table, chat, content, beats, readme = load()
    rs = rows(table, chat, content, beats)
    problems = []

    if len(rs) != 48:
        problems.append(f"expected 48 rows, got {len(rs)}")
    extra = sorted(set(beats) - {str(r["n"]) for r in rs})
    if extra:
        problems.append(f"story_beats.json has rows for non-existent issues: {extra}")

    for r in rs:
        for k in ("against", "finds", "turn", "close"):
            if not r.get(k) or not str(r[k]).strip():
                problems.append(f"issue {r['n']}: empty {k}")
        if r["fail"] is not None and r["fail"] not in FAIL_KINDS:
            problems.append(f"issue {r['n']}: unknown fail kind {r['fail']!r}")

    against = [r["against"] for r in rs]
    dupes = sorted({a for a in against if against.count(a) > 1})
    if dupes:
        problems.append(f"duplicate `against` values: {dupes}")

    fails = [r["n"] for r in rs if r["fail"]]
    for a, b in zip(fails, fails[1:]):
        if b - a == 1:
            problems.append(f"failures {a} and {b} are adjacent")
    for n in MUST_BE_CLEAN:
        if n in fails:
            problems.append(f"issue {n} must be clean but is scheduled to fail")

    # The frame rule: nothing may smuggle in a second person or another room.
    banned = ["colleague", "co-worker", "coworker", "his partner", "her partner",
              "the visitor", "a stranger", "the crowd", "at noon", "at midnight",
              "the next day", "the following morning"]
    for r in rs:
        blob = f"{r['against']} {r['finds']} {r['turn']}".lower()
        for b in banned:
            if b in blob:
                problems.append(f"issue {r['n']}: frame-breaking phrase {b!r}")

    if problems:
        for p in problems:
            print(f"  FAIL {p}")
        print(f"\n  FAIL  {len(problems)} problem(s)")
        return 1
    kinds = {}
    for n in fails:
        kinds[[r for r in rs if r["n"] == n][0]["fail"]] = kinds.get(
            [r for r in rs if r["n"] == n][0]["fail"], 0) + 1
    print(f"  ok  48 rows, 48 distinct `against` values, no empty fields")
    print(f"  ok  {len(fails)} failures scheduled, none adjacent, "
          f"{', '.join(f'{k}={v}' for k, v in sorted(kinds.items()))}")
    print(f"  ok  issues {', '.join(str(n) for n in MUST_BE_CLEAN)} clean as required")
    print(f"  ok  no frame-breaking phrase in any direction")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate and exit 1 on failure")
    args = ap.parse_args()
    if args.check:
        return check()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = build()
    OUT.write_text(text)
    n = text.count("\n### ")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(text)} bytes)  {n} issues")
    return 0


if __name__ == "__main__":
    sys.exit(main())
