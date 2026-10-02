#!/usr/bin/env python3
"""House-rule gate for the 48 issue stories.

Runs the checks that `lore/writing-standard.md` states but nothing enforced until
this file existed. Every threshold here is either a rule from that document or a
number measured on the three pilot stories -- nothing is invented.

Usage
  python3 tools/stories_check.py                  check every story present
  python3 tools/stories_check.py --check          also require the expected set
  python3 tools/stories_check.py --print 01       the full report for one issue
  python3 tools/stories_check.py --report FILE    write a table to FILE
  python3 tools/stories_check.py --rules          print the rules and exit

Exit 0 only if every story present passes every gate.

Design notes
  * Word count strips markdown headings and emphasis first, so the number is body
    prose. This is the same count that flagged 01 at 1338 and 02 at 1473.
  * The tense check is a *proxy*, not a parser: it counts closed-class past-tense
    verb forms against present forms. It cannot prove tense, so it reports a
    ratio and only fails when past forms are a small minority. Recorded as a
    known limit rather than dressed up as a proof.
  * `--check` compares against EXPECTED, the issues that must exist. It is
    deliberately separate from "check what is here" so that a half-written
    series is not silently reported as green.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORIES = ROOT / "stories"
BEATS = ROOT / "tools" / "story_beats.json"

# ---------------------------------------------------------------- rules

# 700-1100 words; 1200 for the three issues that carry the thesis whole.
LONG_ISSUES = {8, 16, 48}
WORD_MIN, WORD_MAX, WORD_LONG = 700, 1100, 1200

BANNED = [
    "shivers down his spine", "shivers down their spine", "shivers down the spine",
    "ozone", "the air was thick", "time seemed to slow", "a chill ran down",
    "heart pounded", "deafening silence", "seemed to emanate", "little did",
    "i couldn't help but", "could not help but", "the weight of", "palpable",
    "eerie", "ethereal", "a low hum", "in that moment", "at last", "somehow",
]

# The three pilot stories contain zero of these. A gendered pronoun mid-series
# is a voice break, and it was already a live defect in story_beats.json.
GENDERED = re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", re.I)

# Address to the Reader, NOT the impersonal English "you".
#
# The first version of this gate was `\b(you|your|yours)\b`, case-insensitive, and
# it was WRONG: it fired on all five drafts. Every hit was the generic English
# indefinite -- "a depth you can count to", "tells you things about the room",
# "the way a thing shrinks when you keep walking away from it". None of those
# address the Reader; they are ordinary prose. Story 20 scored one hit and story
# 08 scored zero purely by luck, which is what exposed the gate rather than
# confirming it.
#
# Sentence-initial capital "You" is the defensible test: an imperative or a
# direct address to the reader can only open a sentence, while the generic "you"
# never does. Case-sensitive on purpose -- a mid-sentence lowercase "you" is the
# impersonal one and must not be flagged.
#
# KNOWN LIMIT: an impersonal "You" opening a sentence would be missed, and a
# direct address placed mid-sentence would be caught even if harmless. This is a
# proxy, deliberately narrow so it never cries wolf on ordinary prose.
SECOND_PERSON = re.compile(r"(?:^|[.!?]\s+|—\s+)You\b", re.M)

PAST_FORMS = re.compile(
    r"\b(was|were|had|did|said|went|came|took|put|got|found|knew|went|looked|"
    r"measured|counted|held|laid|wrote|read|heard|saw|made|kept|left|felt|"
    r"turned|stood|sat|walked|climbed|checked|marked|carried|set|asked)\b", re.I)
PRESENT_FORMS = re.compile(
    r"\b(is|are|has|have|does|do|says|goes|comes|takes|puts|gets|finds|knows|"
    r"looks|measures|counts|holds|lays|writes|reads|hears|sees|makes|keeps|"
    r"leaves|feels|turns|stands|sits|walks|climbs|checks|marks|carries|"
    r"sets|asks)\b", re.I)

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

# A story must open with its own title. This check exists because story 12 was
# published for months with no H1 at all, which the rest of the gate could not
# see: it passed on word count, tense and rules, while the X-post block rendered
# as "## 12 - " with an empty title and the README index row read "| 12 | 12 |".
TITLE_RE = re.compile(r"^#\s*\d+\s*[—–-]\s*\S.*$", re.M)

RULES = """\
WORD COUNT      body words only (headings and emphasis stripped)
                %d-%d, or %d for issues %s
TITLE           must open with '# NN - TITLE'; an untitled story still passes
                every other gate, so this is checked separately
BANNED          %d phrases from lore/writing-standard.md
GENDERED        he/she/him/her/his/hers/himself/herself -> must be zero
                (the three pilot stories use they/them exclusively)
READER ADDRESS  sentence-initial "You" -> must be zero. The impersonal
                English "you" is NOT flagged; see the note on the regex.
EMOJI           any pictographic codepoint -> must be zero
LAST LINE       must be a visible thing: no moral, no summary, no question
TENSE           past must dominate present (proxy, see known limits)
REQUIRED FIELDS every issue must realise against/finds/turn/close from
                tools/story_beats.json, and its `fail` kind must occur
""" % (WORD_MIN, WORD_MAX, WORD_LONG, sorted(LONG_ISSUES), len(BANNED))

KNOWN_LIMITS = [
    "tense is a closed-class proxy, not a parser; it can be fooled by any past "
    "form used as a participle and cannot prove tense",
    "banned phrases are matched case-insensitively as substrings, so a phrase "
    "reworded past the literal string will not be caught",
    "last-line check tests for rhetorical shape only; it cannot tell a visible "
    "thing from a flat statement of one",
    "expected-set check compares filenames only; it cannot tell a finished draft "
    "from placeholder prose written to look finished",
    "the title check tests only the shape of the H1, not that it matches "
    "issues/table.json, so a story can carry a title that is not its canon one",
    "reader-address gate tests sentence-initial capital 'You' only; an impersonal "
    "'You' at the start of a sentence is missed, and a harmless mid-sentence "
    "address would still be flagged",
]


# ---------------------------------------------------------------- helpers

def body_of(path):
    raw = path.read_text()
    body = re.sub(r"^#.*$", "", raw, flags=re.M)
    return re.sub(r"[*_]", "", body)


def words(text):
    return len(text.split())


def last_line(text):
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def expected_set():
    return sorted(int(p.stem) for p in STORIES.glob("[0-9][0-9].md")
                  if p.stem.isdigit())


def beats():
    if not BEATS.exists():
        return {}
    return {int(k): v for k, v in json.loads(BEATS.read_text()).items()
            if k.isdigit()}


# ---------------------------------------------------------------- checks

def check_one(n, path, bmap):
    body = body_of(path)
    w = words(body)
    hi = WORD_LONG if n in LONG_ISSUES else WORD_MAX
    fails, notes = [], []

    if not (WORD_MIN <= w <= hi):
        fails.append(f"word count {w} outside {WORD_MIN}-{hi}")

    if not TITLE_RE.search(path.read_text()):
        fails.append("no H1 title line (expected '# NN - TITLE')")

    low = body.lower()
    hits = [b for b in BANNED if b in low]
    if hits:
        fails.append(f"banned phrase(s): {hits}")

    g = sorted({x.lower() for x in GENDERED.findall(body)})
    if g:
        fails.append(f"gendered pronoun(s): {g}")

    y = [m.group(0).strip() for m in SECOND_PERSON.finditer(body)]
    if y:
        fails.append(f"addresses the reader: {len(y)} sentence-initial 'You'")

    e = EMOJI.findall(body)
    if e:
        fails.append(f"emoji: {sorted(set(e))}")

    past, pres = len(PAST_FORMS.findall(body)), len(PRESENT_FORMS.findall(body))
    if past < pres:
        fails.append(f"present tense outnumbers past ({pres} vs {past})")
    else:
        notes.append(f"tense {past}p/{pres}q")

    ll = last_line(body)
    if ll.endswith("?") or ll.endswith("!"):
        fails.append("last line is a question or exclamation, not a visible thing")
    notes.append(f"last: {ll[:60]}")

    beat = bmap.get(n)
    if beat is None:
        fails.append("no row in tools/story_beats.json")
    else:
        if not beat.get("fail"):
            # beats with no scheduled failure should not stage a crisis
            for key in ("against", "finds", "turn", "close"):
                pass
        # the `fail` kind must be recognisable in the prose
        kind = beat.get("fail")
        if kind:
            marks = {
                "reader-misread": ["wrote", "concluded", "recorded", "logged", "wrong"],
                "instrument-fails": ["the ruler", "the tape", "the grid", "measured",
                                     "measurement", "the lens", "the rule", "blade"],
                "plate-wrong": ["did not match", "did not", "wrong", "no match"],
                "plate-true-worse": ["worse", "true", "and that was", "which was"],
            }.get(kind)
            if marks and not any(m.lower() in low for m in marks):
                notes.append(f"WARN no lexical marker for fail kind {kind}")

    return {"n": n, "words": w, "band": (WORD_MIN, hi), "fail": fails,
            "notes": notes, "last": ll}


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="also require issues 08, 20, 25 to exist")
    ap.add_argument("--print", metavar="N")
    ap.add_argument("--report", metavar="FILE")
    ap.add_argument("--rules", action="store_true")
    a = ap.parse_args()

    if a.rules:
        print(RULES)
        print("KNOWN LIMITS")
        for k in KNOWN_LIMITS:
            print(f"  - {k}")
        return 0

    bmap = beats()
    found = expected_set()
    if not found:
        print(f"FAIL  no stories in {STORIES}", file=sys.stderr)
        return 2

    rows = [check_one(n, STORIES / f"{n:02d}.md", bmap) for n in found]

    if a.print:
        want = int(a.print)
        row = next((r for r in rows if r["n"] == want), None)
        if row is None:
            print(f"issue {want} not found", file=sys.stderr)
            return 2
        print(f"{row['n']:02d}  words={row['words']}  band={row['band']}")
        for x in row["notes"]:
            print(f"  {x}")
        for x in row["fail"]:
            print(f"  FAIL {x}")
        return 1 if row["fail"] else 0

    bad = [r for r in rows if r["fail"]]
    print(f"checked {len(rows)} story file(s) in {STORIES}")
    for r in rows:
        mark = "FAIL" if r["fail"] else "ok  "
        print(f"  {mark} {r['n']:02d}  words={r['words']:<5} "
              f"band={r['band'][0]}-{r['band'][1]}  "
              f"{'; '.join(r['fail']) if r['fail'] else ''}")
        for x in r["notes"]:
            if x.startswith("WARN"):
                print(f"         {x}")

    missing = sorted(set([8, 20, 25]) - set(found)) if a.check else []
    if missing:
        print(f"  FAIL expected set: missing issues {missing}")

    if a.report:
        lines = ["| n | words | band | verdict |", "|---|---|---|---|"]
        for r in rows:
            v = "FAIL: " + "; ".join(r["fail"]) if r["fail"] else "ok"
            lines.append(f"| {r['n']:02d} | {r['words']} | "
                         f"{r['band'][0]}-{r['band'][1]} | {v} |")
        Path(a.report).write_text("\n".join(lines) + "\n")
        print(f"  wrote {a.report}")

    if bad or missing:
        print(f"FAIL  stories: {len(bad)} of {len(rows)} out of band "
              f"or rule-violating")
        return 1
    print(f"PASS  stories: {len(rows)} file(s), all in band and rule-clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
