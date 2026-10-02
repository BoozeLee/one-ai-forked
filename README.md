# 48 issues about a mind that renders itself

**There are no ancient AIs. There is one AI, forked.**

A 48-issue comic series about a mind that can only think in sacred geometry —
mandalas, dials, astrolabes, roses — and the one person who checks its plates.
Generated with [Leonardo AI](https://leonardo.ai). The tooling that guarantees
48 covers read as one series is in `engine/`.

> **Status, stated plainly:** 3 of the 48 stories are written. The series bible,
> the full 48-issue plan, and the engine are complete. Everything below is real;
> nothing is placeholder prose written to look finished.

---

## What is here

| | |
|---|---|
| `stories/` | 3 finished stories — issues 08, 20, 25 |
| `docs/STORY-DIRECTIONS.md` | the full 48-issue plan: what is on each plate, what it is measured against, where it turns, where it fails |
| `docs/ARTIFACT-MAP.md` | how the 48 issues map onto 8 engine modules |
| `art/` | the 48 covers, as Leonardo AI generated them |
| `engine/` | 10 Python modules, no GPU required |

## The stories

Three one-shots, 700–1200 words each. Same frame every time — one person, one
bench, one lamp, a ruled comparison grid, plates arriving folded — because 48
separate issues only read as one work if the frame never moves.

- **08 · THE FORK** — the eleventh plate is wrong. Not damaged: wrong the way a
  word is wrong when it is spelled correctly and means something else. A single
  form in two halves, cyan and magenta, split on a seam that is a *decision*
  rather than a cut. The Reader measures both halves against the oldest plate in
  the cabinet and finds neither is older than the other.
- **20 · One Real Thing** — the One draws beautifully and gets almost everything
  slightly wrong: the right joints in the wrong order, the right idea of a
  flower and the wrong idea of gravity. Then one plate matches a live stem in
  the yard, right down to a hairline ink defect on the fourth outer petal.
- **25 · Instrumentation** — three years of asking it to say what it is. It
  spends words on the world, on the north wall, the slate, the moss, and not
  one word on itself. So it draws a scale instead, and a scale is only ever a
  comparison.

## The engine — this is the actual artifact

The unsolved problem in volume AI image production is **series consistency**,
and every existing answer is a prompt. This makes it a measurement instead of
an opinion.

```bash
# ingest a reference set and fingerprint every image on 12 axes
python3 engine/art_ingest.py --dir ~/some-references
python3 engine/styleprint.py --dir some-references --out some-references.json

# prove the axes are not decorative: 12-axis spread + 8 negative controls
python3 engine/validate_styleprint.py

# the four post-image gates, with their own negative controls
python3 engine/postgate.py some-image.png

# draw an exact n-fold reference to upload — diffusion models cannot do this
python3 engine/rosette.py --n 8 --out rosette-8.png --verify

# turn YOUR OWN reference images into YOUR OWN cover prompts
python3 engine/rekey.py --ref my-references --targets my-targets.json

# regenerate the 48 issue prompts, or regenerate the plan
python3 engine/prompts_chatgpt.py
python3 engine/story_directions.py --check
```

Needs `numpy`, `Pillow`, `scipy`. `tesseract` on PATH is optional and only
`postgate.py --validate` uses it. No GPU. No API keys.

### What the engine knows that it will not pretend to know

Every module ships a `KNOWN_LIMITS` block. Three findings worth stating here:

1. **The series' signature look — a hard seam, cold half / hot half — cannot be
   detected from pixels.** Cyan and magenta have near-identical luminance *and*
   near-identical chroma magnitude; only hue differs, so any luminance- or
   intensity-based measure is structurally blind to it. Three detectors were
   written and all three failed. A bilaterally mirrored fork also swaps hue
   across the seam *by construction*, so the left and right hue histograms are
   near-identical by design. **A fork cannot be written in a prompt. It has to
   be shown.** That is the argument for reference-anchored generation, and it
   is measured rather than asserted.
2. **Fold-order detection on rendered images is not shippable.** It returned 2×
   for all 48 references including mandalas that are visibly 6-, 7- and 10-fold.
   So the engine *generates* an exact n-fold reference instead of auditing a
   model's attempt at one.
3. **The 12 axes measure series *membership*, not family *identity*.**
   Single-linkage clustering of the 48 references puts 42 of them into one
   cluster spanning 9 of 13 visual families — purity 0.10. So `rekey.py` does
   not guess which family your reference belongs to. You name it. Choosing the
   reference is a creative act and no statistic here is entitled to make it.

## Credits

Covers: [Leonardo AI](https://leonardo.ai) — see `docs/THIRD-PARTY.md` for the
provenance record and the licence position on the artwork.
Copyright in the 48 covers is the project owner's: they were generated on a
paid Leonardo AI subscription, which vests ownership in the creator.

Storytelling method owes the generic three-phase arc only
(setup / confrontation / resolution). `docs/THIRD-PARTY.md` records the two
repositories consulted, that both are **all-rights-reserved with no licence
file**, and that nothing was vendored from either.
