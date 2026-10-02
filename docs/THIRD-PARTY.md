# Third-party dependencies

The rule this file exists to enforce: **only SPDX-licensed material may be copied
into this project, and only if its licence and pinned commit are recorded here at
the moment of the copy.** No licence file means all rights reserved, which means
nothing may be copied.

Recorded 2026-10-02. Every licence below was read from the installed artefact or
from the GitHub API, not recalled.

---

## 1. GitHub repositories examined — NOTHING VENDORED

Both were consulted as **method references** while designing the story structure.
Neither has been vendored, copied, or quoted, and nothing from either may appear
in a paid product file.

| Repository | SPDX licence | HEAD commit | Pushed | Stars | Verdict |
|---|---|---|---|---|---|
| `TaigaNoeru/Frankkie` | **`null`** | `ee304b79` | 2025-12-30 | 6 | **DO NOT COPY** |
| `Prompt-And-Circumstance/StoryMode` | **`null`** | `90d6877d` | 2026-02-02 | 28 | **DO NOT COPY** |

**Verified two independent ways each**, 2026-10-02:

1. `gh api repos/<owner>/<name> --jq .license.spdx_id` → `null` for both.
2. `gh api repos/<owner>/<name>/git/trees/HEAD?recursive=1` filtered for
   `licen|copying|notice` → **`[]` for both**: no `LICENSE`, `LICENCE`, `COPYING`
   or `NOTICE` file exists anywhere in either tree.

The second check is the decisive one. A missing licence file is the strongest
form of this problem: nothing grants permission, so default copyright applies and
everything is protected.

`Frankkie` is the worse case of the two. It is a 48-module **roleplay prompt
library**, and prompt prose is protectable literary expression — which is exactly
the kind of material this product sells.

### What was taken, and why it is not infringement

One structural idea, reimplemented from scratch:

> A story arc splits into **setup ~33% / confrontation ~34% / resolution ~33%**,
> with a fixed goal list per phase.

This appears in `StoryMode`'s `lib/core/arc-engine.js` as `setupEnd` /
`confrontationEnd` fractions plus a `PHASE_GOALS` table. Three-phase story
structure with proportional boundaries is a general convention in narrative
theory and screenwriting; it is a **method, not expression**. The
implementation in `tools/story_directions.json` and the story drafts was written
independently and shares no wording, no variable names, and no table structure
with the upstream file. No upstream file was read into any prompt that ships.

If that judgement is ever challenged, the fix is to drop the proportional split
and use three unweighted phases — the method survives either way.

### Licence-clean alternatives found

Found by filtering `gh search code` with an explicit `--license` flag, so each has
a real SPDX identifier:

| Repository | SPDX | Stars | Note |
|---|---|---|---|
| `Feed-Scription/openovel` | `Apache-2.0` | 10 | local-first interactive fiction, story state in plain files |
| `hiddenpeopleclub/cuentitos` | `Apache-2.0` | 4 | game narrative engine, probability at its core |

Neither was vendored either. Recorded so the next session does not repeat the
search.

---

## 2. Python packages shipped inside the product

The engine requires all three. All three are permissive; all three permit
redistribution in binary form, which is what a paid download is.

| Package | Version | SPDX (from package metadata) | Vendored? |
|---|---|---|---|
| numpy | 2.5.1 | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | no — installed dependency |
| Pillow | 12.2.0 | `MIT-CMU` | no — installed dependency |
| scipy | 1.18.0 | BSD-3-Clause (confirmed by reading `scipy-1.18.0.dist-info/LICENSE.txt`) | no — installed dependency |

None of the three is copied into the product; the buyer installs them from PyPI
as ordinary dependencies. Their licence terms therefore govern their own
distribution, not this project's, and none obliges this project to publish a
source offer.

**Attribution owed to buyers:** a shipped `THIRD-PARTY` list naming numpy, Pillow
and scipy with their licences, because that is what MIT-CMU and the BSD
licences ask for in any redistribution.

## 3. External binaries called as subprocesses

Not linked, not redistributed. The buyer's machine supplies them.

| Binary | Version | Package licence | Used by |
|---|---|---|---|
| `tesseract` | 5.5.3 | `APACHE` (`pacman -Qi tesseract`) | `postgate.py` OCR gates — **optional**, only `--validate` needs it |

`rosette.py`, `styleprint.py`, `art_ingest.py` and `rekey.py` need **only** numpy,
Pillow and scipy, and `prompts_chatgpt.py` needs only the standard library. The
OCR gate is the single optional external dependency, and the product must say so
rather than pretending the whole thing needs tesseract.

## 4. Fonts

**No font is vendored.** `/usr/share/fonts/liberation/*` is used only by
`postgate.py`'s own negative control (`stamp_typeblock`) to render a word it
knows the answer to, and only on the machine running the control.

Liberation fonts are SIL Open Font License 1.1. Redistribution would require
including the OFL; since nothing is redistributed, the requirement does not
arise. **Do not add a font file to the product** without adding the OFL text.

## 5. Reference images

`~/Dropbox/Lucid_Origin_*.jpg` — 48 images, 1200×1200, all generated in
**Leonardo AI**. Used as the calibration corpus for `styleprint.py` and
`postgate.py`.

Two facts about them are unresolved and both matter:

1. **Leonardo account tier.** Paid subscribers vest IP in the creator on creation;
   free subscribers grant Leonardo the IP and take a non-exclusive royalty-free
   commercial licence — which still permits selling, but never exclusively. The
   48 JPEGs are metadata-stripped (`im.getexif()` returns NONE), so the tier
   cannot be recovered from the files. Only the account history knows.
2. **Public or private mode.** Public generations grant Leonardo a perpetual,
   irrevocable, transferable licence including for training and any commercial
   purpose.

**The architecture side-steps both.** The 48 covers are used as **advertising**
(displayed on the storefront), never resold inside a paid download. Display is
licensed under every scenario; bundling is the only case that creates the
exclusivity problem, and the product does not need it.

One corpus-specific risk is recorded rather than resolved: at least one image
contains a third-party trademark (a Bitcoin ₿ glyph) and several carry invented
credit blocks.

---

## 6. The rule, restated

Before any file, line, prompt, or table is copied into this project:

1. Confirm an SPDX identifier via `gh api repos/<owner>/<name> --jq .license.spdx_id`.
2. Confirm a licence file actually exists via
   `gh api repos/<owner>/<name>/git/trees/HEAD?recursive=1` filtered for
   `licen|copying|notice`.
3. Record name, SPDX, pinned commit SHA and date in a new section here.
4. If either check fails, **do not copy** — reimplement the method instead.

Never paste upstream prompt prose into a file that ships in a paid product.