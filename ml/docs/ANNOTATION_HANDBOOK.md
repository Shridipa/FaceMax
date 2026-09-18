# FaceMaxx Annotation Handbook

How photographs become evaluable, defensible training/evaluation data.
Read this before shooting or labelling anything.

> **Prime rule:** labels are human judgements, recorded per annotator.
> Nothing in the pipeline generates or "fills in" a season label.

---

## 1. Capture protocol

Shoot **per person** (one `person_alias` per human being):

| Variation | Minimum | Better |
|---|---|---|
| images per person | 2 | 4 |
| lighting | 1 (natural daylight) | natural + warm indoor + cool indoor |
| makeup | as-is | no-makeup + light-makeup pair |
| camera | 1 phone | 2 devices (e.g. iPhone + Samsung) |
| background | plain, non-colour-casting (white/grey wall) | required |

- Face fills ~60-80% of frame, eyes open, no sunglasses, minimal shadow bands
  across the face.
- **Never** use beauty filters or skin smoothing. If a filter was applied,
  record `makeup = Filter` (or `Skin smoothing`) honestly - the evaluator
  needs those rows.
- Record one **capture-manifest row per photo** while shooting:
  `python -m facemax_color annot manifest` creates
  `data/annotations/capture_manifest.csv`. Fill `file`, `person_alias`,
  `lighting_type`, `camera_type`, `makeup` (and optional `notes`).
  Do not guess these later; unknown = leave empty and fix before import.

**Acquisition targets** (reporting numbers, not quotas to fake):
60+ people, 200+ images, **every** skin-depth band represented, and 40+
images of people near the Deep Autumn / Deep Winter boundary.

---

## 2. Importing photos

```bash
# photos land in a staging folder, manifest filled alongside
python -m facemax_color annot import --manifest data/annotations/capture_manifest.csv --staging path/to/photos
```

- Each file is hashed (sha256) and becomes `IMG-<hash12>` - identical files
  collapse automatically (the same photo twice is imported once, no
  double-counted images).
- Re-running the import is idempotent.
- `person_alias` must be consistent per human being; every image of one
  person shares it. It is never a model feature - it exists to keep people
  in one split.

---

## 3. Independent annotation rounds

```bash
python -m facemax_color annot queue --annotator alice --ids-file ids.txt
# -> data/annotations/queues/round-alice.csv  (open in any spreadsheet editor)
# fill ONLY your own columns; do not look at other annotators' files
python -m facemax_color annot record --round-file data/annotations/queues/round-alice.csv
python -m facemax_color annot merge
```

**Rules:**

- **At least 2 annotators per image; 3 preferred.** Each annotator labels
  independently - no conferring, no seeing others' labels.
- Leave `season` empty rather than guessing. Deferred rows simply stay
  queued. **A blank is data; a wrong confident label is poison.**
- Use `season_secondary` whenever the image sits near a boundary
  (e.g. Deep Autumn / Deep Winter). Ambiguity is preserved, never crushed.
- `confidence_1to5`: 5 = certain, 1 = coin flip.

### Axis definitions (use these, not private definitions)

| Axis | Values | How to decide |
|---|---|---|
| **skin_depth** | Very Fair / Fair / Light / Light-Medium / Medium / Medium-Deep / Deep / Very Deep | Depth/darkness of *skin* only, from the face (not hands), judged under the capture's stated lighting. Ordered shallow -> deep. |
| **undertone** | Cool / Neutral-Cool / Neutral / Neutral-Warm / Warm / Olive | The *hue beneath* the surface: silver-vs-gold jewellery reaction, vein colour (blue/purple = cool, green = warm), whether gold or silver flatters. Olive = slight green/grey cast, often "foundation looks orange or ashy". |
| **chroma** | Muted / Medium / Bright | How saturated/clear the whole colouring is. Muted = dusty, soft (soft grey-green eyes, ash-brown hair). Bright = clear, high-saturation colouring. |
| **contrast** | Low / Medium / High | Difference between hair depth, skin depth and eye depth. High = e.g. very deep hair + very fair skin; Low = everything within a close range. |

### The 12 seasons (temperature x depth x chroma)

| | **Spring (warm, light-bright)** | **Summer (cool, light-muted)** | **Autumn (warm, deep-muted)** | **Winter (cool, deep-bright)** |
|---|---|---|---|---|
| **Light/Muted lean** | Light Spring (light, warm, fresh) | Light Summer (light, cool, soft) | Soft Autumn (warm, muted, medium depth) | Light Winter (cool, clear, medium-light depth) |
| **Warm/Cool core** | Warm Spring (golden, clear, medium) | Soft Summer (cool, muted, medium) | Warm Autumn (golden, rich, medium-deep) | Cool Winter (cool, high-contrast) |
| **Deep/Bright lean** | Bright Spring (warm, bright, high clarity) | Cool Summer (cool, even, soft-medium) | Deep Autumn (warm, deep, rich) | Deep Winter (cool, deepest, vivid) |

**Boundary guidance (the hard ones):**

- **Deep Autumn vs Deep Winter**: both deep; autumn is *warm-rich* (browns,
  olive, gold undertone), winter is *cool-vivid* (blue-red, icy, high
  contrast). If torn, record the secondary - that pair is our highest-value
  boundary candidate.
- **Soft Autumn vs Soft Summer**: both muted; autumn leans golden-green,
  summer leans pink-grey-blue.
- **Bright Spring vs Light Winter**: both bright and light; spring is warm
  (peachy-gold), winter is cool (icy pink/blue).
- **Olive undertone**: often mislabelled Warm. If gold jewellery is *okay*
  but silver is also fine and foundation runs ashy/orange, consider Olive.

### Colours per image

`best_colors`, `neutral_colors`, `caution_colors` take **semicolon-separated
draping colour names** from the fixed 18-colour vocabulary (`burgundy`, `red`,
`orange`, `yellow`, `olive`, `emerald`, `forest_green`, `teal`, `cobalt`,
`lavender`, `pink`, `black`, `white`, `cream`, `navy`, `brown`, `rust`,
`plum`). Any other name is rejected by the validator.

---

## 4. What the tooling enforces mechanically

| Rule | Where |
|---|---|
| Identical files collapse; people never double-counted | `annot import` (sha256) |
| Independent labelling (per-annotator round files) | `annot queue` |
| Vocabulary typos rejected with suggestions, not auto-fixed | `annot record` |
| Missing season = row stays pending (never fabricated) | `annot record`/`merge` |
| Split annotator votes leave season empty + flagged | `annot merge` |
| Axis labels merged only on full agreement | `annot merge` |
| All images of a person land in one split | `audit --splits` / `--regen-splits` |
| Test holdout frozen with tamper-evident lock | `audit --regen-splits` |
| No race/ethnicity fields exist anywhere | schema + tests |
| Synthetic demo data cannot touch real data | demo isolation + tests |

---

## 5. Milestone gates

```bash
python -m facemax_color validate    # per-row schema/vocab/range lint
python -m facemax_color audit --splits   # full data-quality audit + manifests
python -m facemax_color audit --regen-splits  # when audit passes: LOCK the holdout
python -m facemax_color ready       # readiness audit: coverage, disagreement,
                                    #   condition stability, holdout untouchedness
                                    #   -> NOT READY / SMOKE-READY / TRAIN-READY
python -m facemax_color gate        # ML-training gate: OPEN/CLOSED + reasons
```

**The holdout is never touched after locking.** `ready` reports SMOKE-READY
when the gate opens, and TRAIN-READY when the dataset additionally covers all
12 seasons, every skin-depth group and axis label, with high agreement, no
conflicting labels and no unstable capture conditions. Only a SMOKE-READY or
better dataset may enter model wiring - and 40 images is a wiring test, never
a benchmark. Only then does the next milestone (CV features -> baselines ->
12-season classifier) begin.
