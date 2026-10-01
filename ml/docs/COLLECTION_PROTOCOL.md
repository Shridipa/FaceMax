# FaceMaxx Real-Photo Collection Protocol

How real photographs are collected, consented, stored, and prepared for
annotation — **before** `ANNOTATION_HANDBOOK.md` takes over.

This document governs **real** photos only. Synthetic/demo data lives in
`ml/data/demo/`, is governed by `ml/docs/ML_TRAINING.md`, and is never mixed
with real data. Demo artifacts are labelled `SYNTHETIC DEMO` and are not
performance benchmarks.

Companion documents (read in this order):

1. `ml/data/DATASET_REGISTRY.md` — what external datasets exist and their
   licence status (incl. CelebAMask-HQ, Fitzpatrick17k, FFHQ).
2. this file — collection, consent, and capture discipline;
3. `ml/docs/ANNOTATION_HANDBOOK.md` — labelling rules and CLI.
4. `ml/docs/ML_TRAINING.md` — training, calibration, holdout, and the
   real-data readiness gate.

> **Prime rule:** no photograph enters the pipeline without a recorded
> consent basis and a pseudonymous `person_alias`. An image with unknown
> provenance is rejected, not "fixed later".

---

## 1. Consent & rights (blocking, not optional)

| Requirement | Rule |
|---|---|
| Informed consent | Every photographed person gets a plain-language explanation of: what is collected (face photos), why (skin-tone model training/evaluation), where it is stored (this repo's `ml/data/`, private), and that they can withdraw. |
| Consent record | One row per person in `data/consent/consent_log.csv` (or equivalent private ledger outside git): `person_alias`, `consent_given` (yes/no), `date`, `method` (written/verbal+note), `withdrawn` (blank until withdrawal), `notes`. |
| Withdrawal | On withdrawal, all images of that `person_alias` are deleted from staging, `data/raw/`, features, and annotation CSVs, and the ledger row is marked. Re-import must re-consent. |
| Minors | No minors unless a guardian consents **and** the project explicitly opts in — default is **exclude**. |
| Public-figure / web-scraped images | **Out of scope.** We do not scrape social media or use scraped face datasets for the real pipeline. External licensed datasets are handled separately per the registry. |
| Retention | Photos are kept only while needed for training/evaluation of this project; delete on project end or on request. `ml/data/raw/` is git-ignored — nothing leaves the machine by default. |
| Attribution/licence | No third-party copyrighted material (magazine shots, stock photos without licence, AI-face sites with unclear terms). |

Consent is a **known gap in the current repo** (documented, not hidden): the
ledger file and workflow above are the fix. Until a consent ledger exists for
a batch, that batch must not be imported.

## 2. Capture conditions

Follow `ANNOTATION_HANDBOOK.md` §1. Summary with the colour-science details
the handbook omits:

- **Light:** diffuse daylight is the default (overcast window or open shade).
  Avoid direct sun (specular hotspots), coloured walls/curtains (colour cast),
  and mixed light sources in one frame.
- **White balance:** lock or fix per session; do not let "auto WB" drift
  between photos of one person. If the phone allows, shoot RAW/Pro mode.
- **Colour chart:** if a chart (e.g. ColorChecker) is available, shoot one
  frame with the chart next to the face per lighting condition — this lets us
  verify/derive a colour correction later. A chart is **strongly preferred,
  not mandatory**; when absent, record the lighting_type honestly. Do **not**
  fabricate chart values when no chart was shot.
- **Framing:** face fills ~60–80% of frame, eyes open, no sunglasses, no
  heavy shadow bands. Hair may cover skin only if annotated honestly.
- **Never** apply beauty filters or skin smoothing. If unavoidable, mark
  `makeup = Filter` / `Skin smoothing` in the manifest (handbook §1).
- **Multi-image per person:** ≥2 (better 4) under different lighting/devices —
  this is what makes person-grouped evaluation meaningful.
- **Screen/monitor lighting** photos are acceptable but must be tagged as
  their own `lighting_type`; never silently mixed with daylight rows.

## 3. Person identity & grouping

- One `person_alias` per human being, consistent across all their images.
  Pseudonyms only (e.g. `P-0007`) — never names, handles, or contact info
  in any file that could leave the machine.
- `person_alias` is never a model feature. It exists so person-grouped splits
  (`StratifiedGroupKFold`, frozen manifests in `data/splits/`) keep one
  person entirely in train or entirely in evaluation. That is the anti-leakage
  backbone; a person split across sides silently inflates metrics.
- Every capture-manifest row must carry the alias (handbook §1: fill it
  during the shoot, never reconstruct from memory).

## 4. Coverage targets (reporting, never quotas to fake)

Check progress at any time:

```bash
python -m facemax_color annot coverage
```

| Metric | Dev target | Note |
|---|---|---|
| people | 60+ | distinct `person_alias` |
| images | 200+ | ≥2 per person |
| skin-depth bands | **every** band | incl. Very Deep — zero-support demo classes must not repeat on real data |
| DA/DW boundary candidates | 40+ | `season_secondary` preserved, not crushed |

These are **development targets**. The real-data **training gate** has its own
minimums (see `ML_TRAINING.md` and `python -m facemax_color gate`); this
protocol does not modify them, and no documentation can. Balance is achieved
by recruiting, never by duplication or label invention.

## 5. Rejection criteria (reject at import, not at training)

Reject (and log) any image that is:

- missing from the consent ledger, or from a withdrawn person;
- unusable for colour extraction: heavy blur, extreme crop (<~40% face),
  strong colour cast that WB cannot fix, or a documented filter applied and
  not honestly tagged;
- duplicated (sha256) — the importer collapses these automatically, but flag
  reshoots of the same pose as distinct images only if they add variation;
- missing manifest fields (`person_alias`, `lighting_type`, `camera_type`,
  `makeup`) — leave empty at capture, fix before `annot import`.

Rejected files are removed from staging; rejection counts are reported in the
batch notes — nothing is silently dropped.

## 6. Annotator protocol (summary — full rules: `ANNOTATION_HANDBOOK.md`)

- ≥2 annotators per image (3 preferred), fully independent rounds; no
  conferring, no seeing other annotators' files.
- Workflow per annotator:
  ```bash
  python -m facemax_color annot queue --annotator alice --ids-file ids.txt
  # fill ONLY your own columns in data/annotations/queues/round-alice.csv
  python -m facemax_color annot record --round-file data/annotations/queues/round-alice.csv
  python -m facemax_color annot merge
  ```
- Leave `season` **blank rather than guessing** — a blank is data; a wrong
  confident label is poison. Use `season_secondary` near boundaries;
  record `confidence_1to5` honestly.
- Axis definitions are frozen in the handbook — never private definitions.
- Disagreement is preserved and surfaced by merge/agreement tooling; it is
  resolved by protocol (secondary label, confidence, possible re-look), never
  by one annotator overwriting another.

## 7. Batches & provenance

A **batch** = photos collected together under one consent effort and one
capture session set. Keep per batch: date, shooter, devices, locations,
lighting conditions, chart availability, consent ledger reference, rejection
counts. Store batch notes next to the staging folder (outside git or under
`data/` as appropriate). Provenance lets any future reviewer reconstruct
exactly how an image entered the pipeline.
