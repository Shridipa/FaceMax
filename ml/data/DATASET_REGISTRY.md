# FaceMaxx DATASET REGISTRY

Every external dataset considered for training or evaluation is tracked here
BEFORE download. No dataset enters `data/raw/` HUMAN_VERIFIED state without a
completed block and a license that permits the intended use.

**Glossary of the `verified:` markers below**

- `verified:` facts confirmed by Buffy (agent) against the official page on
  2026-09-15, with the source URL cited in the block. These are research aids,
  NOT approvals.
- `human-check:` items only a human can certify (terms acceptance, download,
  checksum comparison, distribution intent). Filling these is the approval act.

Rules:
- CelebAMask-HQ: face PARSING (skin/hair/eyes/lips masks) only - never seasonal ground truth.
- FFHQ: preprocessing/robustness research only; respect per-image licensing.
- Fitzpatrick17k (or equivalent): skin-DEPTH robustness evaluation only where the
  license permits; Fitzpatrick labels are NOT undertone/season labels.
- FaceMaxx custom annotated set: PRIMARY source for undertone, depth, chroma,
  contrast, temperature, season, secondary season, confidence.
- FaceMaxx colour-harmony/draping set: PRIMARY source for colour compatibility
  (see annotations/harmony_scores.csv).
- The 12-season colour labels come ONLY from the FaceMaxx custom annotated set.
  No external dataset may be auto-labelled with seasons.

---

## 1. CelebAMask-HQ

- **official source:** CUHK MMLab (The Chinese University of Hong Kong)
- **URL:** https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html (official page); mirror: https://github.com/switchablenorms/CelebAMask-HQ
- **license:** non-commercial research purposes ONLY (verified: official page states "available for non-commercial research purposes only"; use restricted to non-commercial research and educational purposes)
- **commercial use:** no (verified: explicit restriction on the official page)
- **redistribution:** no (verified: reproducing/copying/selling/trading/reselling restricted; share the download link, not the files)
- **attribution required:** yes - cite the CelebAMask-HQ paper (Lee et al., CVPR 2020 "MaskGAN")
- **intended use:** face parsing masks (skin/hair/eyes/lips) for segmentation supervision; NEVER seasonal ground truth
- **known limitations:** constricted pose/lighting diversity; celebrity skew; NOT seasonal labels
- **ingestion provenance (recorded 2026-09-28):** official archive downloaded via gdown from the project README's Google Drive id 1badu11NqxGf6qM3PTTooQDJvQbejgbTv (verified against https://github.com/switchablenorms/CelebAMask-HQ README); sha256 5dc46a36cba18d216b9a031902508b1398e7e4c545bbc9699bec32412c530ca6 (3,153,930,546 bytes); receipt data/ingest/CelebAMask-HQ.receipt.json (verify --rehash OK); stored at data/raw/external/CelebAMask-HQ/; use restricted to non-commercial research - masks are geometry only, NEVER undertone/season ground truth; no redistribution.
- **human-check: terms reviewed and accepted:** [x] (2026-09-28, via project confirmation dialog; non-commercial research use only)
- **downloaded:** no
- **checksum (sha256):**
- **preprocessing version:**

---

## 2. FFHQ (Flickr-Faces-HQ)

- **official source:** NVIDIA
- **URL:** https://github.com/NVlabs/ffhq-dataset
- **license:** research use; dataset hosted "for research purposes"; images retain their original Flickr licences - verify per image before any redistribution (verified: dataset card; NVIDIA hosts as research artifact)
- **commercial use:** no as a dataset (per-image rights vary - a per-image check is required for anything beyond internal research use)
- **redistribution:** do not redistribute images; distribute metadata/scripts and refer users to the official source
- **attribution required:** yes - cite the FFHQ paper (Karras et al., 2019) and respect the original Flickr photographer
- **intended use:** face preprocessing/robustness (detection, alignment, white-balance validation)
- **known limitations:** licensing heterogeneity; not undertone/season labeled; GAN-curation biases possible
- **human-check: terms reviewed and accepted:** [ ]
- **downloaded:** no
- **checksum (sha256):**
- **preprocessing version:**

---

## 3. Fitzpatrick17k (skin-tone benchmark)

- **official source:** Groh et al., Harvard/MIT (github.com/mattgroh/fitzpatrick17k)
- **URL:** https://github.com/mattgroh/fitzpatrick17k (annotations CSV; images via the official request form or original atlas URLs)
- **license:** CC BY-NC-SA 3.0 Unported (verified: "Licensing" section of the official README; original images from Atlas Dermatologico and DermaAmin)
- **commercial use:** no (verified: NonCommercial clause; ShareAlike applies to derivatives)
- **redistribution:** ShareAlike - derivatives must carry the same license; prefer referring users to the official source; image access via official request form
- **attribution required:** yes - cite Groh et al. CVPR 2021 (fitzpatrick17k paper) and Groh et al. CSCW 2022
- **intended use:** skin-DEPTH robustness EVALUATION ONLY; Fitzpatrick types I-VI are sun-reactivity/dermatology labels and are NOT undertone/season/chroma labels
- **known limitations:** dermatoscopic/clinical images (not portraits); Fitzpatrick labels subjective on diseased skin; three-level fitzpatrick scale in csv (-1 = unknown)
- **ingestion provenance (recorded 2026-09-28):** official annotations CSV downloaded from https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv; sha256 851dfaabf1791c2a8b1c2957619f3e0673cf0002f4ce924a5c02d185c7d4a1cd (4,104,341 bytes); receipt data/ingest/Fitzpatrick17k.receipt.json (verify --rehash OK); stored at data/raw/external/Fitzpatrick17k/fitzpatrick17k.csv; 16,577 rows validated (16,012 labelled / 565 unknown) with zero FaceMaxx-axis contamination; IMAGE FILES NOT OBTAINED (official request form required); labels kept as fitzpatrick_scale/three_partition/nine_partition - never converted to undertone/skin_depth/season.
- **human-check: terms reviewed and accepted:** [ ]
- **downloaded:** no
- **checksum (sha256):**
- **preprocessing version:**

---

## 4. FaceMaxx custom annotated dataset (PRIMARY)

- **official source:** this project
- **URL:** data/raw + data/metadata/facemaxx_color_annotations.csv
- **license:** project-owned
- **commercial use:** yes
- **redistribution:** no
- **attribution required:** n/a
- **intended use:** undertone / skin depth / chroma / contrast / temperature / season / secondary season / confidence - the ONLY source of 12-season ground truth
- **known limitations:** size grows with annotation effort; annotator agreement tracked per image; capture protocol defined in docs/ANNOTATION_HANDBOOK.md
- **downloaded:** n/a (self-collected)
- **checksum:** n/a
- **preprocessing version:**

---

## 5. FaceMaxx colour-harmony / draping dataset (PRIMARY)

- **official source:** this project
- **URL:** data/annotations/harmony_scores.csv
- **license:** project-owned
- **commercial use:** yes
- **redistribution:** no
- **attribution required:** n/a
- **intended use:** per-colour compatibility scores (burgundy ... plum) for the harmony model
- **known limitations:** subjective; multi-annotator scores stored long-format
- **downloaded:** n/a (self-collected)
- **checksum:** n/a
- **preprocessing version:**
