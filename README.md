<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://ai.google.dev/static/site-assets/images/share-ais-513315318.png" />
</div>

# FaceMax — AI colour & style analysis, with a built-in bias-evaluation pipeline

(Originally generated as an AI Studio app.)

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/fd25ab3a-b240-4bc2-9bad-baa175063c58

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

## ML bias & robustness evaluation (ml/)

FaceMaxx ships with a Python (stdlib-only) evaluation pipeline that generates a
**Model Bias & Robustness Report** — dataset distribution, per-subgroup
performance across skin depth/lighting/camera/quality, confusion structure,
annotator agreement, robustness stability and person-level split integrity.

```bash
npm run bias:selfcheck   # quick end-to-end smoke test
npm run bias:test        # 34 unit tests
npm run bias:demo        # synthetic biased dataset + sample report in ml/data
```

See [ml/README.md](ml/README.md) for the full documentation: annotation
schema, acceptance thresholds, and how to wire your model's predictions in.
