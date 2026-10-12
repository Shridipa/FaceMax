# FaceMaxx

FaceMaxx is a web app for personal colour analysis and outfit styling. It combines a React website with a Node.js API server and includes Python tools for colour-model development and bias/robustness evaluation.

## What you can do

- Upload a photo for personal colour analysis and view the results.
- Use Outfit Studio to analyse an outfit photo, compare outfit options, or build an outfit from saved wardrobe items.
- Read practical styling information on the About and Privacy pages.
- Run the included Python colour-analysis and model-bias evaluation tools.

## Requirements

- Node.js (a current LTS version is recommended) and npm.
- A Gemini API key for AI-powered analysis.
- Python 3 for the optional ML tools. The bias evaluation and self-check use only the Python standard library. The colour-model tools may require the packages listed in `requirements.txt`.

## Run the website locally

Open a terminal in the project folder (the folder containing `package.json`), then install the JavaScript dependencies:

```bash
npm install
```

Create a local environment file by copying `.env.example` to `.env`. For example, in PowerShell:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace `MY_GEMINI_API_KEY` with your Gemini API key. `APP_URL` is for a hosted deployment and is not normally needed for local development. Keep `.env` private; do not commit API keys.

Start the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The server serves the website and its API endpoints together. Without a valid Gemini API key, features that call Gemini will not complete successfully.

## Build and run the production server

```bash
npm run build
npm start
```

The build output is written to `dist/`. The production server listens on port `3000`.

## Tests and checks

Run the TypeScript check:

```bash
npm run lint
```

Run the outfit feature tests:

```bash
npm run test:outfit
```

Run the Python bias-evaluation tests:

```bash
npm run bias:test
```

## Python ML tools

The Python project lives in `ml/`. Run these commands from the project root:

```bash
npm run bias:selfcheck
npm run bias:demo
```

The self-check runs a quick end-to-end check. The demo generates synthetic data and a sample bias report under `ml/data`. To run the Python test suite directly instead:

```bash
cd ml
python -m unittest discover tests
```

On Windows, if `python` is not recognized, use `py` in place of `python`.

The colour-model tools are also available through npm scripts:

```bash
npm run colour:demo
npm run colour:audit
npm run colour:validate
npm run colour:ready
npm run colour:gate
npm run colour:prepare
npm run colour:train
npm run colour:evaluate
npm run colour:train-demo
```

Install their Python dependencies when needed:

```bash
python -m pip install -r requirements.txt
```

See [ml/README.md](ml/README.md) for the dataset and annotation workflow, evaluation reports, training gates, and the meaning of each ML command.

## Project structure

| Path | Purpose |
| --- | --- |
| `src/` | React website, pages, components, and browser-side services |
| `api/` | API handlers and shared server-side analysis code |
| `server.ts` | Local Express server, API routes, and Vite development setup |
| `ml/facemax_bias/` | Bias and robustness evaluation |
| `ml/facemax_color/` | Colour feature extraction, datasets, and model workflows |
| `tests/` | JavaScript/TypeScript feature tests |
| `dist/` | Generated production build output |

## Main website pages

- `/` — introduction and app entry point
- `/analyze` — start a personal colour analysis
- `/results` — view analysis results
- `/outfit-studio` — analyse, compare, and assemble outfits
- `/about` — project information
- `/privacy` — privacy information

## Configuration and deployment

The server reads environment variables from `.env` during local development. `GEMINI_API_KEY` is needed by AI-powered API features. For deployment, configure secrets in the hosting provider rather than committing them to the repository. The project includes `vercel.json` for Vercel configuration; deployment should also provide the required API key and Python runtime dependencies for the colour-analysis function.

## Privacy

Do not upload photos or provide data that you do not have permission to use. Review the app's [Privacy page](/privacy) before using it.
