# Website

Interactive write-up of the bank marketing project, built with Vite + React + TypeScript and
deployed as a static site. There is no backend: the trained LightGBM pipeline is exported to
JSON and traversed in the browser.

## Run locally

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check + production build into dist/
npm run preview    # serve the production build
```

## Deploy to Vercel

The site lives in this subdirectory, so point Vercel at it:

| Setting | Value |
|---|---|
| Root Directory | `web` |
| Framework Preset | Vite |
| Build Command | `npm run build` (default) |
| Output Directory | `dist` (default) |

Either import the repo at [vercel.com/new](https://vercel.com/new) and set **Root Directory** to
`web`, or from this folder:

```bash
npx vercel        # preview deployment
npx vercel --prod # production
```

`vercel.json` sets long-lived caching for hashed assets and a shorter TTL for `model.json`.

## Where the numbers come from

Nothing on the page is hand-typed. Every figure is generated from `raw_data.csv` and the saved
model by the scripts in [`../tools`](../tools), which write into `src/data/` and `public/`:

| Script | Writes | Contains |
|---|---|---|
| `export_model.py` | `public/model.json`, `src/data/baseline.json` | 250-tree booster as flat arrays, encoder layout, scaler stats |
| `export_insights.py` | `src/data/eda.json`, `src/data/metrics.json` | EDA aggregates, benchmark table, ROC/PR curves, leakage exhibit, the 8,040 test-set scores that drive the threshold lab |
| `export_segments.py` | `src/data/segments.json` | K-Prototypes k=4 personas and the elbow curve |
| `make_fixtures.py` + `check_parity.mjs` | — | Asserts the browser scorer matches `predict_proba` to 1e-6 |

Regenerate everything:

```bash
cd ../tools
python export_model.py
python export_insights.py
python export_segments.py     # ~5 min: refits K-Prototypes on 41,188 rows
python make_fixtures.py && node check_parity.mjs
```

## Client-side inference

`src/lib/lgbm.ts` implements the booster traversal. `tools/export_model.py` flattens each tree
into `feature[] / threshold[] / left[] / right[] / leaf[]` arrays (leaf `k` encoded as `-(k+1)`),
then asserts in Python that the flattened form reproduces scikit-learn's output before writing
the file. `check_parity.mjs` repeats that assertion from Node against the shipped JSON, including
the `unknown` category path, so a regression in the encoder shows up as a failing check rather
than a wrong number on the page.

## Structure

```
src/
  components/          section components, one per page section
    charts/            hand-built SVG and CSS charts
  lib/
    lgbm.ts            booster traversal + leave-one-out attribution
    threshold.ts       score histograms behind the threshold lab
    profiles.ts        predictor presets and macro regimes
    ui.ts              formatting, reveal/measure hooks
  data/                generated JSON (do not edit by hand)
public/model.json      the exported booster, fetched on demand
```
