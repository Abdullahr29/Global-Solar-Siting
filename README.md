# Global solar siting from geospatial embeddings — code

Reproduction code for the accompanying paper (under double-blind review). We learn
solar-siting suitability directly from **AlphaEarth Foundations** 64-dimensional
per-pixel geospatial embeddings: a random forest maps each pixel embedding to
`P(solar)`, trained on land as it appeared two years *before* installation.

This repository lets you (1) run the released model on a bundled pixel sample in
seconds, (2) reproduce the extraction-method comparison, and (3) map and validate
suitability for a whole country inside Google Earth Engine.

> **Scope.** We release the trained model and the code to *apply* and *validate* it.
> Retraining from scratch needs the full multi-terabyte AlphaEarth pixel corpus; that,
> together with the global 10 m suitability layer, is released with the dataset on
> acceptance. The bundled pixel sample reproduces the method *ranking*, not the exact
> full-scale numbers.

## Contents

```
models/rf30_v3_R.joblib     the deployed random forest (30 trees, ~7 MB)
data/px_sample.npz          balanced AlphaEarth pixel sample (40k/class train, 10k/class test)
data/sites_greece.csv       solar-site coordinates for validation (lon, lat, install year)
data/sites_germany.csv
quickstart.py               load the model, score the sample  (no Earth Engine)
benchmark_extractors.py     the 8-method comparison of Table 1 (no Earth Engine)
run_country.py              map + validate a whole country    (needs Earth Engine)
```

## Install

```bash
pip install -r requirements.txt
```
(or `conda create -n solar python=3.11 && conda activate solar && pip install -r requirements.txt`)

## 1. Quick check — no setup (seconds)

```bash
python quickstart.py
```
Loads `rf30_v3_R.joblib`, scores the held-out pixel sample, and prints ROC-AUC / PR-AUC.

## 2. Extraction-method benchmark — no setup (a minute)

```bash
python benchmark_extractors.py
```
Trains the eight extractors from the paper (mean-embedding prototype, logistic probe,
QDA, MaxEnt, random Fourier features, random forest, histogram gradient boosting, MLP)
on the bundled sample and ranks them on the held-out sample, marking which admit native
global-scale inference in Earth Engine. Reproduces the ranking of Table 1.

## 3. Country-scale suitability map — needs Earth Engine

The forest is injected into Earth Engine and applied to a country's AlphaEarth mosaic
entirely server-side (no imagery download), then validated against real sites installed
*after* the imagery year.

**One-time Earth Engine setup (free):**
1. Sign up for Google Earth Engine and create / register a Google Cloud project with the
   Earth Engine API enabled (see https://earthengine.google.com and
   https://developers.google.com/earth-engine/guides/access).
2. Authenticate once:
   ```bash
   earthengine authenticate
   ```
3. Point the scripts at your project:
   ```bash
   export EE_PROJECT=your-earthengine-project-id
   ```

**Run:**
```bash
python run_country.py Greece
python run_country.py Germany --year 2019
```
Outputs go to `outputs/`: a national suitability heatmap (`.png`), a site-vs-random-land
score distribution (`_fig.png`), and the raw scores (`_scores.npz`). The country name must
match the public `USDOS/LSIB_SIMPLE/2017` boundary table; both AlphaEarth
(`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) and the boundaries are public Earth Engine assets.

To validate another country, provide a CSV with columns `lon,lat,year` via `--sites`.

## Data provenance

- **Model:** a random forest trained on 64-d AlphaEarth annual embeddings of existing
  solar sites (sampled two years before installation) versus globally-random non-solar land.
- **Pixel sample (`data/px_sample.npz`):** a small balanced subsample of the held-out
  evaluation pool of AlphaEarth pixel embeddings; `X_sol_*` are solar pixels, `X_neg_*`
  non-solar, each row a 64-d embedding.
- **Site coordinates:** installation locations and years from a public global PV facility
  inventory, used only to validate the maps.

## License

Code: MIT (see `LICENSE`). AlphaEarth embeddings and the site inventory are governed by
their respective providers' terms.
