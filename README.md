# Global solar siting from geospatial embeddings — code

Reproduction code for the accompanying paper (under double-blind review). We learn
solar-siting **suitability** directly from **AlphaEarth Foundations** 64-dimensional
per-pixel geospatial embeddings: a random forest maps each pixel embedding to
`P(solar)`, trained on land as it appeared two years *before* installation and global non-solar land, so it
captures the pre-existing landscape rather than the panels.

![Global solar-suitability preview](figures/global_suitability_map.png)

*Global suitability from the deployed random forest applied to the AlphaEarth 2025 mosaic (100 m preview; brighter = higher predicted P(solar)). A downsampled placeholder for the full 10 m dataset released on acceptance.*

> ⚠️ **Preliminary artifact for peer review.** This is a minimal, self-contained subset
> of the code with a small data sample — enough to reproduce the paper's key results and
> apply the model. **The full codebase and the final global 10 m suitability dataset will
> be released upon acceptance of the paper.** Retraining from scratch needs the full
> multi-terabyte AlphaEarth pixel corpus; the
> bundled pixel sample here reproduces the method *ranking*, not the exact full-scale
> numbers.

## How it works

- **Input.** Each land pixel is a 64-d AlphaEarth annual embedding — a dense vector that
  fuses spatial, spectral, and temporal Earth-observation signals into one global,
  per-pixel representation.
- **Model.** A random forest is trained to separate embeddings of *solar* land (real
  sites, sampled two years before installation) from *non-solar* land (globally-random
  parcels), outputting a per-pixel suitability score `P(solar)`.
- **Global inference.** The trained forest is injected into Google Earth Engine as
  decision-tree strings, so it can score an entire country's (or the world's) AlphaEarth
  mosaic server-side, with no imagery download.
- **Validation.** Maps are checked against real solar sites installed *after* the imagery
  year — genuinely future sites the model never saw — plus random land for the score
  distribution.

## Contents

```
models/rf30_v3_R.joblib     the deployed random forest (30 trees, ~7 MB)
data/px_sample.npz          balanced AlphaEarth pixel sample (40k/class train, 10k/class test)
data/sites_<country>.csv    solar-site coordinates for validation (lon, lat, install year);
                            bundled for the paper's 13 forward-validation countries
quickstart.py               load the model, score the sample        (no Earth Engine)
benchmark_extractors.py     the 8-method comparison of Table 1       (no Earth Engine)
run_country.py              map + validate a whole country           (needs Earth Engine)
```

## Setup

**Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate solar-suitability
```

**Or pip:**
```bash
pip install -r requirements.txt
```

## 1. Quick check — no setup (seconds)

```bash
python quickstart.py
```
Loads `rf30_v3_R.joblib`, scores the held-out pixel sample, and prints ROC-AUC / PR-AUC —
a one-command confirmation the model works.

## 2. Extraction-method benchmark — no setup (about a minute)

```bash
python benchmark_extractors.py
```
Trains the eight extractors compared in the paper (mean-embedding prototype, logistic
probe, QDA, MaxEnt, random Fourier features, random forest, histogram gradient boosting,
MLP) on the bundled sample and ranks them on the held-out sample, marking which admit
native global-scale inference in Earth Engine. Reproduces the ranking of Table 1 and why
the random forest is the deployed model.

## 3. Country-scale suitability map — needs Earth Engine

The random forest is applied to a country's AlphaEarth mosaic entirely server-side, then
validated against real sites installed after the imagery year.

**One-time Earth Engine setup (free):**
1. Sign up for Google Earth Engine and create/register a Google Cloud project with the
   Earth Engine API enabled — see https://earthengine.google.com and
   https://developers.google.com/earth-engine/guides/access .
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
python run_country.py Spain --year 2019      # year argument means model runs on AEF 2019, validation on 2021+
python run_country.py Chile          # any country renders a map, validated where sites are bundled
```
Validation site files are bundled for the paper's thirteen forward-validation countries —
Chile, Colombia, Japan, Greece, Philippines, United States, China, Spain, South Africa,
Malaysia, Germany, India and Poland — so those produce the validated site-vs-random figure
out of the box; any other country still renders its suitability map (validation skipped).

Outputs land in `outputs/`: a national suitability heatmap (`.png`), a
site-vs-random-land score distribution (`_fig.png`), and the raw scores (`_scores.npz`).
The country name must match the public `USDOS/LSIB_SIMPLE/2017` boundary table; both
AlphaEarth (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) and the boundaries are public Earth
Engine assets, so no extra data is needed. Any country in the boundary table works: with
a bundled or `--sites` CSV (columns `lon,lat,year`) the map is validated against real
future sites, and without one it still renders the suitability map and skips validation.

## Data provenance

- **Model:** a random forest trained on 64-d AlphaEarth annual embeddings of existing
  solar sites (sampled two years before installation) versus globally-random non-solar land.
- **Pixel sample (`data/px_sample.npz`):** small balanced subsamples of the training and
  held-out evaluation pools of AlphaEarth pixel embeddings; `X_sol_*` are solar pixels,
  `X_neg_*` non-solar, each row a 64-d embedding. The test sample is class-balanced, so
  PR-AUC reads higher than the paper's (~1:5 imbalanced) pool; ROC-AUC is comparable.
- **Site coordinates:** installation locations and years from a public global PV facility
  inventory (https://zenodo.org/records/18794231), used only to validate the maps.

## License

Code: MIT (see `LICENSE`). The AlphaEarth embeddings and the site inventory are governed
by their respective providers' terms.
