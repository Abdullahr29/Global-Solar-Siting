# Reproducing the paper

Every result in *Global Solar Siting from Geospatial Embeddings* (Rehman, Paletta & Song, NeurIPS 2026 workshop),
what in this repository reproduces it, and what it needs. Times are for a 12-core laptop.

| Paper | Here | Needs | Expected |
|---|---|---|---|
| Fig. 1 pipeline; the global layer | `scripts/predict.py`, `scripts/get_dataset.py` | model; internet (AEF) | the released v1.0 map (AEF 2025) is reproduced pixel-for-pixel within ±1 code on ~1e-6 of pixels |
| Table 1, method benchmark (pixel columns) | `scripts/benchmark.py` | bundled `px_sample.npz` | same ranking; values below |
| Retraining the ensemble (Appendix B recipe) | `scripts/train.py` | bundled sample (or your pool); PyTorch | held-out ROC ~0.92 on the sample |
| Fig. 2b, Greece national map and its validation sites | `predict.py --country Greece --year 2019` + `validate.py` | model; internet; ~45 min | the map and protocol of Fig. 2b; ROC-AUC 0.924 on the paper's 217 held-out sites (paper 0.894 with the train-split model: [why they differ](#forward-validation-predictpy--validatepy)) |
| Table 3 / App. D, 13-country forward validation | the same for each country with `data/validation/sites_<country>.csv` | model; internet; hours to days for the largest countries | the protocol; exact Table 3 values need the train-split model (not released), see below |
| Fig. 3, Table 5, App. H, MCDA comparison | `scripts/mcda_example.py` (Greece) | bundled `data/mcda_example/` | gap ~+0.24 ROC-AUC (paper Greece +0.233) |
| App. E, within-country percentiles | `mosaic_*_pct.vrt` outputs; `docs/dataset.md` | — | ROC-AUC unchanged by the transform |

## `scripts/benchmark.py` (bundled sample, ~15 s)

```
train 72,000 px | test 60,000 px (PR-AUC chance 0.167)

  Mean-embedding cosine    ROC 0.705 | PR 0.319   (paper 0.702 / 0.318)
  QDA                      ROC 0.856 | PR 0.517   (paper 0.854 / 0.518)
  Logistic probe           ROC 0.868 | PR 0.555   (paper 0.866 / 0.551)
  MaxEnt                   ROC 0.876 | PR 0.578   (paper 0.874 / 0.577)
  Random Fourier features  ROC 0.892 | PR 0.626   (paper 0.891 / 0.628)
  Random forest            ROC 0.892 | PR 0.616   (paper 0.914 / 0.696)
  Hist. gradient boosting  ROC 0.911 | PR 0.682   (paper 0.919 / 0.713)
  MLP                      ROC 0.918 | PR 0.717   (paper 0.924 / 0.728)
  RF+MLP ensemble          ROC 0.921 | PR 0.718   (paper 0.932 / 0.752)

best: RF+MLP ensemble (ROC 0.921) | ranking identical to Table 1: yes
```

The paper trains every method on a 1.4 M-pixel balanced subsample of the full pool; the bundled training sample has
72,000 pixels, so the high-capacity models (RF, boosting, MLP, ensemble) score 0.006-0.022 lower here, while the
low-capacity ones match to within ~0.004. Exact values can differ in the third decimal across platforms and library
versions. The paper's other Table 1 columns (13-country forward validation, TZ-SAM) need the full validation
embeddings; the country column can be reproduced map by map with `predict.py` + `validate.py`.

## `scripts/train.py --preset sample` (~15 s, needs PyTorch)

```
  held-out RF               ROC-AUC 0.903 | PR-AUC 0.651
  held-out MLP (5 seeds)    ROC-AUC 0.928 | PR-AUC 0.741
  held-out RF+MLP ensemble  ROC-AUC 0.925 | PR-AUC 0.715
```

At this sample size the MLP alone edges the ensemble; on the full ~45 M-pixel pool the ensemble is best on every
held-out test (paper Table 1). The release recipe itself (`--preset release`) is tuned for the full pool.

## Forward validation (`predict.py` + `validate.py`)

Map year Y, validation sites installed in Y + 2 or later (absent from the scored imagery), against uniformly sampled
pixels of the same region. Measured while testing this repository (all of Greece, AEF 2019, 131,415 km², 46 min with
the faster maths library):

```
python scripts/predict.py --country Greece --year 2019
python scripts/validate.py --run outputs/greece_2019 --sites data/validation/sites_greece.csv
  greece | AEF 2019 map | 1074 sites installed >= 2021 vs 20,000 random pixels
    ROC-AUC 0.937 | median site percentile 96 | sites in top 20% / 10% / 5% of the region: 96% / 77% / 55%
      in paper training chips                      n=  857  ROC-AUC 0.941
      paper's held-out set (test + unseen chips)   n=  217  ROC-AUC 0.924
```

**Which model a number comes from matters.** The paper's evaluations (Table 3, Fig. 2b) use the ensemble trained on
the 80 % training split and score only sites outside its training chips (in Greece, 221 sites; ROC-AUC 0.894). The
released model is the same recipe refit on *all* scenes, so it has also seen those held-out sites. The bundled site
files therefore carry a `paper_split` column (`train` / `test` / `unseen`: whether the site lies inside a training-split
chip, a test-split chip, or neither; same rule as the paper's leakage check), and `validate.py` reports the result by
group. With the released model:

- almost every site (all but the few `unseen` ones) is one the model was trained on, so the numbers show how well the
  released map ranks known solar, not forward skill, and run higher than Table 3 (Greece 0.924 on the held-out set vs
  0.894 in the paper). This matches the project's own check of the released ensemble on the paper's validation points
  (Greece 0.923), so the difference is the model, not this code;
- for an independent test of the released model, use sites outside the inventory: on TZ-SAM sites absent from it the
  released ensemble scores Greece 0.896 (train-split model 0.894) and 0.894 over all 13 countries pooled (0.891).
  TZ-SAM is CC BY-NC and not bundled; download it and pass any `lon, lat, year` CSV to `--sites`. Sites built after
  the inventory (2025 onward) also work.

A smaller example (Western Macedonia, 7,500 km², ~3 min): `predict.py --bbox 21.3 40.0 22.3 40.8 --year 2019` then
`validate.py` with `sites_greece.csv` gives 115 sites, ROC-AUC 0.921 (114 of them in paper training chips).

## MCDA (`scripts/mcda_example.py`, ~2 s)

```
  RF+MLP ensemble (AEF 2019)   ROC-AUC 0.882
  MCDA Richards WLC            ROC-AUC 0.643
  MCDA Richards TOPSIS         ROC-AUC 0.626
  MCDA Chen WLC                ROC-AUC 0.546
  MCDA Chen TOPSIS             ROC-AUC 0.547
best MCDA: MCDA Richards WLC (0.643) | ensemble 0.882 | gap +0.239
```

This uses the paper's figure-resolution raster version (~1 km cells, GPVF sites, no WDPA by default). The paper's
Table 5 scores point samples at 10 m with GPVF + TZ-SAM sites and WDPA (Greece: 0.916 vs 0.683, gap +0.233); the
size of the gap is the comparison that matters.

## What is not in this repository

The full training pools (~45 M labelled pixels extracted from 38,591 sites and 13,440 non-solar parcels) and the
production pipeline that scored the globe on a cluster are not included: the pools are too large to redistribute,
and the production code is cluster-specific. `scripts/predict.py` runs the same model on the same inputs and
reproduces the released map; `docs/methods.md` describes how the global layer was produced.
