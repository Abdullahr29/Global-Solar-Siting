# Bundled data

Small subsets, just enough to run the examples and reproduce the paper's comparisons. Everything else is read from
public sources at run time (see [`docs/data_sources.md`](../docs/data_sources.md)).

## `px_sample.npz` — AlphaEarth pixel sample (32 MB)

64-d AlphaEarth Foundations (AEF) annual embeddings, de-quantised (float32), drawn from the paper's training and
held-out pixel pools (embedding two years before installation for solar pixels):

| array | pixels | |
|---|---|---|
| `X_sol_train`, `X_neg_train` | 12,000 / 60,000 | solar / non-solar, from the training split |
| `X_sol_test`, `X_neg_test` | 10,000 / 50,000 | solar / non-solar, from the held-out test split (scenes disjoint from training) |

Solar pixels lie inside delineated facility footprints (GPVF inventory, below); non-solar pixels come from
globally random parcels at least 5 km from any known solar facility (GPVF or TZ-SAM). Both splits keep the paper's
~1:5 solar : non-solar prevalence, so PR-AUC is comparable with Table 1 (ROC-AUC does not depend on prevalence).
AEF embeddings: produced by Google and Google DeepMind ([licence](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)).

## `validation/sites_<country>.csv` — solar-site points (13 countries, 105k rows)

Columns `lon, lat, year` (WGS84; installation year) for the paper's 13 forward-validation countries: Chile, China,
Colombia, Germany, Greece, India, Japan, Malaysia, the Philippines, Poland, South Africa, Spain, the United States.
One row per facility in the inventory for that country (the inventory's own `longitude`, `latitude` and `year`
fields, years 2017-2024). Use rows with `year >= map year + 2` for forward validation (as `scripts/validate.py`
does by default).

Column `paper_split` records whether the site lies inside a chip of the paper's training split (`train`), only inside
test-split chips (`test`), or in no chip (`unseen`), by the paper's leakage rule (site within ±1.3 km of a chip
centre). The paper's held-out evaluations use `test` + `unseen`; the released all-data model was trained on both the
`train` and `test` chips, so these files measure its fit to known solar rather than forward skill (see
[`docs/reproduce.md`](../docs/reproduce.md#forward-validation-predictpy--validatepy)).

**Source:** the Global PV Facility (GPVF) inventory — Song, R., Yin, F., et al. *Global Facility-Level Solar
Photovoltaic Inventory with Energy Generation and Loss Estimates* (2026), Zenodo,
[doi:10.5281/zenodo.18794231](https://doi.org/10.5281/zenodo.18794231). Licence: **CC BY 4.0**. Only the three
columns needed are redistributed here; please cite the source if you use them.

The paper additionally validated against TZ-SAM (TransitionZero Solar Asset Mapper, Q1 2026 release). TZ-SAM is
licensed **CC BY-NC 4.0** and is not bundled; download it from TransitionZero
([Q1 2024 release on Zenodo](https://doi.org/10.5281/zenodo.11368204)) and pass any `lon, lat, year` CSV to
`scripts/validate.py --sites`.

## `mcda_example/greece_0p01deg.npz` — MCDA example for Greece (~1 km grid)

The layers of the paper's raster MCDA benchmark on a 0.01° grid over Greece (Natural Earth outline), NaN outside
the country:

| layer | source (licence) |
|---|---|
| `ghi`, `gti`, `pvout`, `temp`, `elevation` | Global Solar Atlas 2.0, Solargis / World Bank Group (CC BY 4.0) |
| `slope`, `equatorwardness` | derived from the Global Solar Atlas elevation at ~1 km (as in the paper's raster MCDA) |
| `road_dist_m` | distance to roads, OpenStreetMap contributors (ODbL), Geofabrik extract; computed in equal-area projection |
| `grid_dist_m` | distance to the electricity grid, Gridfinder (Arderne et al. 2020; CC BY 4.0) |
| `pop_density` | Gridded Population of the World v4.11, CIESIN (CC BY 4.0) |
| `worldcover` | ESA WorldCover 2021 v200 class (CC BY 4.0) |
| `ensemble_2019` | this work: the RF+MLP ensemble on AEF 2019, mean of the 10 m map per cell (CC BY 4.0) |

Protected areas (WDPA) are **not** included: the WDPA terms do not allow redistribution. Download them from
[Protected Planet](https://www.protectedplanet.net) and pass `--wdpa` to `scripts/mcda_example.py` to apply the
paper's protected-area exclusion. Map data © OpenStreetMap contributors, available under the Open Database Licence.

To run the MCDA for another country, build the same layers on a regular lon/lat grid from the sources above (all are
public; [`docs/data_sources.md`](../docs/data_sources.md) has links) and save them with the same array names plus
`mask` (bool, country cells), `transform` (6 affine coefficients), `extent` (west, south, east, north) and
`worldcover`; then pass `--data` and `--sites`.
