# Data sources

Every external dataset used by the code, the paper or the released map, what it is used for, and its licence.
Please cite the original sources when you use them.

## Model inputs

| Dataset | Used for | Access | Licence |
|---|---|---|---|
| **AlphaEarth Foundations Satellite Embedding v1, annual** (Brown et al. 2025, Google / Google DeepMind) | the 64-d per-pixel input to the model, 2017 onwards | public COGs on Google Cloud Storage (`gs://alphaearth_foundations/satellite_embedding/v1/annual/`), mirrored on [Source Cooperative](https://source.coop/tge-labs/aef); tile index `aef_index.parquet` (Source Cooperative, TGE Labs). Read on the fly by `scripts/predict.py` | [terms](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) — attribution: *"The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."* |

## Labels and validation

| Dataset | Used for | Access | Licence |
|---|---|---|---|
| **Global PV Facility (GPVF) inventory** — Song, Yin et al. 2026, [doi:10.5281/zenodo.18794231](https://doi.org/10.5281/zenodo.18794231) | solar footprints and installation years (training positives); validation sites (bundled: `data/validation/`) | Zenodo | CC BY 4.0 |
| **TZ-SAM, TransitionZero Solar Asset Mapper** — Phillpott et al., [doi:10.5281/zenodo.11368204](https://doi.org/10.5281/zenodo.11368204) (the paper used the Q1 2026 release) | independent validation sites; screening of non-solar training parcels | TransitionZero / Zenodo | CC BY-NC 4.0 (not bundled) |
| **Generation estimates**, GPVF modelled annual generation (Song, Yin et al. 2026) | capacity / generation concentration analysis (paper, Fig. 2c) | Zenodo (same record) | CC BY 4.0 |

## Covariates and MCDA criteria (paper Appendices G and H; bundled for Greece in `data/mcda_example/`)

| Dataset | Layers | Licence |
|---|---|---|
| **Global Solar Atlas 2.0** (Solargis / World Bank Group, ESMAP), [globalsolaratlas.info](https://globalsolaratlas.info) | GHI, GTI, DNI, DIF, PVOUT (long-term photovoltaic output), air temperature, elevation | CC BY 4.0 |
| **Copernicus DEM GLO-30** (ESA / Sinergise), [doi:10.5069/G9028PQB](https://doi.org/10.5069/G9028PQB) | slope and elevation for the 10 m covariate analysis | free and open, attribution required |
| **ESA WorldCover 2021 v200** (Zanaga et al. 2022), [doi:10.5281/zenodo.7254221](https://doi.org/10.5281/zenodo.7254221) | land cover; MCDA water/wetland exclusion; part of the dataset's land mask | CC BY 4.0 |
| **OpenStreetMap** roads ([Geofabrik](https://download.geofabrik.de) extracts) | distance to roads | ODbL (© OpenStreetMap contributors) |
| **Gridfinder** (Arderne et al. 2020, *Scientific Data* 7:19), [doi:10.1038/s41597-019-0347-4](https://doi.org/10.1038/s41597-019-0347-4) | distance to the (predicted) medium-voltage grid | CC BY 4.0 |
| **GPWv4.11 population density** (CIESIN, Columbia University), [doi:10.7927/H49C6VHW](https://doi.org/10.7927/H49C6VHW) | population density (Chen et al. MCDA weighting) | CC BY 4.0 |
| **World Database on Protected Areas (WDPA)** (UNEP-WCMC & IUCN), [protectedplanet.net](https://www.protectedplanet.net) | protected-area exclusion in the MCDA | WDPA terms: no redistribution — not bundled; download it yourself and pass `--wdpa` |

Published MCDA weights: Richards et al. 2025 (*Environ. Sci. Pollut. Res.* 32:2007-2035, AHP) and Chen Jong &
Mohamud Ahmed 2024 (*IEEE Access* 12:143458); see paper Table 4.

## Boundaries and land mask

| Dataset | Used for | Licence |
|---|---|---|
| **Natural Earth** 1:10m admin-0 countries ([naturalearthdata.com](https://www.naturalearthdata.com)) | `--country` regions; country assignment of tiles | public domain |
| **OpenStreetMap land polygons** ([osmdata.openstreetmap.de](https://osmdata.openstreetmap.de/data/land-polygons.html), 2026-10-13) | the dataset's land mask (with WorldCover) | ODbL |

Natural Earth draws de-facto boundaries; where borders are disputed the code takes no position. Use
`--region-file` with any boundary definition you prefer.
