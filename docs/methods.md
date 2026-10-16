# Methods in brief

The paper has the full description; this page summarises it and documents how the global layer was produced.

## Learning task

- **Positives:** 38,591 solar facilities drawn country-balanced from the GPVF inventory (140,945 facilities with
  footprints and installation years, 2019-2024). Each footprint is rasterised onto the 10 m AlphaEarth grid; the
  pixels inside it are labelled solar, using the AlphaEarth embedding of the **year two years before installation**
  (so the model sees the land before construction, not panels).
- **Negatives:** 13,440 globally random parcels at least 5 km from any known solar facility (GPVF or TZ-SAM), with
  embeddings from a random year 2017-2022. Final solar : non-solar pixel ratio ~1:5.
- **Pool:** ~45 M labelled pixels (chips capped so no single large site dominates); scenes split 80/20 into train and
  test as whole chips, so no site contributes to both.

## Model

An equal-weight ensemble (soft voting) of two models on the 64-d de-quantised AEF embedding:

| | |
|---|---|
| Random forest | 80 trees, `max_features=12`, `min_samples_leaf=14`, full bootstrap, Gini, class-balanced, no depth cap |
| MLP | 64 → 256 → 256 → 1, ReLU, standardised inputs, class-balanced BCE, AdamW (lr 3e-3, wd 1e-5), batch 16,384, one-cycle schedule over 8 epochs stopped after the first; 5 seeds averaged |
| Ensemble | P = 0.5 · P_RF + 0.5 · mean(P_MLP) |

Every choice (MLP configuration from a 64-configuration random search, stopping epoch, ensemble weight) was made on
one half of the country validation sites (split by 0.5° cells), under a rule fixed before the other half was seen.
Nine extractors of increasing complexity were compared (paper Table 1, `scripts/benchmark.py`); the ensemble was
best on every held-out test. The released model is the same recipe refit on **all** labelled data.

## Validation

- **Forward in time:** national maps from the 2019 embedding, scored against sites installed from 2021 onward
  (absent from the imagery), versus random national land; 13 countries; inventory and TZ-SAM sites.
- **Across space:** leave-one-country-out refits; **leak-proof temporal:** training imagery ≤ 2021, validated on 2024
  installs with the 2022 embedding.
- **Against MCDA:** 16 published-weight multi-criteria configurations (Richards / Chen weights, WLC / TOPSIS, two
  land treatments), scored on the same task.

## How the global layer (v1.0) was produced

1. **Inference:** every AlphaEarth 2025 tile north of 60°S (31,039 tiles) was streamed from the public bucket,
   de-quantised, scored by the released model, and written as a uint8 COG in its native UTM grid (12 parallel jobs on
   the JASMIN LOTUS cluster, ~1.5 days). Inputs were bit-identical to the Earth Engine collection.
   `scripts/predict.py` runs the identical computation on any region.
2. **Land mask:** AlphaEarth also covers near-shore sea. Pixels more than 500 m from land were set to no data, with
   land = OpenStreetMap land polygons ∪ ESA WorldCover 2021 (all classes except permanent water). Kept pixels are
   byte-identical to the model output; tiles left empty (open sea, 2,506) were dropped, leaving 28,533 files.
3. **Index and catalogue:** a tile index (GeoParquet / GeoPackage / CSV) with footprints, checksums and Natural
   Earth country assignment; per-country file lists and virtual mosaics; a STAC catalogue.
4. **Percentiles:** global and within-country lookup tables from ~49 M pixels sampled uniformly over released land,
   exposed as on-the-fly VRT views.
5. **Quality control (automated, must pass):** completeness against the AEF index; file format and CRS of every
   tile; decoded values against the production samples; a WorldCover check that masked pixels are water;
   percentile monotonicity and read-back; no stray or partial files.
