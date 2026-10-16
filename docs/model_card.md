# Model card: solar_suitability_model_v1.0

## Model

| | |
|---|---|
| Type | Equal-weight ensemble (soft voting) of a random forest and a 5-seed MLP |
| Random forest | scikit-learn 1.8.0; 80 trees, max_features 12, min_samples_leaf 14, full bootstrap, Gini, class-balanced, seed 0 |
| MLP | 64 → 256 → 256 → 1 (ReLU), standardised inputs; 5 seeds (0-4) averaged; NumPy inference (no deep-learning framework) |
| Input | one 64-d AlphaEarth Foundations annual embedding per 10 m pixel (de-quantised: x = sign(q)·(q/127.5)²) |
| Output | P(solar) ∈ [0, 1]; released as uint8 = round(P × 250) |
| Files | `model.json`, `rf.joblib` (2.1 GB), `mlp.npz` (2 MB); checksums in `model.json` |
| Compute | CPU only. Model in memory: ~5 GB (full) or ~2 MB (`--lite`); a full mapping run peaks at ~9 GB (lite ~3.5 GB) |
| Name in the dataset tags | `ens_unb_alldata+mlp_b55263de13_x5_w0.5` |

## Training data

- **Positives:** footprint pixels of 38,591 solar facilities (GPVF inventory, installed 2019-2024), embedded at the
  AlphaEarth year **two years before installation**.
- **Negatives:** pixels of 13,440 globally random land parcels ≥ 5 km from any known solar facility (GPVF or TZ-SAM),
  embeddings from 2017-2022.
- The released model is trained on **all** ~45 M labelled pixels (no hold-out); the paper's evaluations use the same
  recipe trained on the 80 % training split.

## Evaluation (paper; same recipe on the training split, held-out data)

| Test | Result |
|---|---|
| Held-out pixels (Table 1) | ROC-AUC 0.932, PR-AUC 0.752 (chance 0.17) |
| Forward validation, 13 countries (sites installed ≥ 2 years after the 2019 map, vs random national land) | ROC-AUC 0.83-0.95, median 0.90; 66 % of sites in the most-suitable 10 % of land |
| Independent TZ-SAM sites absent from the training inventory | ROC-AUC 0.88 (n-weighted) |
| Leave-one-country-out (country removed from training) | mean ROC-AUC 0.82; largest drop for the most distinctive countries (China 0.92 → 0.72) |
| Leak-proof temporal (imagery ≤ 2021, 2024 installs) | mean ROC-AUC 0.90 |
| vs published MCDA (16 configurations, 13 countries) | +0.06 to +0.30 ROC-AUC in every country |

The released (all-data) model itself: ROC-AUC 0.922 on validation sites outside every chip it was trained on
(13 countries, n-weighted; split model 0.907), 0.894 on independent TZ-SAM sites (0.891), leave-one-country-out mean
0.81 (0.81). On bundled inventory sites inside its training chips it scores higher, which measures fit, not
forward skill ([`docs/reproduce.md`](reproduce.md#forward-validation-predictpy--validatepy)).

Released (all-data) ensemble vs the all-data random forest alone (paired bootstrap): TZ-SAM +0.010 [+0.009, +0.011],
leak-free country +0.009 [+0.007, +0.010], leave-one-country-out +0.012 [+0.007, +0.017]; temporal −0.002 [−0.004, 0.000].

## Intended use

Screening and research: finding land that resembles where utility-scale solar has been built; comparing with
resource, grid, environmental and social layers; studying siting patterns across countries.

## Limitations and out-of-scope use

- Learns **historical siting** (including infrastructure, policy and market effects), not normative "best" sites; not
  an energy-yield or economic model. Land unlike past deployment may be under-scored.
- Skill varies by country; weaker for the largest (> 100 MW) farms; transfer to countries absent from training is
  weaker (leave-one-country-out).
- Pixel-level: does not check for contiguous area, ownership, grid capacity, permitting or protected status.
- Land already covered by solar scores high; scores rank land and are not calibrated probabilities of development.
- Inherits AlphaEarth's characteristics (effective detail can be coarser than 10 m in places).
- **Not for permitting or investment decisions without local data and assessment.**

## Variants

`--lite` (MLP seeds only): 13-country forward ROC-AUC 0.918 vs 0.920 for the full ensemble; TZ-SAM 0.891 vs 0.891
(training-split versions). Outputs are labelled as the lite variant.
