# Expected output

What each script should print, so you can confirm your run matches. The bundled pixel
sample keeps the paper's ~1:5 solar:non-solar prevalence, so the pixel ROC-AUC and PR-AUC
reproduce Table 1 of the paper. Exact values can differ by a few thousandths across
platforms, BLAS/thread counts and library versions, and because the sample is a finite
draw from the full pools; the rankings and magnitudes are stable. Earth Engine runs also
vary slightly with the random land points drawn.

## `python quickstart.py`  (no Earth Engine, seconds)

```
Random forest scored on 60,000 held-out AlphaEarth pixels (10,000 solar / 50,000 non-solar):
  ROC-AUC 0.900 | PR-AUC 0.636 (PR chance 0.17)
```

Matches the deployed random forest's row in Table 1 (paper: 0.898 / 0.641).

## `python benchmark_extractors.py`  (no Earth Engine, ~1 min)

```
train 72,000 px / test 60,000 px (test chance 0.167)

  Mean-embedding cosine    ROC 0.705 | PR 0.319   prototype
  QDA                      ROC 0.856 | PR 0.517   discriminant
  Logistic probe           ROC 0.868 | PR 0.555   linear
  MaxEnt                   ROC 0.876 | PR 0.578   log-linear (SDM), L1 logistic
  Random Fourier features  ROC 0.892 | PR 0.626   kernel
  Random forest (deployed) ROC 0.900 | PR 0.636   ensemble, from models/
  Histogram grad. boosting ROC 0.910 | PR 0.682   ensemble
  MLP                      ROC 0.918 | PR 0.720   neural
```

Reproduces Table 1 (paper ROC / PR: cosine 0.702/0.318, QDA 0.854/0.521, logistic
0.866/0.551, MaxEnt 0.874/0.577, RFF 0.891/0.628, RF 0.898/0.641, HGB 0.919/0.713,
MLP 0.924/0.728). The deployed random forest is the most accurate method that still has a
native global-scale inference path in Earth Engine (the two above it do not). The two
non-deployable ensembles read a touch lower here only because they are retrained on the
small bundled subsample rather than the full pool.

## `python train_rf.py`  (no Earth Engine, ~10 s)

Trains the deployed recipe from scratch on the bundled sample and scores the held-out sample:

```
DEMO: training on the bundled sample (72,000 px) with the deployed recipe, at the paper's ~1:5 prevalence. ...
  min_samples_leaf=5: 2.52 MB tree strings (9s)
saved models/rf_demo.joblib (2.52 MB tree strings, min_samples_leaf=5)
  held-out px ROC-AUC 0.883 | PR-AUC 0.591 (chance 0.167) on 60,000 px
```

A little under the deployed model (0.900) because it is retrained from scratch on the small
subsample; the full pixel corpus and the Earth-Engine size cap (which forces
`min_samples_leaf ≈ 1500`) give the deployed numbers. `rf_demo.joblib` is a throwaway — the
released model is `models/rf30_v3_R.joblib`.

## `python run_country.py <Country>`  (needs Earth Engine, a few minutes)

Indicative numbers, validating each map against all inventory sites installed from the
imagery year + 2 onward (so exact `n` and ROC differ from the paper's balanced subset, and
from run to run):

| Command | Validation sites | ROC-AUC |
|---|---|---|
| `run_country.py Greece`  | ~1,080 | ~0.89 |
| `run_country.py Germany` | ~1,900 | ~0.85 |
| `run_country.py Spain`   | ~1,940 | ~0.85 |
| `run_country.py Italy`   | — (no bundled sites) | map only, validation skipped |

Each run writes to `outputs/`: `<country>_rf_<year>.png` (national suitability heatmap),
`<country>_rf_<year>_fig.png` (map + site-vs-random score distribution), and
`<country>_rf_<year>_scores.npz` (raw scores). A country with no bundled site file still
renders its suitability map and skips validation.
