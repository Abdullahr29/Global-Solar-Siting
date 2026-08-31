# Expected output

What each script should print, so you can confirm your run matches. Exact ROC/PR values
can differ by roughly ±0.003 across platforms, BLAS/thread counts and library versions;
the rankings and magnitudes are stable. Earth Engine runs also vary slightly with the
random land points drawn.

## `python quickstart.py`  (no Earth Engine, seconds)

```
Random forest scored on 20,000 held-out AlphaEarth pixels (10,000 solar / 10,000 non-solar):
  ROC-AUC 0.901 | PR-AUC 0.887
```

## `python benchmark_extractors.py`  (no Earth Engine, ~1 min)

```
train 80,000 px / test 20,000 px (test chance 0.500)

  Mean-embedding cosine    ROC 0.705 | PR 0.685   prototype
  QDA                      ROC 0.858 | PR 0.833   discriminant
  Logistic probe           ROC 0.866 | PR 0.845   linear
  MaxEnt                   ROC 0.876 | PR 0.859   log-linear (SDM), L1 logistic
  Random Fourier features  ROC 0.893 | PR 0.876   kernel
  Random forest (deployed) ROC 0.901 | PR 0.887   ensemble, from models/
  Histogram grad. boosting ROC 0.917 | PR 0.908   ensemble
  MLP                      ROC 0.922 | PR 0.913   neural
```

The deployed random forest is the most accurate method that still has a native
global-scale inference path in Earth Engine (the two above it do not). This reproduces the
ranking of Table 1 in the paper.

## `python train_rf.py`  (no Earth Engine, ~15 s)

Trains the deployed recipe on the bundled sample and scores the held-out sample:

```
DEMO: training on the bundled sample (80,000 px) with the deployed recipe. ...
  min_samples_leaf=5: 3.33 MB tree strings (12s)
saved models/rf_demo.joblib (3.33 MB tree strings, min_samples_leaf=5)
  held-out px ROC-AUC 0.896 | PR-AUC 0.882 (chance 0.500) on 20,000 px
```

Just under the deployed model's 0.901 because the sample is a small, class-balanced
subset. On the full pixel corpus the Earth-Engine size cap forces `min_samples_leaf ≈ 1500`
(the deployed value); at sample scale the cap is met with a much smaller leaf. `rf_demo.joblib`
is a throwaway — the released model is `models/rf30_v3_R.joblib`.

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
