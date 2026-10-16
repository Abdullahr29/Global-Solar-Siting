# Changelog

## 1.0.0 (2026-10)

Public release accompanying the camera-ready paper and the solar_suitability v1.0 dataset.

- Released model is the RF+MLP ensemble (was a 30-tree random forest in the review artifact), distributed in a
  framework-free format; CPU-only NumPy inference, with a low-memory `--lite` (MLP-only) variant.
- `scripts/predict.py`: map any country, bounding box or polygon for any AlphaEarth year (2017 onwards) from the public
  AlphaEarth tiles, on a laptop; no Earth Engine account. Reproduces the released map.
- `scripts/get_dataset.py`: fetch a country or area of the released global map, with percentile views.
- `scripts/validate.py`, `benchmark.py` (9 methods, Table 1), `train.py` (release recipe), `mcda_example.py`
  (bundled Greece data), `quickstart.py`, `download_model.py`.
- Documentation: dataset format, methods, model card, data sources and licences, reproduction guide.
- Bundled validation sites carry a `paper_split` column (train / test / unseen chips), and `validate.py` reports
  results by it: the released all-data model has seen these sites, unlike the paper's train-split evaluation model.
- Long runs survive dropped connections: remote reads retry on a fresh connection with back-off.
- `get_dataset.py` reads a local dataset copy in place and caches a remote one per source; `--refresh-index`
  re-downloads the tile index and percentile tables together.
- Removed: the Earth Engine route (`run_country.py`), the size-capped review random forest and its scripts.

## 0.1 (2026-08)

Anonymous artifact for double-blind review: random-forest model, pixel sample, Earth Engine country runs.
