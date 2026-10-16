# The dataset: solar_suitability v1.0

**Global 10 m solar PV siting suitability for AEF year 2025**, produced with the RF+MLP ensemble in this repository.
Archived at CEDA (DOI: *coming soon*). Licence: CC BY 4.0.

## What a value means

Each 10 m pixel holds `P(solar)`: the ensemble's probability that the land belongs to the "built solar" class,
judged from its 2025 AlphaEarth embedding, for a model trained on land as it looked **two years before** solar was
built on it, against globally random land. Read it as *how much this land resembles land that later became a
utility-scale solar site*, a relative ranking of siting suitability. It is **not** a measure of solar resource,
yield or economics, and it is not a forecast that a given pixel will be built on.

Values are comparable across the globe (one model, one scale), but some countries' land scores are compressed
into a narrow band (paper, Appendix E). Within-country **percentiles** (below) are the better view for national maps.

## File format

| | |
|---|---|
| Format | Cloud-Optimised GeoTIFF (COG), one file per AlphaEarth tile (~82 km x 82 km) |
| Size | 28,533 files, 851 GB; 1.35 x 10^12 valid pixels (~135 million km²), 250 countries and territories, 114 UTM zones |
| Grid | native UTM (EPSG:326zz / 327zz), 10 m, north-up |
| Band | 1 band, `suitability`, **uint8** |
| Encoding | value = round(P x 250); **P = value x 0.004** (scale 0.004, offset 0, stored in each file); **255 = no data** |
| Compression | DEFLATE level 9 + horizontal predictor; 512 x 512 blocks; internal overviews (2, 4, 8, ...) |
| Tags | product, version, AEF year, model and its SHA-256, encoding, licence, attribution, source AEF tile URL |

The uint8 step (0.004) is far below the model's own uncertainty; the paper's metrics were computed from unrounded
model scores.

## Layout

```
solar_suitability_v1.0/
  README.md  LICENSE  CHANGELOG.md  model_card.md
  data/10m/2025/<utm_zone>/solarsuit_v1.0_2025_<zone>_<aef_tile_id>.tif     the map (one file per AEF tile)
  index/     tiles.parquet | tiles.gpkg | tiles.csv     one row per file: footprint (EPSG:4326), zone, EPSG,
                                                         bounds, valid fraction, mean suitability, size, SHA-256,
                                                         countries (ISO3, ';'-separated), source AEF tile, n_valid
  lookup/    countries.csv                  ISO3, name, continent, number of tiles, UTM zones
             country_tiles.csv              ISO3 -> files (long table)
             <ISO3>.txt                     relative file paths for one country (wget / rsync ready)
             percentiles_global.csv         code -> global percentile of land
             percentiles_by_country.csv     ISO3, code -> within-country percentile
  mosaics/   <ISO3>_<zone>.vrt              per country and UTM zone: a virtual mosaic of the tiles
             <ISO3>_<zone>_pct.vrt          the same as within-country percentiles (Float32, 0-100, no data -1)
             global_<zone>.vrt (+ _pct)     all tiles of a UTM zone, with global percentiles
  stac/      catalog.json + collection + one item per tile (STAC 1.0)
  model/     solar_suitability_model_v1.0/  the model that produced the map (see model_card.md)
  docs/      QA_REPORT.md  inventory.md  quicklook_tiles.png  preview_global_suitability.png
```

File names carry the AlphaEarth tile id, so every file traces to its source embedding tile. There are no
country folders: tiles cross borders and countries span several UTM zones; use `index/` or `lookup/`.

## Percentile views

A pixel's **within-country percentile** is the share of the country's land (released pixels) that it outscores;
the **global percentile** is the same against all land. They are monotonic transforms of the score (ROC-AUC is
unchanged) and are stored as lookup tables (`lookup/`), applied on the fly by the `_pct.vrt` files, so they take no
extra space. Tables were built from ~49 million pixels sampled uniformly from the released land; countries with
too few samples for a stable table (73 small territories) use the global table (`table` column says which).

## Getting the data

- **Part of it:** `python scripts/get_dataset.py --country <name|ISO3> [--percentile]` (or `--bbox`,
  `--region-file`; `--whole-tiles` for unclipped files). Reads only what it needs over HTTPS.
- **A country's files directly:** `lookup/<ISO3>.txt` lists them; prepend the archive URL and fetch with `wget -i`.
- **Everything:** the archive is 851 GB; mirror it with the tools CEDA documents.
- **Open in GIS:** any `.tif` or `mosaics/*.vrt` opens in QGIS/ArcGIS; in Python, `rasterio.open(path)`; the
  scale is in the file, so `read(1, masked=True) * 0.004` gives P(solar).

## Land mask and coverage

- **Land:** pixels are kept if they lie within **500 m of land**, where land is the union of OpenStreetMap land
  polygons (2026-10-13) and ESA WorldCover 2021 (every class except permanent water). AlphaEarth also covers
  near-shore sea, which is removed beyond that margin. Inland water bodies that OSM maps as land (lakes) keep scores.
- **Not covered:** Antarctica and land south of 60°S (excluded by design); places with no AlphaEarth data.
- **Tile borders:** AlphaEarth tiles of neighbouring UTM zones overlap slightly; each file keeps its own pixels.

## Known limitations

- Learns from where solar **has** been built: land unlike past deployment (new markets, unusual terrain) may be
  under-scored even if suitable. Read alongside resource, grid and policy information.
- Pixel scores: does not test whether enough contiguous land exists for a plant of a given size; discrimination is
  weaker for the largest (>100 MW) farms (paper, Appendix F).
- One embedding year (2025): land that already hosts solar in 2025 scores high (it looks like solar land; sites
  around Athens built 2017-2022 sit at a median 98th percentile of the area's land). Mask known installations to see only new land.
- National score levels differ (weak-country effect, Appendix E): compare countries on percentiles.
- **Border tiles in the archive's country mosaics:** `mosaics/<ISO3>_<zone>.vrt` include whole tiles, so pixels of a
  neighbouring country inside a border tile also appear (and, in `_pct.vrt`, get this country's percentiles).
  `scripts/get_dataset.py --country` clips to the country outline and avoids this. Countries follow Natural Earth
  1:10m admin-0 (the same boundaries the scripts use).

## Versions

| | |
|---|---|
| v1.0 (2026-10) | AEF 2025; RF+MLP ensemble (`ens_unb_alldata+mlp_b55263de13_x5_w0.5`); land mask v2 (OSM ∪ WorldCover, 500 m) |
