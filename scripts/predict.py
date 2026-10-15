"""Map solar-siting suitability for any region and any AlphaEarth year (2017 onwards), on your own machine.

Examples
    python scripts/predict.py --country Kenya --year 2024
    python scripts/predict.py --country GRC --year 2019 --out outputs/greece_2019
    python scripts/predict.py --bbox 23.5 37.8 24.2 38.2 --year 2025            # west south east north (degrees)
    python scripts/predict.py --region-file my_area.geojson --year 2022
    python scripts/predict.py --country Malta --year 2025 --lite                  # low-memory MLP-only variant

What it does: finds the AlphaEarth Foundations (AEF) tiles covering the region for that year (public tile index),
streams only the needed parts of each tile over HTTPS, runs the released RF+MLP ensemble on every 10 m pixel inside
the region, and writes uint8 GeoTIFFs encoded like the released dataset (P(solar) = value * 0.004; 255 = no data),
plus per-UTM-zone mosaics, within-region percentiles, and a quicklook PNG. Re-running the same command resumes.

Needs the released model (scripts/download_model.py). Memory: the full model needs ~11 GB of RAM to load (16 GB+
machine); --lite needs < 1 GB. See README "Requirements" for time and disk estimates.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def main():
    sys.stdout.reconfigure(line_buffering=True)          # progress shows up live, also in log files
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("region (give exactly one)")
    g.add_argument("--country", help="country name or ISO3 code, e.g. 'Kenya', 'United States of America', 'KEN'")
    g.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"), help="bounding box in degrees")
    g.add_argument("--region-file", help="vector file (GeoJSON, GeoPackage, Shapefile, ...); features are unioned")
    ap.add_argument("--year", type=int, required=True, help="AEF annual embedding year (2017 onwards)")
    ap.add_argument("--out", help="output folder (default: outputs/<region>_<year>[_lite])")
    ap.add_argument("--model", default=os.environ.get("SOLARSITING_MODEL", "models/solar_suitability_model_v1.0"),
                    help="released model folder (default: models/solar_suitability_model_v1.0 or $SOLARSITING_MODEL)")
    ap.add_argument("--lite", action="store_true",
                    help="MLP-only variant: < 1 GB RAM and much faster, slightly less accurate (not the released model)")
    ap.add_argument("--threads", type=int, default=os.cpu_count(), help="CPU threads (default: all cores)")
    ap.add_argument("--io-threads", type=int, default=8, help="parallel downloads of AEF blocks (default 8)")
    ap.add_argument("--refresh-index", action="store_true", help="re-download the AEF tile index (new years)")
    ap.add_argument("--list-tiles", action="store_true", help="only list the AEF tiles the region needs, then stop")
    a = ap.parse_args()

    from solarsiting import aef, infer, regions
    from solarsiting.model import load_model

    region, label = regions.resolve(a.country, a.bbox, a.region_file)
    label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label.lower()).strip("_")
    if a.refresh_index:
        aef.load_index(refresh=True)
    tiles = aef.tiles_for(region, a.year)
    print(f"region: {label} | AEF {a.year} | {len(tiles)} tile(s) | area ~{_area_km2(region):,.0f} km²")
    if a.list_tiles:
        for r in tiles.itertuples():
            print(f"  {r.utm_zone:>4}  {r.path}")
        return
    if not os.path.isdir(a.model):
        sys.exit(f"model folder not found: {a.model}\n  download it first:  python scripts/download_model.py")
    out = a.out or os.path.join("outputs", f"{label}_{a.year}" + ("_lite" if a.lite else ""))
    loader = lambda: load_model(a.model, threads=a.threads, lite=a.lite)    # loaded only if a tile needs scoring
    infer.run(loader, region, label, a.year, out, tiles=tiles, io_threads=a.io_threads)


def _area_km2(geom):
    import geopandas as gpd
    g = gpd.GeoSeries([geom], crs=4326)
    return float(g.to_crs(g.estimate_utm_crs()).area.iloc[0] / 1e6) if geom.area < 400 else float(g.to_crs(6933).area.iloc[0] / 1e6)


if __name__ == "__main__":        # required on Windows/macOS, where worker threads/processes are spawned
    try:
        main()
    except (ValueError, FileNotFoundError) as e:     # input mistakes: a clear message, not a traceback
        sys.exit(f"error: {e}")
