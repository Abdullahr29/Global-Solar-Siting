"""Region of interest -> one shapely geometry in EPSG:4326 (lon/lat).

  country  name or ISO 3166-1 alpha-3 code, matched against Natural Earth 1:10m admin-0 boundaries (public domain;
           downloaded once, ~5 MB). Natural Earth draws de-facto boundaries; use --region-file for any other definition.
  bbox     west south east north, in degrees
  file     any vector file GeoPandas can read (GeoJSON, GeoPackage, Shapefile, ...); all features are unioned
"""
import os

NE_URLS = ("https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
           "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip")
_NAME_COLS = ("ADMIN", "NAME", "NAME_LONG", "FORMAL_EN", "SOVEREIGNT", "NAME_EN")
_ISO_COLS = ("ADM0_A3", "ISO_A3", "SOV_A3", "ADM0_ISO")


def _countries():
    import geopandas as gpd
    from .aef import cache_dir, _download
    p = os.path.join(cache_dir(), "ne_10m_admin_0_countries.zip")
    if not os.path.exists(p):
        for url in NE_URLS:
            try:
                print(f"downloading Natural Earth country boundaries (once) -> {p}", flush=True)
                _download(url, p); break
            except Exception as e:
                err = e
        else:
            raise RuntimeError(f"could not download Natural Earth boundaries: {err}")
    return gpd.read_file(f"zip://{p}").to_crs(4326)


def country(name):
    g = _countries()
    key = name.strip().lower()
    for cols in (_ISO_COLS, _NAME_COLS):
        for c in cols:
            if c in g.columns:
                hit = g[g[c].astype(str).str.lower() == key]
                if len(hit):
                    return hit.geometry.union_all(), str(hit.iloc[0].get("ADMIN", name))
    near = sorted(g["ADMIN"].astype(str), key=lambda s: _distance(s.lower(), key))[:5]
    raise ValueError(f"country '{name}' not found in Natural Earth admin-0; did you mean: {', '.join(near)}?")


def _distance(a, b):
    """Small edit distance, only for 'did you mean' suggestions."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def bbox(west, south, east, north):
    from shapely.geometry import box
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox must be: west south east north, in degrees, west < east and south < north")
    return box(west, south, east, north)


def from_file(path):
    import geopandas as gpd
    g = gpd.read_file(path)
    if g.crs is None:
        print("warning: region file has no CRS; assuming EPSG:4326 (lon/lat)")
        g = g.set_crs(4326)
    return g.to_crs(4326).geometry.union_all()


def resolve(country_name=None, bbox_vals=None, region_file=None):
    """(geometry, label) from exactly one of the three region options."""
    given = [x is not None for x in (country_name, bbox_vals, region_file)]
    if sum(given) != 1:
        raise ValueError("give exactly one of --country, --bbox or --region-file")
    if country_name:
        geom, label = country(country_name)
        return geom, label
    if bbox_vals:
        return bbox(*bbox_vals), "bbox_" + "_".join(f"{v:g}" for v in bbox_vals)
    return from_file(region_file), os.path.splitext(os.path.basename(region_file))[0]
