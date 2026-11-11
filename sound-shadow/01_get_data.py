"""Step 1: download everything for the Elephant & Castle sound map.

Run:  python3 01_get_data.py
Files land in data/raw. Running it again skips files that already exist.

What it fetches (all open data):
  * Official road noise at 4 m above ground, 10 m grid  (Defra, 2022 mapping)
  * The same noise split into 8 pitch bands                (Defra)
  * Laser-measured surface and ground height, 1 m grid     (Environment Agency)
  * Building outlines, roads and railways                  (OpenStreetMap)
"""
import json
import sys
import time
from pathlib import Path

import requests

# ---- the study area: a 1.6 km square around Elephant & Castle ----------
# British National Grid coordinates, in metres (east, north).
CENTRE_E, CENTRE_N = 531900, 178900
HALF = 800
E0, E1 = CENTRE_E - HALF, CENTRE_E + HALF
N0, N1 = CENTRE_N - HALF, CENTRE_N + HALF

RAW = Path(__file__).parent / "data" / "raw"
DEFRA = "https://environment.data.gov.uk/spatialdata/"
HEADERS = {"User-Agent": "mapciety-sound-shadow (student map project)"}

BANDS = ["63Hz", "125Hz", "250Hz", "500Hz", "1000Hz", "2000Hz", "4000Hz", "8000Hz"]

# name of the file we save -> (service, layer id)
RASTERS = {
    "road_lden.tif": ("road-noise-all-metrics-england-round-4",
                      "562c9d56-7c2d-4d42-83bb-578d6e97a517__Road_Noise_Lden_England_Round_4_All"),
    "road_lnight.tif": ("road-noise-all-metrics-england-round-4",
                        "562c9d56-7c2d-4d42-83bb-578d6e97a517__Road_Noise_Lnight_England_Round_4_All"),
    "surface_1m.tif": ("lidar-composite-digital-surface-model-first-return-dsm-1m",
                       "df4e3ec3-315e-48aa-aaaf-b5ae74d7b2bb__Lidar_Composite_Elevation_FZ_DSM_1m"),
    "ground_1m.tif": ("lidar-composite-digital-terrain-model-dtm-1m",
                      "13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"),
}
for band in BANDS:
    RASTERS["road_lden_" + band + ".tif"] = (
        "road-noise-england-round-4-all",
        "23743291-2503-45f2-a632-c7afef2e8809__Road_Noise_Lden_England_Round_4_All_" + band)

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

problems = []


def raster_url(service, layer):
    return (DEFRA + service + "/wcs?service=WCS&version=2.0.1&request=GetCoverage"
            + "&coverageId=" + layer
            + "&subset=E(%d,%d)&subset=N(%d,%d)" % (E0, E1, N0, N1)
            + "&format=image/tiff")


def get_raster(filename, service, layer):
    target = RAW / filename
    if target.exists() and target.stat().st_size > 1000:
        print("  have    ", filename)
        return
    try:
        reply = requests.get(raster_url(service, layer), headers=HEADERS, timeout=180)
        if reply.status_code != 200 or not reply.content[:2] in (b"II", b"MM"):
            raise RuntimeError("server said %s: %s" % (reply.status_code, reply.text[:200]))
        target.write_bytes(reply.content)
        print("  fetched  %s (%.1f MB)" % (filename, len(reply.content) / 1e6))
    except Exception as error:
        problems.append("%s: %s" % (filename, error))
        print("  FAILED  ", filename, "-", error)


def latlon_box():
    """The study square as south, west, north, east in degrees, for OpenStreetMap."""
    from pyproj import Transformer
    to_degrees = Transformer.from_crs(27700, 4326, always_xy=True)
    lons, lats = [], []
    for e in (E0, E1):
        for n in (N0, N1):
            lon, lat = to_degrees.transform(e, n)
            lons.append(lon)
            lats.append(lat)
    return min(lats), min(lons), max(lats), max(lons)


def get_osm():
    target = RAW / "osm.json"
    if target.exists() and target.stat().st_size > 1000:
        print("  have     osm.json")
        return
    south, west, north, east = latlon_box()
    box = "(%.6f,%.6f,%.6f,%.6f)" % (south, west, north, east)
    query = ("[out:json][timeout:120];("
             + 'way["building"]' + box + ";"
             + 'relation["building"]' + box + ";"
             + 'way["building:part"]' + box + ";"
             + 'way["highway"]' + box + ";"
             + 'way["railway"~"^(rail|light_rail|subway)$"]' + box + ";"
             + ");out body geom;")
    for server in OVERPASS_SERVERS:
        try:
            reply = requests.post(server, data={"data": query}, headers=HEADERS, timeout=180)
            if reply.status_code != 200:
                raise RuntimeError("server said %s" % reply.status_code)
            data = reply.json()
            if not data.get("elements"):
                raise RuntimeError("empty answer")
            target.write_text(json.dumps(data))
            print("  fetched  osm.json (%.1f MB) from %s" % (target.stat().st_size / 1e6, server.split("/")[2]))
            return
        except Exception as error:
            print("  no luck with", server.split("/")[2], "-", error)
            time.sleep(3)
    problems.append("osm.json: no OpenStreetMap server answered")


def report():
    import numpy as np
    import rasterio

    print("=" * 62)
    print("REPORT  (Elephant & Castle, %d m square)" % (2 * HALF))
    print("=" * 62)

    def read(name):
        path = RAW / name
        if not path.exists():
            return None, None
        with rasterio.open(path) as source:
            return source.read(1).astype("float64"), source

    for name, label in (("road_lden.tif", "Road noise, 24 h (Lden)"),
                        ("road_lnight.tif", "Road noise, night (Lnight)")):
        grid, source = read(name)
        if grid is None:
            print(label + ": MISSING")
            continue
        outdoors = grid[grid > 0]
        print("%s: grid %d x %d, cell %.0f m, %s" % (label, source.width, source.height, source.res[0], source.crs))
        print("   cells with a value: %d of %d (the rest are inside buildings or below the cutoff)"
              % (outdoors.size, grid.size))
        if outdoors.size:
            print("   quietest %.1f dB, middle %.1f dB, loudest %.1f dB"
                  % (outdoors.min(), np.median(outdoors), outdoors.max()))

    have_bands = [b for b in BANDS if (RAW / ("road_lden_" + b + ".tif")).exists()]
    print("Pitch bands present: %d of 8  %s" % (len(have_bands), " ".join(have_bands)))

    surface, source = read("surface_1m.tif")
    ground, _ = read("ground_1m.tif")
    if surface is None or ground is None:
        print("Height data: MISSING")
    else:
        print("Height data: grid %d x %d, cell %.2f m, %s" % (source.width, source.height, source.res[0], source.crs))
        ok = (surface > -100) & (ground > -100) & (surface < 1000) & (ground < 1000)
        above = (surface - ground)[ok]
        print("   usable cells: %.1f%%" % (100 * ok.mean()))
        print("   ground level: %.1f to %.1f m above sea" % (ground[ok].min(), ground[ok].max()))
        print("   height above ground: middle %.1f m, top 1%% above %.1f m, tallest %.1f m"
              % (np.median(above), np.percentile(above, 99), above.max()))
        for limit in (10, 30, 60, 100):
            print("   area taller than %3d m: %.2f%%" % (limit, 100 * (above > limit).mean()))

    path = RAW / "osm.json"
    if not path.exists():
        print("OpenStreetMap: MISSING")
    else:
        elements = json.loads(path.read_text())["elements"]
        buildings = [e for e in elements if "building" in e.get("tags", {})]
        parts = [e for e in elements if "building:part" in e.get("tags", {})]
        roads = [e for e in elements if "highway" in e.get("tags", {})]
        rails = [e for e in elements if "railway" in e.get("tags", {})]
        with_height = sum(1 for b in buildings if "height" in b["tags"])
        with_floors = sum(1 for b in buildings if "building:levels" in b["tags"])
        print("OpenStreetMap: %d buildings (%d with a height, %d with a floor count), %d building parts"
              % (len(buildings), with_height, with_floors, len(parts)))
        kinds = {}
        for road in roads:
            kinds[road["tags"]["highway"]] = kinds.get(road["tags"]["highway"], 0) + 1
        top = sorted(kinds.items(), key=lambda item: -item[1])[:10]
        print("   roads: %d   %s" % (len(roads), ", ".join("%s %d" % item for item in top)))
        print("   railway lines: %d" % len(rails))

    print("-" * 62)
    if problems:
        print("Finished with problems:")
        for problem in problems:
            print("  -", problem)
    else:
        print("All datasets are in place.")
    print("=" * 62)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    print("Downloading to", RAW)
    for filename, (service, layer) in RASTERS.items():
        get_raster(filename, service, layer)
    get_osm()
    report()
    sys.exit(1 if problems else 0)
