"""Step 5: the 3D sound map.

Run:  python3 05_map.py      (after 04_slice.py)

Writes map/index.html, a single page you can open in any browser: buildings
in 3D, the official noise map on the ground, and every floor of every facade
lit by its predicted noise. Drag to turn, scroll to zoom, hover for values.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).parent
DATA = HERE / "data"
GROUND_HALF = 520   # metres of official noise map shown around the centre


def main():
    parts = json.loads((DATA / "parts.json").read_text())
    cx, cy = parts["centre"]
    floors = pd.read_csv(DATA / "predictions_floors.csv")
    buildings = {f["properties"]["building"]: f["properties"]
                 for f in json.loads((DATA / "buildings.geojson").read_text())["features"]}

    # Walls: one record per stretch, with the noise for each floor on it.
    floors = floors.sort_values(["wall", "floor"])
    walls, values, facing, names = [], [], [], {}
    for wall, rows in floors.groupby("wall", sort=True):
        first = rows.iloc[0]
        f0 = int(rows["floor"].min())
        count = int(rows["floor"].max()) - f0 + 1
        by_floor = rows.set_index("floor")
        lden = by_floor["lden"].reindex(range(f0, f0 + count)).interpolate().bfill().ffill()
        lnight = by_floor["lnight"].reindex(range(f0, f0 + count)).interpolate().bfill().ffill()
        number = int(first["building"])
        walls += [round(first["x0"] - cx, 1), round(first["y0"] - cy, 1),
                  round(first["x1"] - cx, 1), round(first["y1"] - cy, 1), f0, count, number]
        for a, b in zip(lden, lnight):
            values += [round(float(a), 1), round(float(b), 1)]
        # Compass direction the wall faces: 0 = north, 2 = east, 4 = south, 6 = west.
        angle = np.degrees(np.arctan2(first["facing_x"], first["facing_y"])) % 360
        facing.append(int(round(angle / 45)) % 8)
        name = buildings.get(number, {}).get("name", "")
        if name:
            names[number] = name

    blocks = [{"f": p["floors"], "r": [[[round(x - cx, 1), round(y - cy, 1)] for x, y in ring]
                                       for ring in p["rings"]]} for p in parts["parts"]]

    # Ground: the official map, cropped around the centre.
    ground = {}
    for key, name in (("lden", "road_lden.tif"), ("lnight", "road_lnight.tif")):
        with rasterio.open(DATA / "raw" / name) as source:
            grid = source.read(1)
            t = source.transform
        c0 = int((cx - GROUND_HALF - t.c) / t.a)
        c1 = int((cx + GROUND_HALF - t.c) / t.a)
        r0 = int((t.f - (cy + GROUND_HALF)) / -t.e)
        r1 = int((t.f - (cy - GROUND_HALF)) / -t.e)
        crop = grid[r0:r1, c0:c1]
        ground[key] = np.round(np.where(crop > 0, crop, 0), 1).ravel().tolist()
        ground.update({"rows": crop.shape[0], "cols": crop.shape[1], "cell": t.a,
                       "left": round(t.c + c0 * t.a - cx, 1), "top": round(t.f + r0 * t.e - cy, 1)})

    payload = {"floor_height": parts["floor_height"], "parts": blocks, "walls": walls,
               "values": values, "facing": facing, "names": names, "ground": ground}
    text = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    page = (HERE / "map_template.html").read_text().replace("__DATA__", text)
    out = HERE / "map"
    out.mkdir(exist_ok=True)
    (out / "index.html").write_text(page)
    print("Wrote map/index.html (%.1f MB): %d blocks, %d walls, %d floor panels"
          % (len(page) / 1e6, len(blocks), len(walls) // 7, len(values) // 2))
    print("Open it with:  open map/index.html")


if __name__ == "__main__":
    main()
