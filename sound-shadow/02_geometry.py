"""Step 2: building heights and what each point can "see".

Run:  python3 02_geometry.py      (after 01_get_data.py; takes a few minutes)

For every listening point this works out how much road it can see, how far
away that road is, and how much of it is hidden behind buildings. Two sets
of listening points are made:

  * data/points_4m.csv      the official noise grid at 4 m above ground,
                            used to train and test the model in step 3
  * data/points_floors.csv  1.5 m in front of every facade, once per floor,
                            where step 3 predicts the noise
  * data/parts.json         every building in the map square split into blocks
                            of equal height, for the 3D map

It also writes data/buildings.geojson (outline, height and floors of every
building) for drawing the map.

How "hidden" is worked out: a straight line is drawn from each road point to
the listener, and the laser-scan heights are checked every 2 m along it. If
a building sticks out above the line, the sound has to bend over it, and the
taller the building is above the line, the more sound is lost (a standard
rule of thumb from acoustics, Maekawa's formula). Trees are ignored, because
they barely block traffic noise; only heights inside building outlines count.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from rasterio import features as raster_features
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import unary_union

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data"

RADIUS = 200          # roads further away than this (metres) are ignored
SOURCE_STEP = 8       # one road point every 8 m
SOURCE_HEIGHT = 0.5   # tyres and engines sit about half a metre up
OFFICIAL_HEIGHT = 4.0 # the official noise map is calculated 4 m above ground
TRAIN_MARGIN = 220    # grid points closer than this to the edge of the
                      # downloaded square would miss roads outside it
MAP_HALF = 450        # facades are made for a 900 m square in the middle
FACADE_STEP = 6       # one listening point every 6 m along a facade
FACADE_OFFSET = 1.5   # ... standing 1.5 m in front of it
FLOOR_HEIGHT = 3.2    # metres per storey
WAVELENGTH = 0.68     # sound at 500 Hz, typical for traffic
MAX_SCREENING = 20.0  # a single building rarely removes more than 20 dB

# OpenStreetMap road types, grouped by how much traffic they usually carry.
ROAD_GROUPS = {
    "trunk": "trunk", "trunk_link": "trunk",
    "primary": "primary", "primary_link": "primary", "busway": "primary",
    "secondary": "secondary", "secondary_link": "secondary",
    "tertiary": "tertiary", "tertiary_link": "tertiary",
    "residential": "minor", "unclassified": "minor", "living_street": "minor",
    "service": "service",
}
GROUPS = ["trunk", "primary", "secondary", "tertiary", "minor", "service"]
NOT_BARRIERS = {"roof", "carport", "construction", "no"}

to_metres = Transformer.from_crs(4326, 27700, always_xy=True)


def say(*text):
    print(*text, flush=True)


# --------------------------------------------------------------------------
# Rasters
# --------------------------------------------------------------------------
def load_rasters():
    with rasterio.open(RAW / "surface_1m.tif") as source:
        surface = source.read(1).astype("float32")
        transform = source.transform
    with rasterio.open(RAW / "ground_1m.tif") as source:
        ground = source.read(1).astype("float32")
    bad = (ground < -50) | (ground > 500)
    if bad.any():
        ground[bad] = np.nanmedian(ground[~bad])
    surface = np.where((surface < -50) | (surface > 1000), ground, surface)
    above_ground = np.clip(surface - ground, 0, None)
    return ground, above_ground, transform


class Grid:
    """Look up values from a 1 m raster by easting and northing."""

    def __init__(self, transform, shape):
        self.left = transform.c
        self.top = transform.f
        self.size = transform.a
        self.rows, self.cols = shape

    def index(self, east, north):
        col = np.floor((east - self.left) / self.size).astype(np.int32)
        row = np.floor((self.top - north) / self.size).astype(np.int32)
        np.clip(col, 0, self.cols - 1, out=col)
        np.clip(row, 0, self.rows - 1, out=row)
        return row, col


# --------------------------------------------------------------------------
# OpenStreetMap
# --------------------------------------------------------------------------
def line_from(geometry):
    lons = [p["lon"] for p in geometry]
    lats = [p["lat"] for p in geometry]
    east, north = to_metres.transform(lons, lats)
    return list(zip(east, north))


def load_osm():
    elements = json.loads((RAW / "osm.json").read_text())["elements"]
    buildings, roads = [], []
    for element in elements:
        tags = element.get("tags", {})
        if "building" in tags and tags["building"] not in NOT_BARRIERS:
            try:
                if element["type"] == "way":
                    shape = Polygon(line_from(element["geometry"]))
                else:
                    outers, inners = [], []
                    for member in element.get("members", []):
                        if member.get("type") != "way" or "geometry" not in member:
                            continue
                        points = line_from(member["geometry"])
                        if len(points) < 4 or points[0] != points[-1]:
                            continue
                        (inners if member.get("role") == "inner" else outers).append(Polygon(points))
                    if not outers:
                        continue
                    shape = unary_union(outers)
                    if inners:
                        shape = shape.difference(unary_union(inners))
                shape = shape.buffer(0)
                if shape.area >= 15:
                    buildings.append({"osm_id": "%s/%s" % (element["type"], element["id"]),
                                      "kind": tags["building"], "name": tags.get("name", ""),
                                      "geometry": shape})
            except (ValueError, KeyError):
                continue
        group = ROAD_GROUPS.get(tags.get("highway"))
        if element["type"] == "way" and group and tags.get("tunnel") not in ("yes", "building_passage") \
                and tags.get("area") != "yes":
            roads.append({"group": group, "geometry": LineString(line_from(element["geometry"]))})

    # Drop outlines that are copies of a bigger one (OSM sometimes has both).
    buildings.sort(key=lambda b: -b["geometry"].area)
    kept, tree_shapes = [], []
    for building in buildings:
        shape = building["geometry"]
        if tree_shapes:
            tree = shapely.STRtree(tree_shapes)
            near = tree.query(shape)
            # Only near-identical copies; a tower drawn inside a bigger podium outline stays.
            if any(shape.intersection(tree_shapes[i]).area > 0.9 * shape.area
                   and shape.area > 0.8 * tree_shapes[i].area for i in near):
                continue
        kept.append(building)
        tree_shapes.append(shape)
    return kept, roads


def road_points(roads):
    xs, ys, lengths, groups = [], [], [], []
    for road in roads:
        line = road["geometry"]
        if line.length < 1:
            continue
        count = max(1, int(round(line.length / SOURCE_STEP)))
        step = line.length / count
        for i in range(count):
            point = line.interpolate((i + 0.5) * step)
            xs.append(point.x)
            ys.append(point.y)
            lengths.append(step)
            groups.append(GROUPS.index(road["group"]))
    return np.array(xs), np.array(ys), np.array(lengths), np.array(groups)


# --------------------------------------------------------------------------
# Listening points
# --------------------------------------------------------------------------
def building_parts(id_raster, above_ground, transform, centre):
    """Split every building in the map square into blocks with the same number of floors.

    The number of floors comes from the laser-scan height (3.2 m per floor), smoothed a
    little so roof clutter doesn't make tiny blocks. A tower on a podium becomes two
    blocks, a terrace with a taller end house becomes two, and so on.
    """
    from scipy import ndimage
    from shapely.geometry import shape as to_shape

    cx, cy = centre
    pad = 40
    c0 = int(cx - MAP_HALF - pad - transform.c)
    c1 = int(cx + MAP_HALF + pad - transform.c)
    r0 = int(transform.f - (cy + MAP_HALF + pad))
    r1 = int(transform.f - (cy - MAP_HALF - pad))
    ids = id_raster[r0:r1, c0:c1]
    floors = np.where(ids > 0, np.round(above_ground[r0:r1, c0:c1] / FLOOR_HEIGHT), 0)
    floors = np.where(ids > 0, ndimage.median_filter(floors, size=5), 0)
    floors = np.clip(floors, 0, 199)
    label = np.where((ids > 0) & (floors > 0), ids * 200 + floors, 0).astype("int32")
    window = rasterio.Affine(transform.a, 0, transform.c + c0, 0, transform.e, transform.f - r0)
    parts = []
    for geometry, value in raster_features.shapes(label, mask=label > 0, transform=window):
        polygon = to_shape(geometry).simplify(0.7, preserve_topology=True)
        if polygon.is_empty or polygon.area < 12:
            continue
        middle = polygon.representative_point()
        if abs(middle.x - cx) > MAP_HALF or abs(middle.y - cy) > MAP_HALF:
            continue
        parts.append({"building": int(value) // 200, "floors": int(value) % 200, "geometry": polygon})
    return parts


def facade_points(parts, id_raster, above_ground, grid):
    """One listening point per floor, 1.5 m in front of every stretch of wall (6 m or less)."""
    rows = []
    wall = 0
    for number, part in enumerate(parts):
        shape = part["geometry"]
        top = part["floors"] * FLOOR_HEIGHT
        for ring in [shape.exterior, *shape.interiors]:
            xy = np.asarray(ring.coords)
            for (ax, ay), (bx, by) in zip(xy[:-1], xy[1:]):
                length = np.hypot(bx - ax, by - ay)
                if length < 0.8:
                    continue
                pieces = int(np.ceil(length / FACADE_STEP))
                cuts = np.linspace(0, 1, pieces + 1)
                x0, y0 = ax + cuts[:-1] * (bx - ax), ay + cuts[:-1] * (by - ay)
                x1, y1 = ax + cuts[1:] * (bx - ax), ay + cuts[1:] * (by - ay)
                hx, hy = (x0 + x1) / 2, (y0 + y1) / 2
                nx, ny = (by - ay) / length, -(bx - ax) / length
                if shapely.contains_xy(shape, hx[0] + 0.5 * nx, hy[0] + 0.5 * ny):
                    nx, ny = -nx, -ny
                px, py = hx + FACADE_OFFSET * nx, hy + FACADE_OFFSET * ny
                row, col = grid.index(px, py)
                # Anything standing right in front (a podium, the neighbouring house):
                # only floors above it are open to the street.
                in_front = np.where(id_raster[row, col] > 0, above_ground[row, col], 0)
                for i in range(pieces):
                    wall += 1
                    for floor in range(part["floors"]):
                        height = floor * FLOOR_HEIGHT + 1.5
                        if height > in_front[i] + 0.5:
                            rows.append((part["building"], number, wall, px[i], py[i], nx, ny,
                                         x0[i], y0[i], x1[i], y1[i], floor, height, top))
    return pd.DataFrame(rows, columns=["building", "part", "wall", "x", "y", "facing_x", "facing_y",
                                       "x0", "y0", "x1", "y1", "floor", "height", "facade_height"])


# --------------------------------------------------------------------------
# The core: what each listening point can see
# --------------------------------------------------------------------------
def visibility_features(east, north, height_above_ground, sources, obstacle, ground, grid,
                        label, chunk=60, samples=100):
    src_x, src_y, src_len, src_group = sources
    sr, sc = grid.index(src_x, src_y)
    src_z = ground[sr, sc] + SOURCE_HEIGHT
    rr, rc = grid.index(east, north)
    rec_z = ground[rr, rc] + height_above_ground
    tree = cKDTree(np.c_[src_x, src_y])
    major = src_group <= 1
    major_tree = cKDTree(np.c_[src_x[major], src_y[major]])
    t = (np.arange(samples) + 0.5) / samples

    n = len(east)
    open_energy = np.zeros((n, len(GROUPS)))
    screened_energy = np.zeros((n, len(GROUPS)))
    hidden_share = np.zeros(n)
    started = time.time()
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        lists = tree.query_ball_point(np.c_[east[start:stop], north[start:stop]], RADIUS)
        counts = np.array([len(x) for x in lists])
        if counts.sum() == 0:
            continue
        ri = np.repeat(np.arange(start, stop), counts)
        si = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
        dx, dy = east[ri] - src_x[si], north[ri] - src_y[si]
        flat = np.hypot(dx, dy)
        dz = rec_z[ri] - src_z[si]
        # Heights along every line, sampled 100 times.
        px = src_x[si, None] + t[None, :] * dx[:, None]
        py = src_y[si, None] + t[None, :] * dy[:, None]
        row, col = grid.index(px, py)
        line_z = src_z[si, None] + t[None, :] * dz[:, None]
        excess = obstacle[row, col] - line_z
        worst = np.argmax(excess, axis=1)
        top = excess[np.arange(len(si)), worst]
        blocked = top > 0
        # Extra distance the sound travels bending over the highest edge.
        a = t[worst] * flat
        b = (1 - t[worst]) * flat
        edge_z = obstacle[row[np.arange(len(si)), worst], col[np.arange(len(si)), worst]]
        direct = np.hypot(flat, dz)
        detour = np.hypot(a, edge_z - src_z[si]) + np.hypot(b, rec_z[ri] - edge_z) - direct
        fresnel = 2 * np.clip(detour, 0, None) / WAVELENGTH
        loss = np.where(blocked, np.minimum(10 * np.log10(3 + 20 * fresnel), MAX_SCREENING), 0.0)
        energy = src_len[si] / np.maximum(direct, 1.0) ** 2
        keys = (ri - start) * len(GROUPS) + src_group[si]
        size = (stop - start) * len(GROUPS)
        open_energy[start:stop] += np.bincount(keys, energy * ~blocked, size).reshape(-1, len(GROUPS))
        screened_energy[start:stop] += np.bincount(keys, energy * 10 ** (-loss / 10), size).reshape(-1, len(GROUPS))
        total = np.bincount(ri - start, energy, stop - start)
        hidden = np.bincount(ri - start, energy * blocked, stop - start)
        hidden_share[start:stop] = hidden / np.maximum(total, 1e-12)
        if (start // chunk) % 40 == 0:
            say("   %s: %d of %d points, %.0f s" % (label, stop, n, time.time() - started))

    features = pd.DataFrame(index=np.arange(n))
    for g, name in enumerate(GROUPS):
        features["open_" + name] = 10 * np.log10(open_energy[:, g] + 1e-6)
        features["screened_" + name] = 10 * np.log10(screened_energy[:, g] + 1e-6)
    features["open_all"] = 10 * np.log10(open_energy.sum(1) + 1e-6)
    features["screened_all"] = 10 * np.log10(screened_energy.sum(1) + 1e-6)
    features["hidden_share"] = hidden_share
    # Straight-line distance in 3D, so a point high above a road counts as far from it.
    up = np.maximum(height_above_ground - SOURCE_HEIGHT, 0)
    features["dist_major_road"] = np.minimum(np.hypot(major_tree.query(np.c_[east, north])[0], up), 400)
    features["dist_any_road"] = np.minimum(np.hypot(tree.query(np.c_[east, north])[0], up), 400)
    features["enclosure"] = enclosure(east, north, rec_z, obstacle, grid)
    return features


def enclosure(east, north, rec_z, obstacle, grid, directions=16, reach=25):
    """Share of directions with a wall at least as high as the listener within 25 m.

    High values mean a street canyon or courtyard, where sound bounces around.
    """
    angles = np.arange(directions) * 2 * np.pi / directions
    steps = np.arange(2, reach + 1, 1.0)
    result = np.zeros(len(east))
    for angle in angles:
        px = east[:, None] + np.cos(angle) * steps[None, :]
        py = north[:, None] + np.sin(angle) * steps[None, :]
        row, col = grid.index(px, py)
        wall = (obstacle[row, col] >= rec_z[:, None]).any(axis=1)
        result += wall
    return result / directions


# --------------------------------------------------------------------------
def main():
    started = time.time()
    say("Reading heights and OpenStreetMap ...")
    ground, above_ground, transform = load_rasters()
    grid = Grid(transform, ground.shape)
    buildings, roads = load_osm()
    shapes = [(b["geometry"], i) for i, b in enumerate(buildings, start=1)]
    id_raster = raster_features.rasterize(shapes, out_shape=ground.shape, transform=transform,
                                          fill=0, dtype="int32")
    inside = id_raster > 0
    # Obstacles are buildings only: trees, lamp posts and buses don't count.
    obstacle = ground + np.where(inside, above_ground, 0)

    for number, building in enumerate(buildings, start=1):
        values = above_ground[id_raster == number]
        height = float(np.percentile(values, 95)) if values.size else 0.0
        building["height"] = round(height, 1)
        building["floors"] = int(height // FLOOR_HEIGHT) if height >= 2.5 else 0
    sources = road_points(roads)
    say("   %d buildings, %d road stretches -> %d road points" % (len(buildings), len(roads), len(sources[0])))

    # ---- training points: the official grid at 4 m ---------------------
    with rasterio.open(RAW / "road_lden.tif") as source:
        lden = source.read(1)
        noise_transform = source.transform
    with rasterio.open(RAW / "road_lnight.tif") as source:
        lnight = source.read(1)
    rows, cols = np.indices(lden.shape)
    east = noise_transform.c + (cols + 0.5) * noise_transform.a
    north = noise_transform.f + (rows + 0.5) * noise_transform.e
    left, top = transform.c, transform.f
    right, bottom = left + ground.shape[1], top - ground.shape[0]
    # Share of each 10 m cell covered by buildings.
    covered = np.zeros(lden.shape)
    for i in range(lden.shape[0]):
        for j in range(lden.shape[1]):
            r0, c0 = int(top - north[i, j] - 5), int(east[i, j] - 5 - left)
            if 0 <= r0 and 0 <= c0:
                covered[i, j] = inside[r0:r0 + 10, c0:c0 + 10].mean() if inside[r0:r0 + 10, c0:c0 + 10].size else 1
    keep = ((east > left + TRAIN_MARGIN) & (east < right - TRAIN_MARGIN)
            & (north > bottom + TRAIN_MARGIN) & (north < top - TRAIN_MARGIN)
            & (covered < 0.3) & (lden > 0))
    say("   official grid: %d points kept for training (outdoors, away from the edge)" % keep.sum())
    say("Working out what the 4 m points can see ...")
    features = visibility_features(east[keep], north[keep], np.full(keep.sum(), OFFICIAL_HEIGHT),
                                   sources, obstacle, ground, grid, "4 m grid")
    points_4m = pd.DataFrame({"x": east[keep], "y": north[keep], "height": OFFICIAL_HEIGHT,
                              "lden": lden[keep], "lnight": np.where(lnight[keep] > 0, lnight[keep], np.nan)})
    points_4m = pd.concat([points_4m, features], axis=1)
    points_4m.to_csv(OUT / "points_4m.csv", index=False, float_format="%.3f")

    # ---- prediction points: every floor of every facade ----------------
    centre = ((left + right) / 2, (top + bottom) / 2)
    parts = building_parts(id_raster, above_ground, transform, centre)
    (OUT / "parts.json").write_text(json.dumps({"centre": centre, "floor_height": FLOOR_HEIGHT, "parts": [
        {"building": p["building"], "floors": p["floors"],
         "rings": [np.round(np.asarray(r.coords), 1).tolist()
                   for r in [p["geometry"].exterior, *p["geometry"].interiors]]} for p in parts]}))
    say("   %d blocks of equal height in the map square" % len(parts))
    facades = facade_points(parts, id_raster, above_ground, grid)
    say("   %d facade points over %d buildings (up to floor %d)"
        % (len(facades), facades["building"].nunique(), facades["floor"].max()))
    say("Working out what every floor can see ...")
    features = visibility_features(facades["x"].to_numpy(), facades["y"].to_numpy(),
                                   facades["height"].to_numpy(), sources, obstacle, ground, grid, "floors")
    points_floors = pd.concat([facades.reset_index(drop=True), features], axis=1)
    points_floors.to_csv(OUT / "points_floors.csv", index=False, float_format="%.3f")

    # ---- buildings for drawing -----------------------------------------
    to_degrees = Transformer.from_crs(27700, 4326, always_xy=True)
    collection = {"type": "FeatureCollection", "features": []}
    for number, building in enumerate(buildings, start=1):
        shape = shapely.transform(building["geometry"],
                                  lambda xy: np.c_[to_degrees.transform(xy[:, 0], xy[:, 1])])
        collection["features"].append({
            "type": "Feature", "geometry": mapping(shape),
            "properties": {"building": number, "osm_id": building["osm_id"], "kind": building["kind"],
                           "name": building["name"], "height": building["height"],
                           "floors": building["floors"]}})
    (OUT / "buildings.geojson").write_text(json.dumps(collection))
    roads_out = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"group": road["group"]},
         "geometry": mapping(shapely.transform(road["geometry"],
                                               lambda xy: np.c_[to_degrees.transform(xy[:, 0], xy[:, 1])]))}
        for road in roads]}
    (OUT / "roads.geojson").write_text(json.dumps(roads_out))

    draw_inputs(lden, noise_transform, above_ground, inside, transform, points_4m, roads)
    report(points_4m, points_floors, buildings, time.time() - started)


def draw_inputs(lden, noise_transform, above_ground, inside, transform, points_4m, roads):
    """Three panels side by side, to check that the datasets line up."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts = HERE / "charts"
    charts.mkdir(exist_ok=True)
    left, top = transform.c, transform.f
    extent = (left, left + above_ground.shape[1], top - above_ground.shape[0], top)
    noise_extent = (noise_transform.c, noise_transform.c + lden.shape[1] * noise_transform.a,
                    noise_transform.f + lden.shape[0] * noise_transform.e, noise_transform.f)
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.4), facecolor="#14161a")
    heights = np.where(inside, above_ground, np.nan)
    panels = [
        ("Official road noise at 4 m (Lden, dB)", np.where(lden > 0, lden, np.nan), noise_extent, "magma", (45, 80)),
        ("Building heights from laser scans (m)", heights, extent, "viridis", (0, 60)),
    ]
    for axis, (title, data, box, colours, limits) in zip(axes, panels):
        image = axis.imshow(data, extent=box, cmap=colours, vmin=limits[0], vmax=limits[1],
                            interpolation="nearest")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02).ax.tick_params(colors="#cfd3da")
        axis.set_title(title, color="#e8eaed", fontsize=12)
    axis = axes[2]
    axis.imshow(np.where(inside, 1.0, np.nan), extent=extent, cmap="Greys", vmin=0, vmax=3)
    scatter = axis.scatter(points_4m["x"], points_4m["y"], c=100 * points_4m["hidden_share"],
                           s=2.2, cmap="cividis", vmin=0, vmax=100, marker="s", linewidths=0)
    for road in roads:
        if road["group"] in ("trunk", "primary"):
            xs, ys = road["geometry"].xy
            axis.plot(xs, ys, color="#ff6b4a", linewidth=0.8)
    figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.02).ax.tick_params(colors="#cfd3da")
    axis.set_title("Share of road sound hidden behind buildings (%)", color="#e8eaed", fontsize=12)
    for axis in axes:
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("#14161a")
    figure.suptitle("Elephant & Castle, 1.6 km square: the inputs (red lines = main roads)",
                    color="#e8eaed", fontsize=14)
    figure.tight_layout()
    figure.savefig(charts / "1_inputs.png", dpi=130, facecolor=figure.get_facecolor())
    plt.close(figure)


def report(points_4m, points_floors, buildings, seconds):
    say("=" * 62)
    say("REPORT  step 2")
    say("=" * 62)
    heights = np.array([b["height"] for b in buildings])
    say("Buildings: %d   median height %.1f m   taller than 30 m: %d   taller than 60 m: %d"
        % (len(buildings), np.median(heights), (heights > 30).sum(), (heights > 60).sum()))
    tallest = sorted(buildings, key=lambda b: -b["height"])[:6]
    for b in tallest:
        say("   %5.1f m  %2d floors  %s %s" % (b["height"], b["floors"], b["name"] or "(no name)", b["osm_id"]))
    say("4 m training points: %d" % len(points_4m))
    say("   share of road energy hidden behind buildings: middle %.0f%%" % (100 * points_4m["hidden_share"].median()))
    for column in ("open_all", "screened_all", "dist_major_road", "enclosure"):
        r = np.corrcoef(points_4m[column], points_4m["lden"])[0, 1]
        say("   link with official noise, %-16s r = %+.2f" % (column, r))
    say("Facade points: %d on %d buildings" % (len(points_floors), points_floors["building"].nunique()))
    by_floor = points_floors.groupby(np.minimum(points_floors["floor"], 20))
    say("   floor  points  road hidden  'seen' road level")
    for floor, group in by_floor:
        if floor in (0, 1, 2, 3, 5, 8, 12, 16, 20):
            say("   %s%-4d %6d    %5.0f%%      %6.1f dB" % ("" if floor < 20 else "20+"[:0], floor, len(group),
                100 * group["hidden_share"].median(), group["screened_all"].median()))
    say("-" * 62)
    say("Done in %.0f s. Files: data/points_4m.csv, data/points_floors.csv," % seconds)
    say("data/buildings.geojson, data/roads.geojson, charts/1_inputs.png")
    say("=" * 62)


if __name__ == "__main__":
    main()
