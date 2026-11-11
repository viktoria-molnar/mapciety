"""Step 4: a vertical slice through the city.

Run:  python3 04_slice.py      (after 03_model.py; takes about half a minute)

Imagine cutting the neighbourhood with a giant knife along one line and
looking at the cut from the side. Every 3 m along the line and every 2.5 m
up into the air, the model from step 3 predicts the road noise. Buildings
are drawn as solid shapes, so the sound shadow behind each one shows as a
dark wedge, and you can see how it shrinks with height.

The line runs north to south across New Kent Road, through the towers
south of it, where lower towers stand in front of taller ones.

Writes charts/5_slice.png and data/slice.json (used by the 3D map).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio import features as raster_features

HERE = Path(__file__).parent
DATA = HERE / "data"
CHARTS = HERE / "charts"

# The slice: British National Grid, metres. North end first.
START = (532205, 179250)
END = (532205, 178650)
NAME = "North to south across New Kent Road"
STEP_ALONG = 3.0
STEP_UP = 2.5
TOP = 120.0


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), HERE / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    geometry = load("02_geometry.py")
    model_step = load("03_model.py")

    print("Reading heights, buildings and roads ...", flush=True)
    ground, above_ground, transform = geometry.load_rasters()
    grid = geometry.Grid(transform, ground.shape)
    buildings, roads = geometry.load_osm()
    shapes = [(b["geometry"], i) for i, b in enumerate(buildings, start=1)]
    id_raster = raster_features.rasterize(shapes, out_shape=ground.shape, transform=transform,
                                          fill=0, dtype="int32")
    inside = id_raster > 0
    obstacle = ground + np.where(inside, above_ground, 0)
    for number, building in enumerate(buildings, start=1):
        values = above_ground[id_raster == number]
        building["height"] = float(np.percentile(values, 95)) if values.size else 0.0
    sources = geometry.road_points(roads)

    # Points along the line and up into the air.
    length = float(np.hypot(END[0] - START[0], END[1] - START[1]))
    along = np.arange(0, length + 0.01, STEP_ALONG)
    up = np.arange(1.5, TOP + 0.01, STEP_UP)
    ex = START[0] + (END[0] - START[0]) * along / length
    ny = START[1] + (END[1] - START[1]) * along / length
    # Building height across a 3 m wide strip, so thin walls aren't missed.
    side_x, side_y = -(END[1] - START[1]) / length, (END[0] - START[0]) / length
    roof = np.zeros(len(along))
    for offset in (-1.5, -0.75, 0, 0.75, 1.5):
        r, c = grid.index(ex + offset * side_x, ny + offset * side_y)
        roof = np.maximum(roof, np.where(inside[r, c], above_ground[r, c], 0))
    from scipy import ndimage
    roof = ndimage.grey_closing(roof, size=3)   # close 1-cell light wells
    A, U = np.meshgrid(np.arange(len(along)), up)
    air = U > roof[A] + 0.5
    east, north, height = ex[A[air]], ny[A[air]], U[air]
    print("   %d points in the air along %.0f m" % (air.sum(), length), flush=True)

    print("Working out what each point can see ...", flush=True)
    features = geometry.visibility_features(east, north, height, sources, obstacle, ground, grid, "slice")

    print("Training the model again on the 4 m points ...", flush=True)
    points = pd.read_csv(DATA / "points_4m.csv")
    columns = model_step.feature_columns(points)
    model = model_step.models()["Boosted trees (all features)"][1]
    image = {}
    for target, low, high in (("lden", 35, 90), ("lnight", 30, 85)):
        rows = points.dropna(subset=[target])
        model.fit(rows[columns], rows[target])
        values = np.full(air.shape, np.nan)
        values[air] = np.clip(model.predict(features[columns]), low, high)
        image[target] = values

    # Where the main roads cross the line.
    from shapely.geometry import LineString
    line = LineString([START, END])
    crossings = []
    for road in roads:
        if road["group"] in ("trunk", "primary", "secondary"):
            hit = road["geometry"].intersection(line)
            for point in getattr(hit, "geoms", [hit]):
                if point.geom_type == "Point":
                    crossings.append((line.project(point), road["group"]))

    (DATA / "slice.json").write_text(json.dumps({
        "name": NAME, "start": START, "end": END, "step_along": STEP_ALONG, "step_up": STEP_UP,
        "first_height": float(up[0]), "roof": np.round(roof, 1).tolist(),
        "lden": np.where(np.isnan(image["lden"]), 0, np.round(image["lden"], 1)).tolist(),
        "lnight": np.where(np.isnan(image["lnight"]), 0, np.round(image["lnight"], 1)).tolist(),
    }))
    # Fill the half cell between each roof and the first point above it, so colour meets the roof.
    shown = image["lden"].copy()
    for i in range(shown.shape[1]):
        valid = np.flatnonzero(~np.isnan(shown[:, i]))
        if valid.size:
            shown[:valid[0], i] = shown[valid[0], i]
    draw(along, up, roof, shown, image["lden"], crossings, buildings, roads)
    report(along, up, roof, image["lden"], crossings)


def draw(along, up, roof, lden, lden_air, crossings, buildings, roads):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ramp = LinearSegmentedColormap.from_list("sound", [
        (0, "#121019"), (0.2, "#2c1a55"), (0.4, "#6e1f6e"), (0.58, "#b8325e"),
        (0.75, "#ef6a3a"), (0.9, "#fbb14a"), (1, "#fdf0b2")])
    background, ink, soft = "#0b0c10", "#ecebf2", "#9a98aa"
    CHARTS.mkdir(exist_ok=True)
    figure = plt.figure(figsize=(16, 7.6), facecolor=background)
    axis = figure.add_axes([0.05, 0.12, 0.70, 0.74])
    axis.set_facecolor(background)
    picture = axis.imshow(lden, origin="lower", aspect="auto", cmap=ramp, vmin=45, vmax=80,
                          extent=(along[0], along[-1] + STEP_ALONG, 0, up[-1] + STEP_UP / 2),
                          interpolation="bilinear")
    levels = axis.contour(along + STEP_ALONG / 2, up, lden_air,
                          levels=[53, 60, 65, 70], colors=["#ffffff"], linewidths=[1.4, 0.6, 0.6, 0.6],
                          alpha=0.65)
    axis.clabel(levels, fmt="%d dB", fontsize=8, colors=ink)
    axis.fill_between(along + STEP_ALONG / 2, 0, roof, step="mid", color="#4a4e5c", linewidth=0)
    axis.plot(along + STEP_ALONG / 2, roof, drawstyle="steps-mid", color="#9aa0b4", linewidth=0.8)
    names = {"trunk": "main road", "primary": "main road", "secondary": "road"}
    done = set()
    for distance, group in sorted(crossings):
        key = round(distance / 20)
        if key in done:
            continue
        done.add(key)
        axis.text(distance, 2, names[group], color="#121019", ha="center", va="bottom", fontsize=9,
                  fontweight="bold", rotation=90)
    axis.set_xlim(along[0], along[-1])
    axis.set_ylim(0, up[-1])
    axis.set_xlabel("metres along the slice  (north on the left, south on the right)", color=soft)
    axis.set_ylabel("height above the street (m)", color=soft)
    axis.tick_params(colors=soft)
    for side in axis.spines.values():
        side.set_color("#2a2c36")
    bar = figure.colorbar(picture, ax=axis, fraction=0.025, pad=0.01)
    bar.set_label("predicted road noise, Lden (dB)", color=soft)
    bar.ax.tick_params(colors=soft)
    figure.text(0.05, 0.93, "A vertical slice through Elephant & Castle", color=ink, fontsize=17,
                fontweight="bold")
    figure.text(0.05, 0.895, "%s. Grey shapes are buildings; dark wedges are their sound shadow. "
                "The thick white line is the WHO guideline, 53 dB." % NAME, color=soft, fontsize=10.5)

    # Small map showing where the slice runs.
    inset = figure.add_axes([0.80, 0.18, 0.18, 0.62])
    inset.set_facecolor(background)
    cx, cy = (START[0] + END[0]) / 2, (START[1] + END[1]) / 2
    for building in buildings:
        shape = building["geometry"]
        if shape.distance(__import__("shapely").geometry.Point(cx, cy)) > 420:
            continue
        for polygon in getattr(shape, "geoms", [shape]):
            xs, ys = polygon.exterior.xy
            tone = "#6b7080" if building["height"] > 30 else "#33363f"
            inset.fill(xs, ys, color=tone, linewidth=0)
    for road in roads:
        if road["group"] in ("trunk", "primary"):
            xs, ys = road["geometry"].xy
            inset.plot(xs, ys, color="#ef6a3a", linewidth=1.2)
    inset.plot([START[0], END[0]], [START[1], END[1]], color="#fdf0b2", linewidth=2)
    inset.text(START[0], START[1] + 12, "N", color=ink, ha="center", fontsize=9)
    inset.set_xlim(cx - 330, cx + 330)
    inset.set_ylim(cy - 400, cy + 400)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    for side in inset.spines.values():
        side.set_color("#2a2c36")
    inset.set_title("where the slice runs\n(light = taller than 30 m, red = main roads)", color=soft, fontsize=9)
    figure.savefig(CHARTS / "5_slice.png", dpi=150, facecolor=background)
    plt.close(figure)


def report(along, up, roof, lden, crossings):
    print("=" * 62)
    print("REPORT  step 4 (slice)")
    print("=" * 62)
    print("Slice %.0f m long, up to %.0f m. Main roads crossed at: %s m"
          % (along[-1], up[-1], ", ".join("%.0f" % d for d, g in sorted(crossings) if g in ("trunk", "primary"))))
    print("Predicted noise (Lden) at a few heights:")
    print("   along   roof    1.5 m   10 m   20 m   40 m   80 m")
    for i in range(0, len(along), 20):
        cells = []
        for h in (1.5, 10, 20, 40, 80):
            j = int(round((h - up[0]) / STEP_UP))
            v = lden[j, i]
            cells.append("   -  " if np.isnan(v) else "%5.1f " % v)
        print("   %4.0f m %5.0f m %s" % (along[i], roof[i], " ".join(cells)))
    print("-" * 62)
    print("Files: charts/5_slice.png, data/slice.json")
    print("=" * 62)


if __name__ == "__main__":
    main()
