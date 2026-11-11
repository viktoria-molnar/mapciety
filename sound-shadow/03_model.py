"""Step 3: learn noise from geometry, test it, then predict every floor.

Run:  python3 03_model.py      (after 02_geometry.py; takes about a minute)

The model only ever sees geometry: how much road of each type a point can
see, how far away it is, how much is hidden behind buildings, and how
enclosed the spot is. It never sees the height of the point itself, so it
has to get the effect of height through what changes with height (roads
further away, more of them visible over neighbouring roofs).

Testing: the square is cut into 200 m blocks. Each model is trained on some
blocks and tested on the others, five times over, so every block is tested
once by a model that never saw it. For comparison it is also done the naive
way, with points shuffled at random, which looks better than it is because
neighbouring points are almost identical.

Writes:
  data/predictions_floors.csv   predicted noise for every floor of every facade
  charts/2_model_test.png       how well each model does on unseen streets
  charts/3_floor_profiles.png   noise floor by floor for some buildings
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
DATA = HERE / "data"
CHARTS = HERE / "charts"
BLOCK = 200  # metres
SEED = 7


def say(*text):
    print(*text, flush=True)


def models():
    return {
        "Average everywhere": None,
        "Distance to main road only": ("distance", make_pipeline(StandardScaler(), LinearRegression())),
        "Straight line (all features)": ("all", make_pipeline(StandardScaler(), LinearRegression())),
        "Boosted trees (all features)": ("all", HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, random_state=SEED)),
    }


def feature_columns(table):
    skip = {"x", "y", "height", "lden", "lnight", "building", "part", "wall", "facing_x", "facing_y",
            "x0", "y0", "x1", "y1", "floor", "facade_height"}
    return [c for c in table.columns if c not in skip]


def distance_columns(table):
    out = pd.DataFrame({"log_distance": np.log10(table["dist_major_road"] + 5)})
    return out


def evaluate(points, target):
    y = points[target].to_numpy()
    blocks = (points["x"] // BLOCK).astype(int) * 1000 + (points["y"] // BLOCK).astype(int)
    columns = feature_columns(points)
    results, held_out = [], {}
    for name, spec in models().items():
        for split_name, splitter, groups in (("unseen blocks", GroupKFold(5), blocks),
                                             ("random points", KFold(5, shuffle=True, random_state=SEED), None)):
            if spec is None:
                prediction = np.zeros_like(y)
                for train, test in splitter.split(points, y, groups):
                    prediction[test] = y[train].mean()
                fit_error = np.abs(y - y.mean()).mean()
            else:
                kind, model = spec
                X = distance_columns(points) if kind == "distance" else points[columns]
                prediction = cross_val_predict(model, X, y, cv=splitter, groups=groups)
                fit_error = np.abs(model.fit(X, y).predict(X) - y).mean()
            error = np.abs(prediction - y)
            results.append({"model": name, "test": split_name, "mae": error.mean(),
                             "within_3db": (error <= 3).mean(), "train_mae": fit_error})
            if split_name == "unseen blocks":
                held_out[name] = prediction
    return pd.DataFrame(results), held_out


def wall_check(floors, predictions):
    """Compare floor 1 (about 4.7 m up) with the official value right in front of the wall."""
    with rasterio.open(DATA / "raw" / "road_lden.tif") as source:
        lden = source.read(1)
        t = source.transform
    cols = np.floor((floors["x"].to_numpy() - t.c) / t.a).astype(int)
    rows = np.floor((floors["y"].to_numpy() - t.f) / t.e).astype(int)
    ok = (floors["floor"].to_numpy() == 1) & (rows >= 0) & (cols >= 0) & (rows < lden.shape[0]) & (cols < lden.shape[1])
    official = lden[rows[ok], cols[ok]]
    predicted = predictions[ok]
    good = official > 0
    return official[good], predicted[good]


def draw_test(results, held_out, points, wall):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    background, ink, soft = "#14161a", "#e8eaed", "#8a919c"
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8), facecolor=background)
    for axis in axes:
        axis.set_facecolor(background)
        axis.tick_params(colors=soft)
        for side in axis.spines.values():
            side.set_color("#3a3f47")

    axis = axes[0]
    names = list(models())
    blocks = results[results["test"] == "unseen blocks"].set_index("model").loc[names]
    shuffled = results[results["test"] == "random points"].set_index("model").loc[names]
    ypos = np.arange(len(names))
    axis.barh(ypos + 0.18, blocks["mae"], height=0.34, color="#f2a541", label="tested on unseen 200 m blocks")
    axis.barh(ypos - 0.18, shuffled["mae"], height=0.34, color="#5b6472", label="tested on random points (too easy)")
    for y, value in zip(ypos, blocks["mae"]):
        axis.text(value + 0.1, y + 0.18, "%.1f dB" % value, va="center", color=ink, fontsize=10)
    axis.set_yticks(ypos)
    axis.set_yticklabels(names, color=ink)
    axis.invert_yaxis()
    axis.set_xlabel("average error (dB), lower is better", color=soft)
    axis.legend(frameon=False, labelcolor=ink, loc="lower right", fontsize=9)
    axis.set_title("How far off each model is", color=ink)

    axis = axes[1]
    best = held_out["Boosted trees (all features)"]
    axis.hexbin(points["lden"], best, gridsize=45, cmap="magma", mincnt=1, bins="log")
    axis.plot([40, 86], [40, 86], color="#5fd3c6", linewidth=1)
    axis.set_xlim(40, 86)
    axis.set_ylim(40, 86)
    axis.set_xlabel("official noise at 4 m (dB)", color=soft)
    axis.set_ylabel("predicted, block never seen (dB)", color=soft)
    axis.set_title("Boosted trees on unseen blocks", color=ink)

    axis = axes[2]
    official, predicted = wall
    axis.hexbin(official, predicted, gridsize=40, cmap="viridis", mincnt=1, bins="log")
    axis.plot([40, 86], [40, 86], color="#f2a541", linewidth=1)
    axis.set_xlim(40, 86)
    axis.set_ylim(40, 86)
    axis.set_xlabel("official noise in the 10 m cell in front of the wall (dB)", color=soft)
    axis.set_ylabel("predicted on the facade, 1st floor (dB)", color=soft)
    axis.set_title("Check on the facades: 1st floor vs official", color=ink)
    figure.tight_layout()
    figure.savefig(CHARTS / "2_model_test.png", dpi=130, facecolor=background)
    plt.close(figure)


def draw_profiles(out, buildings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    background, ink, soft = "#14161a", "#e8eaed", "#8a919c"
    # Towers: buildings with the most facade points at 20 floors or more.
    high = out[out["floor"] >= 20].groupby("building").size().sort_values(ascending=False)
    picks = list(high.index[:4])
    short = out.groupby("building")["floor"].max()
    short = short[(short >= 1) & (short <= 3)].index
    ground_floor = out[(out["floor"] == 0) & out["building"].isin(short)].groupby("building")["lden"].median()
    if len(ground_floor):
        picks += [ground_floor.idxmax(), ground_floor.idxmin()]
    figure, axes = plt.subplots(1, len(picks), figsize=(3.1 * len(picks), 6), facecolor=background, sharex=True)
    plt.subplots_adjust(wspace=0.35)
    for axis, number in zip(np.atleast_1d(axes), picks):
        rows = out[out["building"] == number]
        # Use only the stretches of wall that rise to (nearly) the full height.
        full = rows[rows["facade_height"] >= 0.8 * rows["facade_height"].max()]
        ground = full[full["floor"] == full["floor"].min()].set_index("wall")["lden"]
        share = max(1, len(ground) // 4)
        for xy, colour, label in ((ground.nlargest(share).index, "#f2a541", "loudest side"),
                                  (ground.nsmallest(share).index, "#5fd3c6", "quietest side")):
            side = full[full["wall"].isin(xy)]
            profile = side.groupby("floor")["lden"].median()
            axis.plot(profile.values, profile.index + 1, color=colour, linewidth=2, label=label,
                      marker="o" if len(profile) < 6 else None)
        axis.set_facecolor(background)
        axis.tick_params(colors=soft)
        for spine in axis.spines.values():
            spine.set_color("#3a3f47")
        info = buildings.get(number, {})
        floors_here = int(rows["floor"].max()) + 1
        title = info.get("name") or ("tower, no name" if floors_here > 10 else "low building")
        axis.set_title("%s\n%d floors" % (title[:24], floors_here), color=ink, fontsize=10)
        axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        axis.set_ylim(0.5, floors_here + 0.5)
        axis.set_xlabel("dB (Lden)", color=soft)
    np.atleast_1d(axes)[0].set_ylabel("floor (1 = ground floor)", color=soft)
    np.atleast_1d(axes)[0].legend(frameon=False, labelcolor=ink, fontsize=9)
    figure.suptitle("Predicted road noise floor by floor (higher floors are extrapolated)", color=ink)
    figure.tight_layout()
    figure.savefig(CHARTS / "3_floor_profiles.png", dpi=130, facecolor=background)
    plt.close(figure)


def main():
    CHARTS.mkdir(exist_ok=True)
    points = pd.read_csv(DATA / "points_4m.csv")
    floors = pd.read_csv(DATA / "points_floors.csv")
    columns = feature_columns(points)
    say("Training points: %d   facade points: %d   features: %d" % (len(points), len(floors), len(columns)))

    say("Testing four models two ways ...")
    results, held_out = evaluate(points, "lden")

    say("Fitting the final models on all training points ...")
    final = models()["Boosted trees (all features)"][1]
    out = floors[["building", "part", "wall", "x", "y", "facing_x", "facing_y", "x0", "y0", "x1", "y1",
                  "floor", "height", "facade_height"]].copy()
    final.fit(points[columns], points["lden"])
    out["lden"] = np.clip(final.predict(floors[columns]), 35, 90)
    night = points.dropna(subset=["lnight"])
    final.fit(night[columns], night["lnight"])
    out["lnight"] = np.clip(final.predict(floors[columns]), 30, 85)
    out["hidden_share"] = floors["hidden_share"]
    out.round(3).to_csv(DATA / "predictions_floors.csv", index=False)

    wall = wall_check(floors, out["lden"].to_numpy())
    import json
    buildings = {f["properties"]["building"]: f["properties"]
                 for f in json.loads((DATA / "buildings.geojson").read_text())["features"]}
    draw_test(results, held_out, points, wall)
    draw_profiles(out, buildings)

    say("=" * 62)
    say("REPORT  step 3")
    say("=" * 62)
    say("Average error in dB (lower is better)")
    say("   %-30s %13s %13s %10s" % ("", "unseen blocks", "random points", "own data"))
    for name in models():
        b = results[(results["model"] == name) & (results["test"] == "unseen blocks")].iloc[0]
        r = results[(results["model"] == name) & (results["test"] == "random points")].iloc[0]
        say("   %-30s %10.2f dB %10.2f dB %7.2f dB" % (name, b["mae"], r["mae"], b["train_mae"]))
    best = results[(results["model"] == "Boosted trees (all features)") & (results["test"] == "unseen blocks")].iloc[0]
    say("   boosted trees, unseen blocks: %.0f%% of points within 3 dB" % (100 * best["within_3db"]))
    official, predicted = wall
    say("Facade check (1st floor vs official cell in front of the wall, %d points):" % len(official))
    say("   average difference %.2f dB, predicted minus official on average %+.2f dB"
        % (np.abs(predicted - official).mean(), (predicted - official).mean()))
    # Going up the same stretch of wall, split by how loud its ground floor is.
    ground = out[out["floor"] == 0].set_index("wall")["lden"]
    start = out["wall"].map(ground)
    say("Going up the same wall (middle change compared with its ground floor, dB):")
    say("   floor   loud walls (ground > 65 dB)   quiet walls (ground < 55 dB)")
    for floor in (1, 2, 4, 6, 8, 10, 15, 20, 30):
        rows = out["floor"] == floor
        loud = (out["lden"] - start)[rows & (start > 65)]
        quiet = (out["lden"] - start)[rows & (start < 55)]
        if len(loud) + len(quiet):
            say("   %4d    %+8.1f  (%4d points)          %+8.1f  (%4d points)"
                % (floor, loud.median(), len(loud), quiet.median(), len(quiet)))
    say("-" * 62)
    say("Files: data/predictions_floors.csv, charts/2_model_test.png, charts/3_floor_profiles.png")
    say("=" * 62)


if __name__ == "__main__":
    main()
