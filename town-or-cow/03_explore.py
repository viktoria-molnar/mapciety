"""
Town or Cow - step 3: look at the data.

Reads data/municipalities.gpkg (made by step 2) and draws five charts into
the charts/ folder:

  1_urban_share.png        where people live: city, town or suburb, rural
  2_city_sizes.png         how the population is split across small and big places
  3_features_by_class.png  how each feature differs between the three classes
  4_density_vs_height.png  two features against each other, one dot per municipality
  5_map_labels.png         a first map of the official classes

Run it from the town-or-cow folder:

    python3 03_explore.py
"""

import sqlite3
from pathlib import Path

import geopandas
import matplotlib
matplotlib.use("Agg")  # draw straight into files, without opening windows
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import pandas

# --- Settings ---------------------------------------------------------------

DATA = Path(__file__).parent / "data" / "municipalities.gpkg"
CHARTS_DIR = Path(__file__).parent / "charts"

COUNTRIES = {"CH": "Switzerland", "HU": "Hungary"}
# For drawing, each country gets its own national map projection, so that
# north points straight up. In the shared European projection the data is
# stored in, Hungary comes out tilted.
MAP_PROJECTIONS = {"CH": "EPSG:2056", "HU": "EPSG:23700"}
CLASSES = ["city", "town or suburb", "rural"]

# One colour per class, the same in every chart and map: one teal blue that
# gets darker the more urban the class is. Countries are never told apart by
# colour: they always get their own panel.
COLOURS = {"city": "#0b5563", "town or suburb": "#4fa9b5", "rural": "#cde8eb"}
NEUTRAL = "#52514e"
BACKGROUND = "#fcfcfb"
INK = "#0b0b0b"
GRID = "#e4e3df"

SIZE_BANDS = ["under 2,000", "2,000 to 10,000", "10,000 to 50,000",
              "50,000 to 200,000", "over 200,000"]

# The features to compare: column, readable name, whether to use a scale
# where each step is "ten times more" (good for values that range from 10 to
# 10,000), and what to multiply by (100 turns a share into a percentage).
FEATURES = [
    ("density", "Inhabitants per km2", True, 1),
    ("built_up_share", "Share of land built-up (%)", True, 100),
    ("town_average_height_m", "Average building height (m),\nwithout industry", False, 1),
    ("town_built_up_m2_per_person", "Footprint per inhabitant (m2),\nwithout industry", True, 1),
    ("town_built_volume_m3_per_person", "Volume per inhabitant (m3),\nwithout industry", True, 1),
    ("industrial_share", "Share of building volume\non industrial land (%)", False, 100),
]

plt.rcParams.update({
    "figure.facecolor": BACKGROUND, "axes.facecolor": BACKGROUND,
    "savefig.facecolor": BACKGROUND, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": NEUTRAL,
    "xtick.color": NEUTRAL, "ytick.color": NEUTRAL,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


# --- Reading ----------------------------------------------------------------

def read_table():
    """The municipality table without the boundaries, read with SQL."""
    connection = sqlite3.connect(DATA)
    columns = ", ".join(feature[0] for feature in FEATURES)
    table = pandas.read_sql(
        f"SELECT country, name, label_name, population, {columns} FROM municipalities",
        connection)
    connection.close()
    return table


def size_band(population):
    """Which size band a municipality with this many inhabitants belongs to."""
    if population < 2_000:
        return SIZE_BANDS[0]
    elif population < 10_000:
        return SIZE_BANDS[1]
    elif population < 50_000:
        return SIZE_BANDS[2]
    elif population < 200_000:
        return SIZE_BANDS[3]
    else:
        return SIZE_BANDS[4]


def save(figure, name):
    CHARTS_DIR.mkdir(exist_ok=True)
    figure.savefig(CHARTS_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"  saved charts/{name}")


def class_legend(figure):
    """One legend for the whole figure: which colour is which class."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[name]) for name in CLASSES]
    figure.legend(handles, CLASSES, loc="upper right", ncol=3, frameon=False)


# --- Charts -----------------------------------------------------------------

def chart_urban_share(table):
    """Two stacked bars per country: share of people and share of municipalities."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 2.8), sharey=True)
    measures = [("Share of inhabitants", table.groupby(["country", "label_name"])["population"].sum()),
                ("Share of municipalities", table.groupby(["country", "label_name"]).size())]

    for axis, (title, counts) in zip(axes, measures):
        print(f"\n{title} (%)")
        for row, country in enumerate(COUNTRIES):
            shares = 100 * counts[country] / counts[country].sum()
            print(f"  {country}: " + ", ".join(f"{name} {shares[name]:.1f}" for name in CLASSES))
            start = 0
            for name in CLASSES:
                axis.barh(row, shares[name], left=start, height=0.55, color=COLOURS[name])
                if start > 0:  # a thin gap between two segments
                    axis.plot([start, start], [row - 0.275, row + 0.275],
                              color=BACKGROUND, linewidth=1)
                # The axis is flipped (first country on top), so "above the
                # bar" is a smaller number.
                if shares[name] >= 4:
                    axis.text(start + shares[name] / 2, row - 0.32, f"{shares[name]:.0f}%",
                              ha="center", va="bottom", fontsize=9)
                else:
                    # A sliver is too thin to carry a label, so it gets one a
                    # line higher, with its name and one decimal.
                    axis.text(start, row - 0.50, f"{shares[name]:.1f}% {name}",
                              ha="left", va="bottom", fontsize=9)
                start += shares[name]
        axis.set_title(title)
        axis.set_yticks(range(len(COUNTRIES)), list(COUNTRIES.values()))
        axis.set_ylim(len(COUNTRIES) - 0.5, -0.8)  # first country on top
        axis.set_xlim(0, 100)
        axis.xaxis.set_major_formatter("{x:.0f}%")
        axis.grid(axis="y", visible=False)

    class_legend(figure)
    figure.suptitle("Where people live", x=0.01, y=1.12, ha="left", fontsize=13, fontweight="bold")
    save(figure, "1_urban_share.png")


def chart_city_sizes(table):
    """Share of each country's population living in small, medium and big places."""
    table = table.assign(band=[size_band(people) for people in table["population"]])
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)

    print("\nShare of inhabitants by size of municipality (%)")
    for axis, (country, country_name) in zip(axes, COUNTRIES.items()):
        rows = table[table["country"] == country]
        people = rows.groupby("band")["population"].sum().reindex(SIZE_BANDS, fill_value=0)
        places = rows.groupby("band").size().reindex(SIZE_BANDS, fill_value=0)
        shares = 100 * people / people.sum()
        print(f"  {country}: " + ", ".join(f"{band} {shares[band]:.1f}" for band in SIZE_BANDS))

        axis.bar(range(len(SIZE_BANDS)), shares, width=0.6, color=NEUTRAL)
        for position, band in enumerate(SIZE_BANDS):
            unit = "place" if places[band] == 1 else "places"
            axis.text(position, shares[band] + 1, f"{shares[band]:.0f}%\n{places[band]:,} {unit}",
                      ha="center", va="bottom", fontsize=9)
        axis.set_title(country_name)
        axis.set_xticks(range(len(SIZE_BANDS)), [band.replace(" to ", "\nto ") for band in SIZE_BANDS])
        axis.set_xlabel("Inhabitants of the municipality")
        axis.yaxis.set_major_formatter("{x:.0f}%")
        axis.grid(axis="x", visible=False)
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.25)

    figure.suptitle("How the population is split across small and big municipalities",
                    x=0.01, y=1.04, ha="left", fontsize=13, fontweight="bold")
    save(figure, "2_city_sizes.png")


def chart_features_by_class(table):
    """One panel per feature. Each box shows where the middle half of the
    municipalities of one class lie, the line inside is the median."""
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5))

    for axis, (column, title, tenfold_steps, times) in zip(axes.ravel(), FEATURES):
        position = 0
        ticks = []
        for country in COUNTRIES:
            for name in CLASSES:
                values = table.loc[(table["country"] == country) & (table["label_name"] == name),
                                   column].dropna() * times
                if tenfold_steps:
                    values = values[values > 0]  # this kind of scale cannot show zero
                axis.boxplot(values, positions=[position], widths=0.6, whis=(5, 95),
                             showfliers=False, patch_artist=True,
                             boxprops={"facecolor": COLOURS[name], "edgecolor": BACKGROUND},
                             medianprops={"color": INK, "linewidth": 1.5},
                             whiskerprops={"color": NEUTRAL}, capprops={"color": NEUTRAL})
                position += 1
            ticks.append(position - 2)
            position += 0.8  # a gap between the two countries
        axis.set_title(title)
        axis.set_xticks(ticks, list(COUNTRIES.values()))
        axis.grid(axis="x", visible=False)
        if tenfold_steps:
            axis.set_yscale("log")
            # tick marks at 1, 2, 5, 10, 20, 50, ... written as plain numbers
            axis.yaxis.set_major_locator(LogLocator(subs=[1, 2, 5]))
            axis.yaxis.set_major_formatter("{x:,g}")
            axis.yaxis.set_minor_formatter(NullFormatter())

    class_legend(figure)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.suptitle("How the features differ between the classes",
                    x=0.01, y=0.985, ha="left", fontsize=13, fontweight="bold")
    figure.text(0.01, 0.94, "Box: the middle half of the municipalities. Line: the median. "
                "Whiskers: all but the lowest and highest 5%.", color=NEUTRAL, fontsize=9)
    save(figure, "3_features_by_class.png")


def chart_density_vs_height(table):
    """One dot per municipality: density against building height."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for axis, (country, country_name) in zip(axes, COUNTRIES.items()):
        for name in reversed(CLASSES):  # rural first, so the few cities end up on top
            rows = table[(table["country"] == country) & (table["label_name"] == name)]
            axis.scatter(rows["density"], rows["town_average_height_m"], s=14,
                         color=COLOURS[name], alpha=0.6, edgecolor=BACKGROUND, linewidth=0.3)
        axis.set_title(country_name)
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter("{x:,g}")
        axis.set_xlabel("Inhabitants per km2")
    axes[0].set_ylabel("Average building height (m), without industry")

    class_legend(figure)
    figure.suptitle("Density and building height", x=0.01, y=1.0, ha="left",
                    fontsize=13, fontweight="bold")
    save(figure, "4_density_vs_height.png")


def chart_map_labels():
    """The official class of every municipality, as a map."""
    shapes = geopandas.read_file(DATA, layer="municipalities",
                                 columns=["country", "label_name"])
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.06, top=0.82, wspace=0.02)

    # Same scale for both countries: each map shows a window of the same
    # width and height in kilometres, centred on its country.
    parts = {country: shapes[shapes["country"] == country].to_crs(MAP_PROJECTIONS[country])
             for country in COUNTRIES}
    width = max(p.total_bounds[2] - p.total_bounds[0] for p in parts.values()) * 1.04
    tall = max(p.total_bounds[3] - p.total_bounds[1] for p in parts.values()) * 1.04

    for axis, (country, country_name) in zip(axes, COUNTRIES.items()):
        rows = parts[country]
        rows.plot(ax=axis, color=rows["label_name"].map(COLOURS),
                  edgecolor=BACKGROUND, linewidth=0.15)
        left, bottom, right, top = rows.total_bounds
        axis.set_xlim((left + right - width) / 2, (left + right + width) / 2)
        axis.set_ylim((bottom + top - tall) / 2, (bottom + top + tall) / 2)
        slot = axis.get_subplotspec().get_position(figure)
        figure.text(slot.x0 + 0.01, slot.y1 + 0.004, country_name, fontsize=11,
                    fontweight="bold", ha="left", va="bottom")
        axis.set_axis_off()

    class_legend(figure)
    figure.suptitle("Official degree of urbanisation (Eurostat, 2024)", x=0.01, y=1.0,
                    ha="left", fontsize=13, fontweight="bold")
    figure.text(0.01, 0.02, "Both countries are drawn at the same scale.",
                color=NEUTRAL, fontsize=9)
    save(figure, "5_map_labels.png")


# --- Main program -----------------------------------------------------------

def main():
    table = read_table()
    print(f"{len(table)} municipalities read from {DATA.name}")

    chart_urban_share(table)
    chart_city_sizes(table)
    print()
    chart_features_by_class(table)
    chart_density_vs_height(table)
    chart_map_labels()
    print(f"\nOpen the charts with:  open {CHARTS_DIR.name}")


if __name__ == "__main__":
    main()
