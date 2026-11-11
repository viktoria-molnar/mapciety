"""
Town or Cow - step 6: do houses or home ownership explain the difference?

Draws the two overlay maps of step 4 again, each time with a third row of
census data underneath:

    charts/10_map_census_houses.png      share of homes in single-family houses
    charts/11_map_census_owners.png      share of households owning their home

and tests which of the two explains better where the satellite and Eurostat
disagree.

Run it from the town-or-cow folder, after steps 4 and 5:

    python3 06_census_maps.py
"""

import importlib.util
from pathlib import Path

import geopandas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"


def load_step(file_name):
    spec = importlib.util.spec_from_file_location(file_name[:-3], HERE / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


swap = load_step("04_train_and_swap.py")

MEASURES = {
    "house_share": {
        "file": "10_map_census_houses.png",
        "title": "homes in single-family houses",
        "legend": "Share of homes in single-family houses",
        # light to dark grey-green: neutral, so it does not compete with the class colours
        "colours": LinearSegmentedColormap.from_list("houses", ["#f4f3ee", "#8c9a6b", "#2f3a1f"]),
        "how": {"CH": "Counted, {year}. Register of buildings and dwellings",
                "HU": "Estimated, {year} census: homes not heated building-wide.\n"
                      "Too high in old city centres (flats with their own boiler count as houses)"},
    },
    "owner_share": {
        "file": "11_map_census_owners.png",
        "title": "households owning their home",
        "legend": "Share of households owning their home",
        "colours": LinearSegmentedColormap.from_list("owners", ["#f4f3ee", "#a48a6a", "#4a3523"]),
        "how": {"CH": "Estimated, {year} census: occupied homes minus rented ones.\n"
                      "The newest ownership figures per municipality",
                "HU": "Counted, {year} census"},
    },
}


def read_table():
    table = geopandas.read_file(DATA_DIR / "predictions.gpkg", layer="predictions")
    census = pandas.read_csv(DATA_DIR / "census.csv").drop(columns=["country"])
    volume = geopandas.read_file(DATA_DIR / "municipalities.gpkg", layer="municipalities",
                                 columns=["id", "town_built_volume_m3_per_person"],
                                 ignore_geometry=True)
    return table.merge(census, on="id", how="left").merge(volume, on="id", how="left")


def detail(measure, settings):
    def text(rows):
        country = rows["country"].iloc[0]
        year_column = "house_year" if measure == "house_share" else "owner_year"
        year = int(rows[year_column].dropna().iloc[0]) if rows[year_column].notna().any() else "?"
        typical = 100 * rows[measure].median()
        return (f"{settings['how'][country].format(year=year)}\n"
                f"Typical municipality: {typical:.0f}%. Grey: no data")
    return text


def evaluate(table, measure):
    """How well does the measure explain the satellite's 'more urban' calls?

    1. Does it go with more building per person? (the suspected mechanism)
       Spearman's rank correlation: +1 = always together, 0 = unrelated.
    2. Among places Eurostat calls rural, can it tell the ones the satellite
       upgrades from the ones it leaves alone? AUC: pick one upgraded place
       and one unchanged place at random; how often does the upgraded one
       have the higher value? 50% = no better than a coin, 100% = always.
    """
    results = {}
    for country in swap.COUNTRIES:
        rows = table[(table["country"] == country) & table[measure].notna()]
        rho, _ = spearmanr(rows[measure], rows["town_built_volume_m3_per_person"], nan_policy="omit")
        rural = rows[rows["label"] == 3]
        upgraded = rural["guess_own"] < rural["label"]
        auc = roc_auc_score(upgraded, rural[measure]) if 0 < upgraded.sum() < len(rural) else float("nan")
        results[country] = (rho, 100 * auc, int(upgraded.sum()), len(rural))
    return results


def main():
    table = read_table()
    print(f"{len(table)} municipalities, census data for "
          f"{table['house_share'].notna().sum()} (houses) and {table['owner_share'].notna().sum()} (owners)")

    summary = {}
    for measure, settings in MEASURES.items():
        swap.chart_overlays(table, extra={"column": measure, "colours": settings["colours"],
                                          "title": settings["title"], "legend": settings["legend"],
                                          "detail": detail(measure, settings)},
                            file_name=settings["file"])
        summary[measure] = evaluate(table, measure)

    print("\n=== Which measure explains the difference better? ===")
    print("  rho: rank correlation with building volume per person (the suspected mechanism)")
    print("  AUC: how well it picks out the rural places the satellite calls more urban (50% = coin toss)\n")
    for measure, results in summary.items():
        print(f"  {MEASURES[measure]['title']}")
        for country, (rho, auc, upgraded, rural) in results.items():
            print(f"    {swap.COUNTRIES[country]:<12} rho {rho:+.2f}   AUC {auc:3.0f}%   "
                  f"({upgraded} of {rural} rural places upgraded)")
    print("\nOpen the charts with:  open charts")


if __name__ == "__main__":
    main()
