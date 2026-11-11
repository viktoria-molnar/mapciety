"""
Town or Cow - check the results.

Steps 2 and 4 produce numbers. This script tries to break them. Each check
prints PASS or FAIL with the evidence, so nobody has to take the results on
trust.

    A. Is the table built correctly?      (checks 1-5)
    B. Are the scores honest?             (checks 6-10)

Run it from the town-or-cow folder, after 02_prepare.py:

    python3 check_results.py

It takes a few minutes, because it repeats the experiment many times.
"""

import importlib.util
from pathlib import Path

import geopandas
import numpy
import pandas
import rasterio
import rasterio.mask
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict

HERE = Path(__file__).parent
RAW_DIR = HERE / "data" / "raw"
RESULTS = []  # (name, passed, evidence)


def load(file_name):
    """Load one of the numbered step scripts so we can reuse its functions."""
    spec = importlib.util.spec_from_file_location(file_name[:-3], HERE / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("02_prepare.py")
swap = load("04_train_and_swap.py")
COUNTRIES = list(swap.COUNTRIES)


def record(name, passed, evidence):
    RESULTS.append((name, passed, evidence))
    print(f"\n[{'PASS' if passed else 'FAIL'}] {name}\n       {evidence}")


def balanced(rows, guesses):
    return 100 * balanced_accuracy_score(rows["label"], guesses)


def swap_scores(table, features=None, make_model=None):
    """Balanced score for learn-in-A, test-in-B, both directions.
    Without arguments it uses the main model: the simple rule."""
    features = features or swap.RULE
    make_model = make_model or swap.make_rule
    scores = {}
    for learn, test in [("CH", "HU"), ("HU", "CH")]:
        test_rows = table[table["country"] == test]
        guess = swap.guess_other_country(table[table["country"] == learn], test_rows,
                                         features, make_model)
        scores[f"{learn}->{test}"] = balanced(test_rows, guess)
    return scores


def recompute_patches(table):
    """Recompute the built-up area sizes for every municipality in table."""
    changed = table.copy()
    for country in COUNTRIES:
        sums = prepare.sums_for_country(changed, country, prepare.read_industrial(country))
        for column in ("town_area_m3", "city_area_m3"):
            changed.loc[sums.index, column] = sums[column]
    changed["log_town_area"] = numpy.log10(changed["town_area_m3"] + 1)
    changed["log_city_area"] = numpy.log10(changed["city_area_m3"] + 1)
    return changed


# --- A. Is the table built correctly? ---------------------------------------

def check_labels_against_source(table):
    """Count the classes straight from Eurostat's Excel file and compare."""
    sheets = pandas.read_excel(RAW_DIR / "lau_labels_2024.xlsx", sheet_name=COUNTRIES)
    problems = []
    for country in COUNTRIES:
        source = sheets[country].dropna(subset=["EU LAU CODE"])["DEGURBA"].value_counts()
        ours = table[table["country"] == country]["label"].value_counts()
        for label in (1, 2, 3):
            if source.get(label, 0) != ours.get(label, 0):
                problems.append(f"{country} class {label}: source {source.get(label, 0)}, ours {ours.get(label, 0)}")
    record("1. No municipality lost or relabelled on the way from Eurostat's file",
           not problems, "; ".join(problems) or
           f"{len(table)} municipalities, class counts identical in both countries")


def check_built_up_sums(table):
    """Recompute built-up area for a sample with a different, slower method."""
    sample = table.groupby("country").sample(60, random_state=7)
    differences = []
    for country in COUNTRIES:
        with rasterio.open(RAW_DIR / f"built_up_{country}.tif") as raster:
            rows = sample[sample["country"] == country].to_crs(raster.crs)
            for _, row in rows.iterrows():
                cut, _ = rasterio.mask.mask(raster, [row.geometry], crop=True, filled=True,
                                            nodata=raster.nodata)
                cut = cut.astype("float64")
                cut[cut == raster.nodata] = 0
                again = cut.sum() / 1_000_000
                differences.append(abs(again - row["built_up_km2"]) / max(again, 1e-9))
    worst = 100 * max(differences)
    record("2. Built-up area per municipality matches an independent recount",
           worst < 1, f"120 municipalities recounted one by one; largest difference {worst:.3f}%")


def check_exact_boundaries(table):
    """Our boundaries are simplified (drawn for a 1:1 million map). For
    Switzerland we also have the exact ones. Do the guesses change?"""
    exact = geopandas.read_file(RAW_DIR / "swissboundaries3d_2026.gpkg", layer="tlm_hoheitsgebiet",
                                columns=["bfs_nummer", "icc", "einwohnerzahl"])
    exact = exact[(exact["icc"] == "CH") & (exact["einwohnerzahl"] > 0)]
    exact["geometry"] = exact.geometry.force_2d()
    exact = exact.dissolve(by="bfs_nummer").to_crs(table.crs)
    exact.index = [f"CH_CH{int(number):04d}" for number in exact.index]

    # Only municipalities that still exist unchanged in the 2026 file.
    use = table["population_source"].str.startswith("swisstopo") & table["id"].isin(exact.index)
    sharper = table.copy()
    sharper.loc[use, "geometry"] = exact.geometry.reindex(sharper.loc[use, "id"]).values
    sharper = recompute_patches(sharper)

    swiss = table["country"] == "CH"
    hungary = table[~swiss]
    before = swap.guess_other_country(hungary, table[swiss], swap.RULE, swap.make_rule)
    after = swap.guess_other_country(hungary, sharper[swiss], swap.RULE, swap.make_rule)
    same = 100 * (before == after).mean()
    move = abs(balanced(table[swiss], before) - balanced(sharper[swiss], after))
    record("3. Exact Swiss boundaries give the same guesses (95%+ identical, score within 2 points)",
           same >= 95 and move <= 2,
           f"{use.sum()} Swiss boundaries replaced by exact ones: {same:.1f}% of guesses identical, "
           f"score moves {move:.1f} points")


def check_value_ranges(table):
    shares = ["built_up_share", "industrial_share", "core_built_up_share", "dense_share"]
    problems = [name for name in shares if table[name].min() < 0 or table[name].max() > 1.0001]
    if (table["population"] <= 0).any():
        problems.append("population <= 0")
    if table["id"].duplicated().any():
        problems.append("duplicate municipalities")
    record("4. Shares lie between 0 and 1, populations are positive, no duplicates",
           not problems, ", ".join(problems) or "all four share columns, population and ids are fine")


def check_population_totals(table):
    """Compare country totals with the published national figures."""
    published = {"CH": (9_048_900, "31 Dec 2024, provisional"), "HU": (9_585_000, "1 Jan 2024")}
    notes, ok = [], True
    for country, (figure, date) in published.items():
        ours = table[table["country"] == country]["population"].sum()
        gap = 100 * (ours - figure) / figure
        ok = ok and abs(gap) < 1
        notes.append(f"{country}: ours {ours:,.0f}, published {figure:,} ({date}), gap {gap:+.2f}%")
    record("5. Population totals match the published national figures within 1%", ok, "; ".join(notes))


# --- B. Are the scores honest? ----------------------------------------------

def check_simple_is_enough(table):
    """The main model is a two-question rule. A forest of 300 trees with four
    features should not do clearly better, or the rule is too simple."""
    rule = swap_scores(table)
    forest = swap_scores(table, swap.CORES, swap.make_forest)
    ok = all(forest[name] <= rule[name] + 2 for name in rule)
    record("6. A forest of 300 trees does not beat the simple rule by more than 2 points",
           ok, "; ".join(f"{name}: rule {rule[name]:.0f}%, forest {forest[name]:.0f}%" for name in rule))


def check_neighbours_do_not_leak(table):
    """Neighbouring municipalities resemble each other and can share a feature
    value (the size of their common built-up area). If neighbours end up on
    both sides of a test, the score is flattered. Here whole 50 km squares
    are held out together."""
    notes, ok = [], True
    for country in COUNTRIES:
        rows = table[table["country"] == country]
        centre = rows.geometry.centroid
        square = (centre.x // 50_000).astype(int).astype(str) + "_" + (centre.y // 50_000).astype(int).astype(str)
        blocked = cross_val_predict(swap.make_rule(), rows[swap.RULE], rows["label"],
                                    cv=GroupKFold(n_splits=5), groups=square)
        mixed = swap.guess_own_country(rows, swap.RULE, swap.make_rule)
        drop = balanced(rows, mixed) - balanced(rows, blocked)
        ok = ok and drop < 3
        notes.append(f"{country}: neighbours mixed {balanced(rows, mixed):.0f}%, "
                     f"neighbours held out together {balanced(rows, blocked):.0f}%")
    record("7. Own-country scores hold up when neighbours are held out together (drop under 3 points)",
           ok, "; ".join(notes))


def check_rule_is_stable(table):
    """Does the rule depend on exactly which municipalities it learned from?
    Learn it 20 times from a random half of each country and compare the
    size above which a place is called a city."""
    generator = numpy.random.default_rng(1)
    notes, ok = [], True
    for country in COUNTRIES:
        rows = table[table["country"] == country]
        cuts = []
        for _ in range(20):
            half = rows.sample(frac=0.5, random_state=int(generator.integers(1_000_000)))
            _, city_from = swap.rule_cuts(swap.make_rule().fit(half[swap.RULE], half["label"]))
            cuts.append(city_from / swap.M3_PER_PERSON)
        if numpy.isnan(cuts).any():
            ok = False
            notes.append(f"{country}: in {int(numpy.isnan(cuts).sum())} of 20 halves the rule found no cities at all")
            continue
        low, high = min(cuts), max(cuts)
        ok = ok and high / low < 2
        notes.append(f"{country}: 'city' starts between {low:,.0f} and {high:,.0f} people-equivalents")
    record("8. The rule's city threshold is stable (varies by less than a factor of 2 "
           "over 20 random halves)", ok, "; ".join(notes))


def check_uncertainty(table):
    """With only 18 Hungarian cities, how much could the score move by luck?
    Redraw the test municipalities 500 times and see how far the score swings."""
    generator = numpy.random.default_rng(1)
    notes, ok = [], True
    for learn, test in [("CH", "HU"), ("HU", "CH")]:
        test_rows = table[table["country"] == test].reset_index(drop=True)
        guess = swap.guess_other_country(table[table["country"] == learn], test_rows,
                                         swap.RULE, swap.make_rule)
        scores = []
        for _ in range(500):
            pick = generator.integers(0, len(test_rows), len(test_rows))
            scores.append(balanced(test_rows.iloc[pick], guess[pick]))
        low, high = numpy.percentile(scores, [2.5, 97.5])
        ok = ok and (high - low) < 10
        notes.append(f"{learn}->{test}: {low:.0f}% to {high:.0f}%")
    record("9. Swap scores are pinned down to within 10 points (95% range)", ok, "; ".join(notes))


def check_threshold(table):
    """Building volume stands in for people at 450 m3 per person. That number
    was chosen by hand. Do the scores survive other choices?"""
    rows_out = []
    for exchange in (300, 450, 600, 750):
        prepare.M3_PER_PERSON = exchange
        rows_out.append({"m3 per person": exchange, **swap_scores(recompute_patches(table))})
    prepare.M3_PER_PERSON = 450
    result = pandas.DataFrame(rows_out).set_index("m3 per person")
    spread = result.max() - result.min()
    record("10. Swap scores survive other exchange rates between building volume and people "
           "(no direction moves more than 8 points between 300 and 750 m3 per person)",
           (spread < 8).all(),
           "; ".join(f"{index}: CH->HU {row['CH->HU']:.0f}%, HU->CH {row['HU->CH']:.0f}%"
                     for index, row in result.iterrows()))


# --- Main program -----------------------------------------------------------

def main():
    table = swap.read_municipalities()

    print("\nA. Is the table built correctly?")
    check_labels_against_source(table)
    check_built_up_sums(table)
    check_exact_boundaries(table)
    check_value_ranges(table)
    check_population_totals(table)

    print("\nB. Are the scores honest?")
    check_simple_is_enough(table)
    check_neighbours_do_not_leak(table)
    check_rule_is_stable(table)
    check_uncertainty(table)
    check_threshold(table)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 70)
    print(f"{passed} of {len(RESULTS)} checks passed")
    for name, ok, _ in RESULTS:
        if not ok:
            print(f"  FAILED: {name}")


if __name__ == "__main__":
    main()
