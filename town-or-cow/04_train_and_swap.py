"""
Town or Cow - step 4: train, test and swap.

Question: is the official class of a municipality (city / town or suburb /
rural) true to what a satellite sees?

A model learns to guess the official class from the buildings alone, with no
population figures. Three versions are compared:

  1. a random forest (300 decision trees that vote) on averages over the
     whole municipality
  2. a random forest on features of the densest part of the municipality
  3. a simple rule: two yes/no questions about one number, the size of the
     connected densely built area the municipality belongs to

Version 3 does as well as the others or better, so it is the one used for the
map. Each version is tested in four ways:

    learn in Switzerland, test in Switzerland
    learn in Hungary,     test in Hungary
    learn in Switzerland, test in Hungary       <- the swap
    learn in Hungary,     test in Switzerland   <- the swap, reversed

It writes:
    data/predictions.gpkg                  every municipality with the guesses
    charts/6_swap_scores.png               how good each of the four tests is
    charts/7_map_satellite_vs_official.png where the guess and the label differ

Run it from the town-or-cow folder, after 02_prepare.py:

    python3 04_train_and_swap.py
"""

from pathlib import Path

import geopandas
import matplotlib
matplotlib.use("Agg")  # draw straight into files, without opening windows
import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --- Settings ---------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
CHARTS_DIR = Path(__file__).parent / "charts"

COUNTRIES = {"CH": "Switzerland", "HU": "Hungary"}
# For drawing, each country gets its own national map projection, so that
# north points straight up. In the shared European projection the data is
# stored in, Hungary comes out tilted.
MAP_PROJECTIONS = {"CH": "EPSG:2056", "HU": "EPSG:23700"}
CLASSES = {1: "city", 2: "town or suburb", 3: "rural"}

# What the satellite sees, as averages over the whole municipality.
# No population in here.
AVERAGES = ["log_built_up_share", "average_height_m", "log_volume_per_km2"]
# What the satellite sees in the densest part of the municipality, without
# industry. Not affected by how much empty land a municipality includes.
CORES = ["core_built_up_share", "core_height_m", "dense_share", "log_patch_volume"]
# One number: the size of the connected densely built area a place belongs to.
RULE = ["log_patch_volume"]
# Everything we have, including the features that need a head count.
EVERYTHING = AVERAGES + CORES + ["log_density", "log_footprint_per_person",
                                 "log_volume_per_person", "industrial_share"]

RANDOM = 1  # fixes the random choices, so every run gives the same result

BACKGROUND = "#fcfcfb"
INK = "#0b0b0b"
NEUTRAL = "#52514e"
GRID = "#e4e3df"
# How the guess compares with the official class. Negative = the buildings
# look more urban than the label says.
DIFFERENCE_COLOURS = {-2: "#c2362f", -1: "#f0a195", 0: "#dddcd7", 1: "#9cc1ee", 2: "#2a78d6"}
DIFFERENCE_NAMES = {-2: "much more urban", -1: "more urban", 0: "same as the label",
                    1: "less urban", 2: "much less urban"}

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


# --- Data -------------------------------------------------------------------

def read_municipalities():
    table = geopandas.read_file(DATA_DIR / "municipalities.gpkg", layer="municipalities")

    # Values like density range from 5 to 10,000. Taking the logarithm turns
    # "ten times more" into "one step more", which models handle better.
    table["log_built_up_share"] = numpy.log10(table["built_up_share"])
    table["log_volume_per_km2"] = numpy.log10(table["built_volume_m3"] / table["area_km2"])
    table["log_density"] = numpy.log10(table["density"])
    table["log_footprint_per_person"] = numpy.log10(table["town_built_up_m2_per_person"])
    table["log_volume_per_person"] = numpy.log10(table["town_built_volume_m3_per_person"])
    # + 1 because many places belong to no dense patch at all, and log of 0 does not exist
    table["log_patch_volume"] = numpy.log10(table["patch_volume_m3"] + 1)

    # A model cannot work with empty cells or infinite numbers.
    usable = numpy.isfinite(table[EVERYTHING]).all(axis=1)
    print(f"{len(table)} municipalities, {(~usable).sum()} left out for missing values")
    return table[usable].reset_index(drop=True)


# --- Models -----------------------------------------------------------------

def make_forest():
    """A random forest: many decision trees that vote.

    class_weight="balanced" makes rare classes count as much as common ones.
    Without it the model could ignore Hungary's 18 cities and still look good.
    """
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                  class_weight="balanced", random_state=RANDOM, n_jobs=-1)


def make_rule():
    """The simplest model that works here: a decision tree that may ask two
    yes/no questions in a row. A decision tree is a chain of questions like
    "is this number above 24?", learned from the examples."""
    return DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=RANDOM)


def guess_own_country(rows, features, make_model=make_forest):
    """Guess every municipality with a model that has not seen it.

    The country is cut into 5 parts. Each part is guessed by a model that
    learned from the other 4. This is called cross-validation.
    """
    parts = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)
    return cross_val_predict(make_model(), rows[features], rows["label"], cv=parts)


def guess_other_country(learn_rows, test_rows, features, make_model=make_forest):
    """Learn from one country, guess the other."""
    model = make_model()
    model.fit(learn_rows[features], learn_rows["label"])
    return model.predict(test_rows[features])


def score(rows, guesses):
    """Two ways of saying how good the guesses are, in percent.

    accuracy: share of municipalities guessed right. Misleading when one class
              dominates: "always say rural" is right 89% of the time in Hungary.
    balanced: the average of "share guessed right" per class, so each class
              counts the same however many members it has.
    """
    return (100 * accuracy_score(rows["label"], guesses),
            100 * balanced_accuracy_score(rows["label"], guesses))


def print_confusion(rows, guesses):
    """Rows: the official class. Columns: what the model guessed."""
    official = rows["label"].map(CLASSES).rename("official")
    guessed = pandas.Series(guesses, index=rows.index).map(CLASSES).rename("guessed")
    counts = pandas.crosstab(official, guessed).reindex(
        index=list(CLASSES.values()), columns=list(CLASSES.values()), fill_value=0)
    print("    " + counts.to_string().replace("\n", "\n    "))


# --- The experiments --------------------------------------------------------

def run_swap(table, features, feature_set_name, make_model=make_forest):
    """The four tests for one set of features and one kind of model.
    Returns scores and guesses."""
    print(f"\n=== {feature_set_name} ===")
    scores = []
    guesses = {}
    for test_country in COUNTRIES:
        test_rows = table[table["country"] == test_country]
        for learn_country in COUNTRIES:
            if learn_country == test_country:
                guess = guess_own_country(test_rows, features, make_model)
            else:
                guess = guess_other_country(table[table["country"] == learn_country],
                                            test_rows, features, make_model)
            accuracy, balanced = score(test_rows, guess)
            print(f"\n  learn in {COUNTRIES[learn_country]}, test in {COUNTRIES[test_country]}: "
                  f"{accuracy:.0f}% right, balanced {balanced:.0f}%")
            print_confusion(test_rows, guess)
            scores.append({"features": feature_set_name, "learn": learn_country,
                           "test": test_country, "accuracy": accuracy, "balanced": balanced})
            guesses[(learn_country, test_country)] = pandas.Series(guess, index=test_rows.index)
    return scores, guesses


def print_rule(table):
    """Say in words what the simple rule learned in each country."""
    print("\n=== The simple rule each country teaches ===")
    print("  (size of the connected densely built area a place belongs to,")
    print("   in million m3 of building)")
    sizes = numpy.linspace(0, 10, 2001)  # from 1 m3 to 10 billion m3, in small steps
    for country, country_name in COUNTRIES.items():
        rows = table[table["country"] == country]
        model = make_rule().fit(rows[RULE], rows["label"])
        guesses = model.predict(pandas.DataFrame({RULE[0]: sizes}))
        parts = []
        start = 0
        for position in range(1, len(sizes) + 1):
            if position == len(sizes) or guesses[position] != guesses[start]:
                low = "0" if start == 0 else f"{10 ** sizes[start] / 1_000_000:,.1f}"
                high = "" if position == len(sizes) else f"{10 ** sizes[position] / 1_000_000:,.1f}"
                span = f"above {low}" if not high else f"{low} to {high}"
                parts.append(f"{span}: {CLASSES[guesses[start]]}")
                start = position
        print(f"  {country_name}: " + "; ".join(parts))


def compare_models(table):
    """Too simple, too detailed, and in between: how models overfit and underfit.

    "learned" is the score on the municipalities the model learned from,
    "new" the score on municipalities it has not seen. A big gap between the
    two means the model memorised instead of learning (overfitting). Two low
    scores mean it is too simple to capture the pattern (underfitting).
    """
    models = {
        "simple rule, 1 feature": (make_rule(), RULE),
        "tree with 1 question (too simple)": (DecisionTreeClassifier(
            max_depth=1, class_weight="balanced", random_state=RANDOM), CORES),
        "tree with 4 levels of questions": (DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=RANDOM), CORES),
        "tree without limit (too detailed)": (DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM), CORES),
        "random forest (300 trees voting)": (make_forest(), CORES),
        "small neural network": (make_pipeline(
            StandardScaler(),  # neural networks need all features on a similar scale
            MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=RANDOM)), CORES),
    }
    print("\n=== Six models compared inside each country (balanced score, %) ===")
    parts = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)
    for country, country_name in COUNTRIES.items():
        rows = table[table["country"] == country]
        print(f"\n  {country_name:<36} learned    new")
        for name, (model, features) in models.items():
            result = cross_validate(model, rows[features], rows["label"], cv=parts,
                                    scoring="balanced_accuracy", return_train_score=True)
            print(f"    {name:<34} {100 * result['train_score'].mean():5.0f}  "
                  f"{100 * result['test_score'].mean():5.0f}")


def print_importance(table):
    """Which features the forest leans on when it may use everything."""
    print("\n=== Which features matter (all features, share of the forest's decisions) ===")
    for country, country_name in COUNTRIES.items():
        rows = table[table["country"] == country]
        model = make_forest().fit(rows[EVERYTHING], rows["label"])
        ranked = sorted(zip(model.feature_importances_, EVERYTHING), reverse=True)
        print(f"  {country_name}: " + ", ".join(f"{name} {100 * share:.0f}%" for share, name in ranked))


def print_biggest_surprises(table):
    """The largest places where the buildings and the label disagree."""
    print("\n=== Largest places where the buildings disagree with the label ===")
    for wanted, text in [(-1, "look MORE urban than their label"),
                         (1, "look LESS urban than their label")]:
        for country, country_name in COUNTRIES.items():
            rows = table[(table["country"] == country)
                         & (numpy.sign(table["difference_own"]) == wanted)]
            rows = rows.sort_values("population", ascending=False).head(6)
            places = [f"{row['name']} ({row['label_name']} -> {CLASSES[row['guess_own']]})"
                      for _, row in rows.iterrows()]
            print(f"  {country_name}, {text}:\n    " + "; ".join(places))


# --- Charts -----------------------------------------------------------------

def chart_scores(scores):
    """Balanced score of the four tests, for each set of features."""
    scores = pandas.DataFrame(scores)
    feature_sets = list(scores["features"].unique())
    figure, axes = plt.subplots(1, len(feature_sets), figsize=(5 * len(feature_sets), 3.8),
                                sharex=True, sharey=True)

    for axis, feature_set in zip(axes, feature_sets):
        rows = scores[scores["features"] == feature_set].reset_index(drop=True)
        names = [f"learn in {COUNTRIES[r['learn']]}\ntest in {COUNTRIES[r['test']]}"
                 for _, r in rows.iterrows()]
        # the two swaps in a stronger tone than the two home tests
        tones = [INK if r["learn"] != r["test"] else "#9a9993" for _, r in rows.iterrows()]
        axis.barh(range(len(rows)), rows["balanced"], height=0.6, color=tones)
        for position, value in enumerate(rows["balanced"]):
            axis.text(value + 1, position, f"{value:.0f}%", va="center", fontsize=9)
        axis.set_yticks(range(len(rows)), names)
        axis.set_ylim(len(rows) - 0.5, -0.95)  # first test on top, room above for the note
        axis.set_xlim(0, 100)
        axis.axvline(100 / 3, color=NEUTRAL, linewidth=1, linestyle=":")
        axis.text(100 / 3 + 1, -0.62, "guessing blindly", color=NEUTRAL, fontsize=8)
        axis.set_title(feature_set)
        axis.xaxis.set_major_formatter("{x:.0f}%")
        axis.grid(axis="y", visible=False)

    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.suptitle("How well the official class can be guessed (balanced score)",
                    x=0.01, y=0.99, ha="left", fontsize=13, fontweight="bold")
    figure.text(0.01, 0.89, "Dark bars: the model learned in one country and was tested in the other.",
                color=NEUTRAL, fontsize=9)
    CHARTS_DIR.mkdir(exist_ok=True)
    figure.savefig(CHARTS_DIR / "6_swap_scores.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    print("  saved charts/6_swap_scores.png")


def chart_map(table):
    """Where the buildings look more or less urban than the official class."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 9.4))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.07, top=0.885, wspace=0.02, hspace=0.12)
    panels = [("difference_own", "model learned in the same country"),
              ("difference_other", "model learned in the OTHER country")]

    for row_of_axes, (column, how) in zip(axes, panels):
        for axis, (country, country_name) in zip(row_of_axes, COUNTRIES.items()):
            rows = table[table["country"] == country].to_crs(MAP_PROJECTIONS[country])
            rows.plot(ax=axis, color=rows[column].map(DIFFERENCE_COLOURS),
                      edgecolor=BACKGROUND, linewidth=0.1)
            agree = 100 * (rows[column] == 0).mean()
            axis.set_title(f"{country_name}, {how}: {agree:.0f}% same as the label")
            axis.set_axis_off()

    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for colour in DIFFERENCE_COLOURS.values()]
    figure.legend(handles, DIFFERENCE_NAMES.values(), loc="lower center", ncol=5, frameon=False,
                  title="From the buildings alone, the place looks ...")
    figure.suptitle("Is the official class true to the satellite view?", x=0.01, y=0.985,
                    ha="left", fontsize=13, fontweight="bold")
    figure.text(0.01, 0.945, "Official class: Eurostat degree of urbanisation 2024. "
                "The model sees one number: the size of the connected densely built area a place belongs to.",
                color=NEUTRAL, fontsize=9)
    figure.savefig(CHARTS_DIR / "7_map_satellite_vs_official.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    print("  saved charts/7_map_satellite_vs_official.png")


# --- Main program -----------------------------------------------------------

def main():
    table = read_municipalities()

    scores_a, _ = run_swap(table, AVERAGES, "Averages, random forest")
    scores_b, _ = run_swap(table, CORES, "Densest part, random forest")
    scores_c, guesses = run_swap(table, RULE, "Built-up area size, simple rule", make_rule)
    print_rule(table)

    # Store the simple rule's guesses next to the official class.
    table["guess_own"] = pandas.concat([guesses[(c, c)] for c in COUNTRIES])
    table["guess_other"] = pandas.concat(
        [guesses[(learn, test)] for learn in COUNTRIES for test in COUNTRIES if learn != test])
    # 1 = city, 3 = rural, so a guess below the label means "more urban".
    table["difference_own"] = table["guess_own"] - table["label"]
    table["difference_other"] = table["guess_other"] - table["label"]

    compare_models(table)
    print_importance(table)
    print_biggest_surprises(table)

    print()
    keep = ["id", "country", "name", "population", "label", "label_name", "guess_own",
            "guess_other", "difference_own", "difference_other", "geometry"]
    table[keep].to_file(DATA_DIR / "predictions.gpkg", layer="predictions", driver="GPKG")
    print("  saved data/predictions.gpkg")
    chart_scores(scores_a + scores_b + scores_c)
    chart_map(table)
    print("\nOpen the charts with:  open charts")


if __name__ == "__main__":
    main()
