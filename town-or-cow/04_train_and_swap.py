"""
Town or Cow - step 4: train, test and swap.

Question: is the official class of a municipality (city / town or suburb /
rural) true to what a satellite sees?

A model learns to guess the official class from the buildings alone, with no
population figures. Three versions are compared:

  1. a random forest (300 decision trees that vote) on averages over the
     whole municipality
  2. a random forest on features of the densest part of the municipality
  3. a simple rule: two yes/no questions about two numbers, the size of the
     connected built-up area the municipality belongs to, counted once at
     town density and once at city density

Version 3 does as well as the others or better, so it is the one used for the
map. Each version is tested in four ways:

    learn in Switzerland, test in Switzerland
    learn in Hungary,     test in Hungary
    learn in Switzerland, test in Hungary       <- the swap
    learn in Hungary,     test in Switzerland   <- the swap, reversed

It writes:
    data/predictions.gpkg                  every municipality with the guesses
    charts/6_swap_scores.png               how good each of the four tests is
    charts/7_map_satellite_vs_official.png the official map next to the guessed one
    charts/8_map_where_they_differ.png     only the places where the two differ
    charts/9_map_overlays.png              both answers on one map, as fill and stripes

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
CORES = ["core_built_up_share", "core_height_m", "dense_share",
         "log_town_area", "log_city_area"]
# Two numbers: the size of the connected built-up area a place belongs to,
# once counting everything of at least town density, once only city density.
RULE = ["log_town_area", "log_city_area"]
# Everything we have, including the features that need a head count.
EVERYTHING = AVERAGES + CORES + ["log_density", "log_footprint_per_person",
                                 "log_volume_per_person", "industrial_share"]
M3_PER_PERSON = 450  # the same exchange rate between building volume and people as in step 2

RANDOM = 1  # fixes the random choices, so every run gives the same result

BACKGROUND = "#fcfcfb"
INK = "#0b0b0b"
NEUTRAL = "#52514e"
GRID = "#e4e3df"
# The official classes: one teal blue, darker = more urban (same as in step 3).
CLASS_COLOURS = {1: "#0b5563", 2: "#4fa9b5", 3: "#cde8eb"}
# The classes as guessed from the satellite data: the same idea in orange-pink.
GUESS_COLOURS = {1: "#c2472f", 2: "#f59a7c", 3: "#fde4d8"}
# The classes according to the rule learned in the other country: purple,
# used only as stripes on top of the orange-pink.
OTHER_COLOURS = {1: "#3d1457", 2: "#8a4fb0", 3: "#ffffff"}
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
    # + 1 because many places belong to no built-up area at all, and log of 0 does not exist
    table["log_town_area"] = numpy.log10(table["town_area_m3"] + 1)
    table["log_city_area"] = numpy.log10(table["city_area_m3"] + 1)

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


class GentleTree(DecisionTreeClassifier):
    """A decision tree that pays some extra attention to rare classes.

    Hungary has 18 cities among 3,155 municipalities. A tree that only tries
    to be right as often as possible ignores them. A tree told that all
    classes are equally important ("balanced") finds them, but then calls far
    too many places a city. This one sits in between: a class that is 100
    times rarer counts 10 times more (the square root).
    """

    def fit(self, X, y, **arguments):
        counts = pandas.Series(y).value_counts()
        self.class_weight = {label: float((len(y) / count) ** 0.5)
                             for label, count in counts.items()}
        return super().fit(X, y, **arguments)


def make_rule():
    """The simplest model that works here: a decision tree that may ask two
    yes/no questions in a row. A decision tree is a chain of questions like
    "is this number above 24?", learned from the examples."""
    return GentleTree(max_depth=2, random_state=RANDOM)


def rule_cuts(model):
    """Read a fitted rule: from which built-up area size on does it say
    "town or suburb", and from which size "city"? Both in m3 of building."""
    steps = numpy.linspace(0, 10, 2001)  # sizes from 1 m3 to 10 billion m3
    # town: grow the town-density area while there is no city-density area
    town = model.predict(pandas.DataFrame({"log_town_area": steps, "log_city_area": 0.0}))
    # city: grow the city-density area inside a huge town-density area
    city = model.predict(pandas.DataFrame({"log_town_area": 10.0, "log_city_area": steps}))
    town_from = 10 ** steps[town != 3].min() if (town != 3).any() else numpy.nan
    city_from = 10 ** steps[city == 1].min() if (city == 1).any() else numpy.nan
    return town_from, city_from


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
    for country, country_name in COUNTRIES.items():
        rows = table[table["country"] == country]
        town_from, city_from = rule_cuts(make_rule().fit(rows[RULE], rows["label"]))
        print(f"  {country_name}:")
        print(f"    town or suburb if it is part of a built-up area of town density holding at least "
              f"{town_from / 1e6:,.1f} million m3 of building (room for about "
              f"{town_from / M3_PER_PERSON:,.0f} people)")
        print(f"    city if it is part of a built-up area of city density holding at least "
              f"{city_from / 1e6:,.1f} million m3 (about {city_from / M3_PER_PERSON:,.0f} people)")
    print("  Eurostat's own rule: 5,000 people for a town, 50,000 for a city.")


def compare_models(table):
    """Too simple, too detailed, and in between: how models overfit and underfit.

    "learned" is the score on the municipalities the model learned from,
    "new" the score on municipalities it has not seen. A big gap between the
    two means the model memorised instead of learning (overfitting). Two low
    scores mean it is too simple to capture the pattern (underfitting).
    """
    models = {
        "simple rule, 2 questions": (make_rule(), RULE),
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


def draw_maps(table, rows_of_maps, file_name, subtitle, legends, height):
    """A grid of maps: one row per entry in rows_of_maps, one column per country.

    Each entry is (column to show, colours, text for the title, a function
    that turns one country's rows into a second title line).
    """
    figure, axes = plt.subplots(len(rows_of_maps), 2, figsize=(13, height))
    top = 1 - 1.8 / height
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.9 / height, top=top,
                           wspace=0.02, hspace=0.3)

    # Same scale for both countries: every map shows a window of the same
    # width and height in kilometres, centred on its country.
    shapes = {country: table[table["country"] == country].to_crs(MAP_PROJECTIONS[country])
              for country in COUNTRIES}
    width = max(s.total_bounds[2] - s.total_bounds[0] for s in shapes.values()) * 1.04
    tall = max(s.total_bounds[3] - s.total_bounds[1] for s in shapes.values()) * 1.04

    for row_number, (row_of_axes, (column, colours, what, detail)) in enumerate(zip(axes, rows_of_maps)):
        for axis, (country, country_name) in zip(row_of_axes, COUNTRIES.items()):
            rows = shapes[country]
            rows.plot(ax=axis, color=rows[column].map(colours),
                      edgecolor=BACKGROUND, linewidth=0.1)
            left, bottom, right, top_edge = rows.total_bounds
            axis.set_xlim((left + right - width) / 2, (left + right + width) / 2)
            axis.set_ylim((bottom + top_edge - tall) / 2, (bottom + top_edge + tall) / 2)
            if row_number == 0:  # a 100 km bar, to show the scale
                x0, y0 = axis.get_xlim()[0] + 0.03 * width, axis.get_ylim()[0] + 0.04 * tall
                axis.plot([x0, x0 + 100_000], [y0, y0], color=NEUTRAL, linewidth=2,
                          solid_capstyle="butt")
                axis.text(x0 + 50_000, y0 + 0.02 * tall, "100 km", ha="center", va="bottom",
                          fontsize=8, color=NEUTRAL)
            # The two countries have different shapes, so their maps end up
            # at different heights. Writing the title at the top of the map's
            # slot (not of the map itself) keeps the titles of a row level.
            slot = axis.get_subplotspec().get_position(figure)
            # the title sits above three lines of room for the details
            figure.text(slot.x0 + 0.01, slot.y1 + 0.60 / height, f"{country_name}: {what}",
                        fontsize=11, fontweight="bold", ha="left", va="bottom")
            figure.text(slot.x0 + 0.01, slot.y1 + 0.55 / height, detail(rows),
                        fontsize=9, color=NEUTRAL, ha="left", va="top")
            axis.set_anchor("N")
            axis.set_axis_off()

    for position, (title, colours, names) in enumerate(legends):
        handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for colour in colours.values()]
        figure.legend(handles, names, loc="lower center", ncol=len(names), frameon=False,
                      title=title, bbox_to_anchor=((position + 0.5) / len(legends), 0))
    figure.suptitle("Is the official class true to the satellite view?", x=0.01,
                    y=1 - 0.12 / height, ha="left", fontsize=13, fontweight="bold")
    figure.text(0.01, 1 - 0.42 / height, subtitle, color=NEUTRAL, fontsize=9, va="top")
    figure.savefig(CHARTS_DIR / file_name, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"  saved charts/{file_name}")


def class_counts(rows, column):
    """For a title: how many municipalities are in each class."""
    counts = rows[column].value_counts()
    return ", ".join(f"{counts.get(number, 0):,} {name}" for number, name in CLASSES.items())


def match(rows, first, second):
    """Share of municipalities, and of people, where two columns agree."""
    same = rows[first] == rows[second]
    return (100 * same.mean(),
            100 * rows.loc[same, "population"].sum() / rows["population"].sum())


def chart_maps(table):
    subtitle = ("Teal: Eurostat degree of urbanisation 2024, based on where people live.\n"
                "Orange-pink: guessed from satellite data alone, from the size of the connected "
                "built-up area a place belongs to (building volume as a stand-in for people).")

    def from_same_country(rows):
        places, people = match(rows, "guess_own", "label")
        return (f"{class_counts(rows, 'guess_own')}\n"
                f"Matches Eurostat for {places:.0f}% of municipalities, home to {people:.0f}% of the people")

    def from_other_country(rows):
        places, people = match(rows, "guess_other", "label")
        with_own, _ = match(rows, "guess_other", "guess_own")
        return (f"{class_counts(rows, 'guess_other')}\n"
                f"Matches Eurostat for {places:.0f}% of municipalities, home to {people:.0f}% of the people\n"
                f"Matches the rule from the same country for {with_own:.0f}% of municipalities")

    # 7: the official map and the satellite maps side by side, same style
    draw_maps(table, [
        ("label", CLASS_COLOURS, "Eurostat's official class",
         lambda rows: class_counts(rows, "label")),
        ("guess_own", GUESS_COLOURS, "from satellite, rule learned in the same country",
         from_same_country),
        ("guess_other", GUESS_COLOURS, "from satellite, rule learned in the other country",
         from_other_country),
    ], "7_map_satellite_vs_official.png", subtitle,
        [("Official class", CLASS_COLOURS, CLASSES.values()),
         ("Class guessed from satellite data", GUESS_COLOURS, CLASSES.values())], height=13.6)

    # 8: only the differences
    def share_same(column):
        return lambda rows: f"{100 * (rows[column] == 0).mean():.0f}% of municipalities same as the label"

    draw_maps(table, [
        ("difference_own", DIFFERENCE_COLOURS, "rule learned in the same country",
         share_same("difference_own")),
        ("difference_other", DIFFERENCE_COLOURS, "rule learned in the other country",
         share_same("difference_other")),
    ], "8_map_where_they_differ.png",
        "Where the class guessed from satellite data differs from the official class.",
        [("From the buildings alone, the place looks ...", DIFFERENCE_COLOURS,
          DIFFERENCE_NAMES.values())], height=9.6)


def chart_overlays(table, extra=None, file_name="9_map_overlays.png"):
    """Two comparisons, each as one map per country.

    The filled colour is the first source. Where the second source says
    something else, its class is drawn on top as stripes, so both answers
    stay visible. No stripes = the two agree.

    extra adds a third row below: a dictionary with "column" (a share between
    0 and 1), "colours" (a matplotlib colour map), "title", and "detail" (a
    function that turns one country's rows into text). Step 6 uses it for
    census data.
    """
    comparisons = [
        ("label", CLASS_COLOURS, "Eurostat",
         "guess_own", GUESS_COLOURS, "satellite, rule from the same country"),
        ("guess_own", GUESS_COLOURS, "satellite, rule from the same country",
         "guess_other", OTHER_COLOURS, "satellite, rule from the other country"),
    ]
    height = 10.6 if extra is None else 15.2
    figure, axes = plt.subplots(2 if extra is None else 3, 2, figsize=(13, height))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=(0.95 if extra is None else 1.25) / height,
                           top=1 - 2.05 / height, wspace=0.02, hspace=0.42)
    plt.rcParams["hatch.linewidth"] = 1.1

    shapes = {country: table[table["country"] == country].to_crs(MAP_PROJECTIONS[country])
              for country in COUNTRIES}
    width = max(s.total_bounds[2] - s.total_bounds[0] for s in shapes.values()) * 1.04
    tall = max(s.total_bounds[3] - s.total_bounds[1] for s in shapes.values()) * 1.04

    for row_number, (row_of_axes, (base, base_colours, base_name, top, top_colours, top_name)) \
            in enumerate(zip(axes, comparisons)):
        for axis, (country, country_name) in zip(row_of_axes, COUNTRIES.items()):
            rows = shapes[country]
            rows.plot(ax=axis, color=rows[base].map(base_colours),
                      edgecolor=BACKGROUND, linewidth=0.1)
            # stripes only where the second source disagrees, one class at a time
            different = rows[rows[base] != rows[top]]
            for number, colour in top_colours.items():
                part = different[different[top] == number]
                if len(part):
                    part.plot(ax=axis, facecolor="none", edgecolor=colour, hatch="//////",
                              linewidth=0)
            left, bottom, right, top_edge = rows.total_bounds
            axis.set_xlim((left + right - width) / 2, (left + right + width) / 2)
            axis.set_ylim((bottom + top_edge - tall) / 2, (bottom + top_edge + tall) / 2)
            if row_number == 0:  # a 100 km bar, to show the scale
                x0, y0 = axis.get_xlim()[0] + 0.03 * width, axis.get_ylim()[0] + 0.04 * tall
                axis.plot([x0, x0 + 100_000], [y0, y0], color=NEUTRAL, linewidth=2,
                          solid_capstyle="butt")
                axis.text(x0 + 50_000, y0 + 0.02 * tall, "100 km", ha="center", va="bottom",
                          fontsize=8, color=NEUTRAL)

            places, people = match(rows, base, top)
            more = (different[top] < different[base]).sum()
            slot = axis.get_subplotspec().get_position(figure)
            figure.text(slot.x0 + 0.01, slot.y1 + 0.70 / height,
                        f"{country_name}: {places:.0f}% of municipalities get the same class",
                        fontsize=11, fontweight="bold", ha="left", va="bottom")
            figure.text(slot.x0 + 0.01, slot.y1 + 0.65 / height,
                        f"Filled: {base_name}.\nStriped: {top_name}, where it differs.\n"
                        f"{len(different):,} places differ: {more:,} more urban in the striped view, "
                        f"{len(different) - more:,} less urban.\nSame class for {people:.0f}% of the people.",
                        fontsize=9, color=NEUTRAL, ha="left", va="top")
            axis.set_anchor("N")
            axis.set_axis_off()

    if extra is not None:
        for axis, (country, country_name) in zip(axes[2], COUNTRIES.items()):
            rows = shapes[country]
            known = rows[rows[extra["column"]].notna()]
            known.plot(ax=axis, column=extra["column"], cmap=extra["colours"], vmin=0, vmax=1,
                       edgecolor=BACKGROUND, linewidth=0.1)
            missing = rows[rows[extra["column"]].isna()]
            if len(missing):
                missing.plot(ax=axis, color="#e4e3df", edgecolor=BACKGROUND, linewidth=0.1)
            left, bottom, right, top_edge = rows.total_bounds
            axis.set_xlim((left + right - width) / 2, (left + right + width) / 2)
            axis.set_ylim((bottom + top_edge - tall) / 2, (bottom + top_edge + tall) / 2)
            slot = axis.get_subplotspec().get_position(figure)
            figure.text(slot.x0 + 0.01, slot.y1 + 0.62 / height, f"{country_name}: {extra['title']}",
                        fontsize=11, fontweight="bold", ha="left", va="bottom")
            figure.text(slot.x0 + 0.01, slot.y1 + 0.57 / height, extra["detail"](rows),
                        fontsize=9, color=NEUTRAL, ha="left", va="top")
            axis.set_anchor("N")
            axis.set_axis_off()
        # a colour bar for the census share, under the two census maps
        bar = figure.add_axes([0.32, 0.86 / height, 0.36, 0.12 / height])
        scale = plt.cm.ScalarMappable(cmap=extra["colours"], norm=plt.Normalize(0, 100))
        figure.colorbar(scale, cax=bar, orientation="horizontal", format="%.0f%%")
        bar.set_title(extra["legend"], fontsize=9)
        bar.tick_params(labelsize=8)

    def filled(colours):
        return [plt.Rectangle((0, 0), 1, 1, color=colour) for colour in colours.values()]

    def striped(colours):
        # on a grey patch, so that light stripes show up in the legend too
        return [plt.Rectangle((0, 0), 1, 1, facecolor="#cfcec8", edgecolor=colour,
                              hatch="//////", linewidth=0) for colour in colours.values()]

    legends = [("Eurostat, filled", filled(CLASS_COLOURS)),
               ("Satellite, same-country rule, filled", filled(GUESS_COLOURS)),
               ("Satellite, same-country rule, striped", striped(GUESS_COLOURS)),
               ("Satellite, other-country rule, striped", striped(OTHER_COLOURS))]
    for position, (title, handles) in enumerate(legends):
        figure.legend(handles, CLASSES.values(), loc="lower center", ncol=3, frameon=False,
                      title=title, bbox_to_anchor=((position + 0.5) / len(legends), 0),
                      handleheight=1.4, columnspacing=1.0, handletextpad=0.5, fontsize=9)
    figure.suptitle("Where do the classifications differ?", x=0.01, y=1 - 0.12 / height,
                    ha="left", fontsize=13, fontweight="bold")
    figure.text(0.01, 1 - 0.42 / height,
                "Top: Eurostat's official class against the class guessed from satellite data. "
                "Bottom: the satellite rule learned at home against the one learned in the other country.\n"
                "Plain colour means both agree. Stripes show what the second one says instead.",
                color=NEUTRAL, fontsize=9, va="top")
    figure.savefig(CHARTS_DIR / file_name, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"  saved charts/{file_name}")


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

    print("\n=== How often the maps agree (share of municipalities / share of people) ===")
    for country, country_name in COUNTRIES.items():
        rows = table[table["country"] == country]
        for text, first, second in [
                ("Eurostat vs rule from the same country ", "label", "guess_own"),
                ("Eurostat vs rule from the other country", "label", "guess_other"),
                ("the two rules with each other          ", "guess_own", "guess_other")]:
            places, people = match(rows, first, second)
            print(f"  {country_name:<12} {text}  {places:3.0f}% / {people:3.0f}%")

    compare_models(table)
    print_importance(table)
    print_biggest_surprises(table)

    print()
    keep = ["id", "country", "name", "population", "label", "label_name", "guess_own",
            "guess_other", "difference_own", "difference_other", "geometry"]
    table[keep].to_file(DATA_DIR / "predictions.gpkg", layer="predictions", driver="GPKG")
    print("  saved data/predictions.gpkg")
    chart_scores(scores_a + scores_b + scores_c)
    chart_maps(table)
    chart_overlays(table)
    print("\nOpen the charts with:  open charts")


if __name__ == "__main__":
    main()
