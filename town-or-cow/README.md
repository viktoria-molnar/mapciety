# Town or Cow

Made for Day 9 (Urban-rural) of the #30DayMapChallenge 2026.

**Question:** does "urban" mean the same thing in Switzerland and in Hungary?

## The idea

Two countries of similar population and very different shape. Switzerland is mountainous, with several mid-sized cities and none that dominates. Hungary is mostly plain, with one very large capital and a long step down to the next city.

A model learns to tell urban from rural municipalities in one country, and is then tested on the other country. Then the same thing in reverse. If "urban" looked the same everywhere, both directions would work equally well. Where and by how much each direction fails is the result, and the map.

Three things to show:

- **How urban each country is:** the share of people and of municipalities in cities, in towns and suburbs, and in rural areas.
- **How the urban share is split:** across how many small, medium and big cities.
- **What transfers and what doesn't:** train on Switzerland and test on Hungary, train on Hungary and test on Switzerland, and compare both with a normal test inside one country.

## First step: buildings, density, inhabitants

The first version uses only information about people and buildings per municipality:

| Feature | Meaning |
|---|---|
| Inhabitants | How many people live there |
| Density | Inhabitants per square kilometre of land |
| Built-up share | Part of the municipality's land covered by buildings (footprint) |
| Built-up area per inhabitant | Square metres of building footprint per person |
| Average building height | Building volume divided by footprint, in metres |
| Building volume per inhabitant | Cubic metres of building per person |

Footprint alone treats a ten-storey block like a bungalow. Volume and height bring in the vertical side of "urban".

Averages over a whole municipality are misleading when its boundary includes a lot of empty land: a compact city surrounded by farmland looks like a village. So four more features describe only the densest part of each place, without industry:

| Feature | Meaning |
|---|---|
| `core_built_up_share` | Share of the municipality's densest square kilometre covered by buildings |
| `core_height_m` | Average building height in that square kilometre |
| `dense_share` | Share of the municipality's building volume standing in densely built 1 km squares (at least 10% covered by buildings) |
| `patch_volume_m3` | Size of the connected densely built area the municipality belongs to. It can spread over many municipalities, so a suburb gets the size of the whole city it is part of |

Factories and warehouses are large buildings that nobody lives in, and they make a place look bigger than it feels. So the last three features exist twice: once for all buildings, and once without the buildings on industrial land (columns starting with `town_`). Offices and shops stay in both, because they are part of what makes a place urban. A further feature, `industrial_share`, is the part of the building volume that stands on industrial land.

## Data

| What | Source | Used for |
|---|---|---|
| Municipality boundaries, population and area, 2024 edition | Eurostat GISCO, local administrative units | The map, inhabitants, density |
| List of municipalities with degree of urbanisation, 2024 edition | Eurostat | The official urban/rural label the model learns from |
| Built-up surface and built-up volume 2020, 100 m cells | Global Human Settlement Layer (GHSL), European Commission | Building footprint, volume and height |
| Outlines of industrial areas, current | OpenStreetMap, via the Overpass service | Separating factories and warehouses from the rest |
| Swiss boundary file, 2026-01 edition | swisstopo, swissBOUNDARIES3D | Newer population figures for Switzerland |

One source for both countries, so the same definition of "urban" applies to both. National definitions don't compare: Hungary's legal "town" status and Switzerland's own typology measure different things.

### How current is it?

| Data | Year |
|---|---|
| Boundaries and labels | 2024 edition, the newest with matching boundaries. A 2025 label list exists but is only partially validated and has no boundary file yet. |
| Inhabitants, Hungary | 2024 |
| Inhabitants, Switzerland | In the Eurostat file the total matches the end of 2022. Step 2 replaces it with the figure from the 2026 swisstopo file wherever the municipality is unchanged; merged municipalities keep the older figure. The column `population_source` says which is which. |
| Built-up surface and volume | 2020, the newest epoch measured from satellite images. Later epochs in the same dataset are projections. |

## Plan

| Step | Script | What happens | Machine learning topic |
|---|---|---|---|
| 1 | `01_get_data.py` | Download the datasets and print what is in them | Data acquisition |
| 2 | `02_prepare.py` | Add up built-up area per municipality, compute density, store one table in SQLite | Preparing data, SQL |
| 3 | `03_explore.py` | Urban share per country, city size split, histograms and scatter plots | Visualising data, bias |
| 4 | `04_train_and_swap.py` | Guess the official class from the buildings alone (no population), inside each country and swapped between them. Three versions: a random forest on averages, a random forest on the densest part, and a two-question rule on one number. The rule does as well or better and draws the map. Also compares a too-simple tree, a too-detailed tree and a small neural network | Classification, generalisation, overfitting and underfitting, neural networks, limits of ML |
| check | `check_results.py` | Ten tests that try to break the results: recounts, exact boundaries, neighbours held out together, other thresholds, how far the scores could move by luck | Designing tests |
| 5 | `05_place_types.py` | Group municipalities without using the labels, reduce the features to two dimensions to plot them | Clustering, dimensionality reduction |
| 6 | `06_maps.py` | The final maps | |

A regression (predicting inhabitants from built-up area) fits into step 4 as a second experiment.

## Ideas for later

- **Mountains and plains:** elevation and steepness as features, to explain why the swap fails where it does.
- **Dense but not lively:** shops, everyday restaurants and services from OpenStreetMap, to find places that are dense without feeling like a town.
- **One city or many centres:** clustering to find the centres of the Zurich and Budapest regions.

## Known weak spots

These are part of the project and will be discussed, not hidden:

- **Municipalities are not alike.** Hungarian plain towns often have very large territories with scattered farms, so their density looks low even when the centre is compact. A large alpine municipality can contain one dense village and a lot of empty mountain.
- **The label is itself built from population density.** The model partly re-learns a rule, so a high score inside one country is less impressive than it looks. The swap between countries is the honest test.
- **Built-up area comes from satellite images.** Small scattered buildings are easy to miss, and greenhouses or large farm buildings count as built-up.
- **Industrial areas depend on volunteers.** OpenStreetMap is mapped more thoroughly in some places than others, so a missing industrial area can mean "no industry" or "nobody drew it yet". Farm buildings are not removed at all.
- **Building height is an estimate.** It is modelled from satellite data, not measured building by building. The dataset never goes below about 2.5 m, and most rural municipalities sit exactly on that floor, so height says little about differences between villages.
- **We count people per municipality, not per building.** The data has no number of buildings and no number of flats.
- **The years don't line up exactly.** Labels are from 2024, buildings from 2020, and population from 2024 or later. These things change slowly, but not at the same speed everywhere.
- **"Densely built" is our own threshold, and it matters.** A 1 km square counts as dense at 10% building cover, chosen to roughly match Eurostat's population threshold. Between 5% and 10% the results hold. At 15% and above they collapse for Switzerland, whose towns are less tightly built than Hungary's.
- **Hungary has only 18 cities.** A score for Hungary can move by about 6 points either way depending on which few cities are guessed right.
- **The features were chosen after seeing results.** The switch from averages to the densest part was made because the first version transferred badly, and it was judged on the same two countries. A third country would be the clean test.
- **The boundaries are simplified.** They are drawn for a 1:1 million map, the only version Eurostat publishes. With exact boundaries for Switzerland, 99% of the guesses stay the same.
- **Urban and rural are a scale, not boxes.** Some of the model's "mistakes" will be places where the official label is debatable.

## How to run

You need Python 3.10 or newer. The Python that ships with Xcode on a Mac (3.9) is too old: the OpenStreetMap servers refuse its connections.

The first time, set up the project's own Python environment. With [uv](https://docs.astral.sh/uv/):

```
cd ~/dev/mapciety/town-or-cow
uv venv --python 3.13 --seed
source .venv/bin/activate
uv pip install -r requirements.txt
```

In every new Terminal window, switch it on again:

```
cd ~/dev/mapciety/town-or-cow
source .venv/bin/activate
```

Then run the steps in order:

```
python3 01_get_data.py
python3 02_prepare.py
python3 03_explore.py
python3 04_train_and_swap.py
python3 check_results.py
```

Step 1 downloads into `data/raw/` and skips files that are already there. The OpenStreetMap servers are often busy; the script tries three of them several times, and if it still fails, running it again later usually works. Step 2 writes `data/municipalities.gpkg`. Steps 3 and 4 draw their charts into `charts/`, and step 4 writes its guesses to `data/predictions.gpkg`. The `data/` folder is not uploaded to GitHub.

## Sources

- Eurostat GISCO, local administrative units 2024: https://gisco-services.ec.europa.eu/distribution/v2/lau
- Eurostat, local administrative units and correspondence tables: https://ec.europa.eu/eurostat/web/nuts/local-administrative-units
- swisstopo, swissBOUNDARIES3D: https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissboundaries3d
- OpenStreetMap contributors, data under the Open Database License: https://www.openstreetmap.org/copyright
- GHSL built-up surface and volume (GHS-BUILT-S and GHS-BUILT-V R2023A), European Commission Joint Research Centre, CC BY 4.0: https://human-settlement.emergency.copernicus.eu
