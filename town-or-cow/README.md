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

The first version uses only three kinds of information per municipality:

| Feature | Meaning |
|---|---|
| Inhabitants | How many people live there |
| Density | Inhabitants per square kilometre |
| Built-up area | Square metres covered by buildings, also as a share of the municipality and per inhabitant |

## Data

| What | Source | Used for |
|---|---|---|
| Municipality boundaries, population and area, 2024 edition | Eurostat GISCO, local administrative units | The map, inhabitants, density |
| List of municipalities with degree of urbanisation, 2024 edition | Eurostat | The official urban/rural label the model learns from |
| Built-up surface 2020, 100 m cells | Global Human Settlement Layer (GHSL), European Commission | Built-up area |
| Swiss boundary file, 2026-01 edition | swisstopo, swissBOUNDARIES3D | Newer population figures for Switzerland |

One source for both countries, so the same definition of "urban" applies to both. National definitions don't compare: Hungary's legal "town" status and Switzerland's own typology measure different things.

### How current is it?

| Data | Year |
|---|---|
| Boundaries and labels | 2024 edition, the newest with matching boundaries. A 2025 label list exists but is only partially validated and has no boundary file yet. |
| Inhabitants, Hungary | 2024 |
| Inhabitants, Switzerland | In the Eurostat file the total matches the end of 2022. Step 2 replaces it with the figure from the 2026 swisstopo file wherever the municipality is unchanged; merged municipalities keep the older figure. The column `population_source` says which is which. |
| Built-up area | 2020, the newest epoch measured from satellite images. Later epochs in the same dataset are projections. |

## Plan

| Step | Script | What happens | Machine learning topic |
|---|---|---|---|
| 1 | `01_get_data.py` | Download the datasets and print what is in them | Data acquisition |
| 2 | `02_prepare.py` | Add up built-up area per municipality, compute density, store one table in SQLite | Preparing data, SQL |
| 3 | `03_explore.py` | Urban share per country, city size split, histograms and scatter plots | Visualising data, bias |
| 4 | `04_train_and_swap.py` | Train in one country, test in the other, both directions; compare a decision tree, a random forest and a small neural network | Classification, generalisation, overfitting and underfitting, neural networks, limits of ML |
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
- **The years don't line up exactly.** Labels are from 2024, buildings from 2020, and population from 2024 or later. These things change slowly, but not at the same speed everywhere.
- **Urban and rural are a scale, not boxes.** Some of the model's "mistakes" will be places where the official label is debatable.

## How to run

The first time, set up the project's own Python environment:

```
cd ~/dev/mapciety/town-or-cow
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
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
```

Step 1 downloads into `data/raw/` and skips files that are already there. Step 2 writes `data/municipalities.gpkg`. The `data/` folder is not uploaded to GitHub.

## Sources

- Eurostat GISCO, local administrative units 2024: https://gisco-services.ec.europa.eu/distribution/v2/lau
- Eurostat, local administrative units and correspondence tables: https://ec.europa.eu/eurostat/web/nuts/local-administrative-units
- swisstopo, swissBOUNDARIES3D: https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissboundaries3d
- GHSL built-up surface (GHS-BUILT-S R2023A), European Commission Joint Research Centre, CC BY 4.0: https://human-settlement.emergency.copernicus.eu
