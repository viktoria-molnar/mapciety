"""
Town or Cow - step 2: prepare the data.

Turns the raw downloads into one clean table with one row per municipality:

    country, name, label (city / town or suburb / rural), city region,
    population, area, density, built-up area and volume, built-up share,
    built-up area and volume per inhabitant, average building height,
    and the same again without buildings on industrial land

and saves it as data/municipalities.gpkg. That file holds the table and the
boundaries together, and because a GeoPackage is an SQLite database, every
later step can read it with plain SQL.

Run it from the town-or-cow folder, after 01_get_data.py:

    python3 02_prepare.py
"""

import json
import sqlite3
from pathlib import Path

import geopandas
import numpy
import pandas
import rasterio
import rasterio.features
import rasterio.windows
import shapely
import shapely.ops
from shapely.geometry import LineString, Polygon

# --- Settings ---------------------------------------------------------------

COUNTRIES = ["CH", "HU"]
LABEL_NAMES = {1: "city", 2: "town or suburb", 3: "rural"}

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT = DATA_DIR / "municipalities.gpkg"

# A Swiss municipality whose area differs by more than this between the two
# sources has probably merged with a neighbour since 2024.
AREA_TOLERANCE = 0.03


# --- Reading ----------------------------------------------------------------

def read_boundaries():
    """Boundaries, population and area for our countries (Eurostat GISCO)."""
    wanted = ", ".join(f"'{country}'" for country in COUNTRIES)
    table = geopandas.read_file(RAW_DIR / "lau_2024.gpkg",
                                where=f"CNTR_CODE IN ({wanted})")
    table = table.rename(columns={
        "GISCO_ID": "id", "CNTR_CODE": "country", "LAU_NAME": "name",
        "POP_2024": "population", "AREA_KM2": "area_km2",
    })
    return table[["id", "country", "name", "population", "area_km2", "geometry"]]


def read_labels():
    """Eurostat's official class for every municipality, and its city region."""
    sheets = pandas.read_excel(RAW_DIR / "lau_labels_2024.xlsx", sheet_name=COUNTRIES)
    parts = []
    for country in COUNTRIES:
        sheet = sheets[country].dropna(subset=["EU LAU CODE"])
        parts.append(pandas.DataFrame({
            "id": sheet["EU LAU CODE"].astype(str),
            "label": sheet["DEGURBA"].astype(int),
            "city_region": sheet["FUA_NAME"].replace("", None),
            "official_area_km2": sheet["TOTAL AREA (m2)"] / 1_000_000,
        }))
    return pandas.concat(parts, ignore_index=True)


def read_newer_swiss_population():
    """Population per Swiss municipality from the swisstopo 2026 boundary file.

    Returns a table with id, new_population and new_area_km2, or None if the
    file is missing or does not look the way we expect.
    """
    path = RAW_DIR / "swissboundaries3d_2026.gpkg"
    if not path.exists():
        print("  swissboundaries3d_2026.gpkg not found - run 01_get_data.py again")
        return None

    connection = sqlite3.connect(path)
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT table_name FROM gpkg_contents WHERE lower(table_name) LIKE '%hoheitsgebiet%'")]
        if not tables:
            print("  no municipality layer found in the swisstopo file")
            return None
        # A municipality with separate pieces has one row per piece, each
        # repeating the totals, so we take one value per municipality number.
        # icc is the country: the file also covers Liechtenstein.
        # gem_flaeche is the area in hectares including lakes, see_flaeche the
        # lake part. Eurostat's area is without lakes, so we subtract it.
        rows = connection.execute(f"""
            SELECT bfs_nummer, MAX(einwohnerzahl),
                   MAX(gem_flaeche) - COALESCE(MAX(see_flaeche), 0)
            FROM "{tables[0]}"
            WHERE icc = 'CH' AND einwohnerzahl > 0
            GROUP BY bfs_nummer
        """).fetchall()
    except sqlite3.Error as error:
        print(f"  could not read the swisstopo file: {error}")
        return None
    finally:
        connection.close()

    return pandas.DataFrame({
        "id": [f"CH_CH{int(number):04d}" for number, _, _ in rows],
        "new_population": [people for _, people, _ in rows],
        "new_area_km2": [hectares / 100 for _, _, hectares in rows],
    })


def read_industrial(country):
    """Industrial areas from OpenStreetMap as a list of shapes.

    The download is a list of "elements". Most are simple outlines ("way"):
    a list of corner points. Some are "relations": several outline pieces
    that together form one area, sometimes with holes in it.
    """
    path = RAW_DIR / f"osm_industrial_{country}.json"
    if not path.exists():
        print(f"  {path.name} not found - run 01_get_data.py again")
        return None
    elements = json.loads(path.read_text(encoding="utf-8"))["elements"]

    def to_line(points):
        return LineString([(point["lon"], point["lat"]) for point in points])

    def to_area(lines):
        # Glue outline pieces together and turn every closed ring into an area.
        if not lines:
            return Polygon()
        return shapely.union_all(list(shapely.ops.polygonize(shapely.union_all(lines))))

    shapes = []
    for element in elements:
        if element["type"] == "way":
            points = element.get("geometry", [])
            if len(points) >= 4 and points[0] == points[-1]:
                shapes.append(Polygon(to_line(points).coords))
        elif element["type"] == "relation":
            pieces = [member for member in element.get("members", [])
                      if member["type"] == "way" and len(member.get("geometry") or []) >= 2]
            outer = [to_line(m["geometry"]) for m in pieces if m.get("role") != "inner"]
            inner = [to_line(m["geometry"]) for m in pieces if m.get("role") == "inner"]
            shapes.append(to_area(outer).difference(to_area(inner)))

    shapes = geopandas.GeoSeries(shapes, crs="EPSG:4326").make_valid()
    return shapes[~shapes.is_empty]


# --- Combining --------------------------------------------------------------

def update_swiss_population(table):
    """Replace the Swiss population figures with the newer swisstopo ones.

    Only where the municipality is clearly still the same place: same number
    and (almost) the same area. Merged municipalities keep the older figure.
    """
    table["population_source"] = "Eurostat LAU 2024"
    newer = read_newer_swiss_population()
    if newer is None:
        print("  keeping the Eurostat population for Switzerland")
        return table

    table = table.merge(newer, on="id", how="left")
    area_change = (table["new_area_km2"] / table["official_area_km2"] - 1).abs()
    same_place = table["new_population"].notna() & (area_change <= AREA_TOLERANCE)

    swiss = table["country"] == "CH"
    print(f"  Swiss municipalities: {swiss.sum()}, "
          f"updated: {same_place.sum()}, kept older figure: {(swiss & ~same_place).sum()}")
    print(f"  Swiss population before: {table.loc[swiss, 'population'].sum():,.0f}")

    # Safety net: a country's population does not jump by 10% in two years.
    # If the new total is that far off, we misread the file, so we keep the
    # old figures.
    before = table.loc[swiss, "population"].sum()
    after = table.loc[swiss, "population"].where(~same_place, table["new_population"]).sum()
    print(f"  Swiss population after:  {after:,.0f}")
    if abs(after / before - 1) > 0.10:
        print("  WARNING: that change is implausible - keeping the Eurostat population")
    else:
        table.loc[same_place, "population"] = table.loc[same_place, "new_population"]
        table.loc[same_place, "population_source"] = "swisstopo 2026-01"
    return table.drop(columns=["new_population", "new_area_km2"])


def read_grid(path, window_bounds):
    """Read the part of a satellite map inside window_bounds.

    Returns the values, and the "transform" that says where each cell is.
    """
    with rasterio.open(path) as raster:
        left, bottom, right, top = window_bounds
        if (left < raster.bounds.left or right > raster.bounds.right
                or bottom < raster.bounds.bottom or top > raster.bounds.top):
            print(f"  WARNING: the area reaches beyond {path.name}")
        window = rasterio.windows.from_bounds(left, bottom, right, top, raster.transform)
        window = window.round_offsets().round_lengths()
        values = raster.read(1, window=window).astype("float64")
        values[values == raster.nodata] = 0
        return values, raster.window_transform(window), raster.crs


def sums_for_country(table, country, industrial):
    """Add up building footprint and volume inside each municipality.

    The satellite maps are grids of 100 m cells. We paint every municipality
    onto the same grid with its row number, then add up the values of all
    cells that carry the same number. Cells on industrial land are added up
    a second time separately.
    """
    shapes = table[table["country"] == country]
    with rasterio.open(RAW_DIR / f"built_up_{country}.tif") as raster:
        shapes = shapes.to_crs(raster.crs)  # same map projection as the grid
    bounds = shapes.total_bounds

    surface, transform, crs = read_grid(RAW_DIR / f"built_up_{country}.tif", bounds)
    volume, _, _ = read_grid(RAW_DIR / f"built_volume_{country}.tif", bounds)

    numbered = [(shape, number) for number, shape in enumerate(shapes.geometry, start=1)]
    zones = rasterio.features.rasterize(numbered, out_shape=surface.shape,
                                        transform=transform, fill=0, dtype="int32")

    if industrial is None or len(industrial) == 0:
        on_industrial_land = numpy.zeros(surface.shape, dtype=bool)
    else:
        painted = rasterio.features.rasterize(list(industrial.to_crs(crs)),
                                              out_shape=surface.shape, transform=transform,
                                              fill=0, default_value=1, dtype="uint8")
        on_industrial_land = painted == 1

    def per_municipality(values):
        sums = numpy.bincount(zones.ravel(), weights=values.ravel(), minlength=len(shapes) + 1)
        return sums[1:]  # sums[0] is everything outside the country

    return pandas.DataFrame({
        "built_up_km2": per_municipality(surface) / 1_000_000,
        "built_volume_m3": per_municipality(volume),
        "industrial_built_up_km2": per_municipality(surface * on_industrial_land) / 1_000_000,
        "industrial_built_volume_m3": per_municipality(volume * on_industrial_land),
    }, index=shapes.index)


def add_features(table):
    """Compute the numbers the model will learn from."""
    table["density"] = table["population"] / table["area_km2"]
    table["built_up_share"] = table["built_up_km2"] / table["area_km2"]
    table["industrial_share"] = table["industrial_built_volume_m3"] / table["built_volume_m3"]

    # Where there are no inhabitants or no buildings, "per person" and
    # "height" make no sense, so those cells stay empty.
    people = table["population"].where(table["population"] > 0)

    # Everything: homes, offices, shops, factories, barns.
    footprint_m2 = (table["built_up_km2"] * 1_000_000).where(table["built_up_km2"] > 0)
    table["built_up_m2_per_person"] = footprint_m2 / people
    table["built_volume_m3_per_person"] = table["built_volume_m3"] / people
    # Volume = footprint x height, so volume / footprint is the average height.
    table["average_height_m"] = table["built_volume_m3"] / footprint_m2

    # The same three numbers without buildings on industrial land. Offices and
    # shops stay in: they are part of what makes a place urban.
    footprint_m2 = (table["built_up_km2"] - table["industrial_built_up_km2"]) * 1_000_000
    footprint_m2 = footprint_m2.where(footprint_m2 > 0)
    volume_m3 = table["built_volume_m3"] - table["industrial_built_volume_m3"]
    table["town_built_up_m2_per_person"] = footprint_m2 / people
    table["town_built_volume_m3_per_person"] = volume_m3 / people
    table["town_average_height_m"] = volume_m3 / footprint_m2

    table["label_name"] = table["label"].map(LABEL_NAMES)
    return table


# --- Checking the result ----------------------------------------------------

def report(path):
    connection = sqlite3.connect(path)

    print("\nPeople and municipalities per class")
    query = """
        SELECT country, label_name,
               COUNT(*)                   AS municipalities,
               SUM(population)            AS people,
               ROUND(SUM(built_up_km2))   AS built_up_km2
        FROM municipalities
        GROUP BY country, label
        ORDER BY country, label
    """
    rows = connection.execute(query).fetchall()
    totals = {}
    for country, _, _, people, _ in rows:
        totals[country] = totals.get(country, 0) + people
    for country, label_name, municipalities, people, built_up in rows:
        share = 100 * people / totals[country]
        print(f"  {country}  {label_name:<15} {municipalities:>5} municipalities "
              f"{people:>10,.0f} people ({share:4.1f}%)  {built_up:>5,.0f} km2 built-up")

    print("\nTypical values (median) per class")
    table = pandas.read_sql(
        "SELECT country, label, label_name, density, industrial_share, "
        "built_volume_m3_per_person, town_built_volume_m3_per_person, "
        "average_height_m, town_average_height_m FROM municipalities", connection)
    medians = table.groupby(["country", "label", "label_name"]).median().round(3)
    print(medians.to_string())

    print("\nMost industrial (share of building volume on industrial land)")
    for row in connection.execute("""
            SELECT country, name, population, ROUND(100 * industrial_share) || '%'
            FROM municipalities WHERE population > 5000
            ORDER BY industrial_share DESC LIMIT 8"""):
        print("  ", row)

    print("\nTallest on average")
    for row in connection.execute("""
            SELECT country, name, population, ROUND(average_height_m, 1)
            FROM municipalities
            ORDER BY average_height_m DESC LIMIT 8"""):
        print("  ", row)

    empty = connection.execute(
        "SELECT COUNT(*) FROM municipalities WHERE built_up_km2 = 0").fetchone()[0]
    print(f"\nMunicipalities where the satellite map shows no buildings at all: {empty}")
    connection.close()


# --- Main program -----------------------------------------------------------

def main():
    print("Reading boundaries ...")
    boundaries = read_boundaries()
    print(f"  {len(boundaries)} areas")

    print("Reading labels (the Excel file is big, this takes a moment) ...")
    labels = read_labels()
    print(f"  {len(labels)} labelled municipalities")

    # Keep only areas that have a label. This drops lakes and shared forests,
    # which are in the boundary file but are not municipalities.
    table = boundaries.merge(labels, on="id", how="inner")
    print(f"Joined: {len(table)} municipalities "
          f"({len(boundaries) - len(table)} areas without a label dropped)")

    print("Newer population for Switzerland ...")
    table = update_swiss_population(table)

    print("Adding up built-up surface and volume ...")
    parts = []
    for country in COUNTRIES:
        industrial = read_industrial(country)
        sums = sums_for_country(table, country, industrial)
        parts.append(sums)
        total = sums.sum()
        print(f"  {country}: {total['built_up_km2']:,.0f} km2 footprint, "
              f"{total['built_volume_m3'] / 1_000_000:,.0f} million m3 volume, "
              f"{100 * total['industrial_built_volume_m3'] / total['built_volume_m3']:.0f}% "
              f"of the volume on industrial land")
    table = table.join(pandas.concat(parts))

    table = add_features(table)
    table = table.drop(columns=["official_area_km2"])

    table.to_file(OUTPUT, layer="municipalities", driver="GPKG")
    print(f"\nSaved {len(table)} municipalities to {OUTPUT}")
    report(OUTPUT)


if __name__ == "__main__":
    main()
