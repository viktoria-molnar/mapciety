"""
Town or Cow - step 1: get the data.

Downloads five things for Switzerland and Hungary into data/raw/ :

  1. lau_2024.gpkg          boundaries of every municipality in Europe, with
                            population and area (Eurostat GISCO, "LAU" = local
                            administrative unit = municipality)
  2. lau_labels_2024.xlsx   Eurostat's list of municipalities, which should
                            carry the official "degree of urbanisation" label
  3. swissboundaries3d_2026.gpkg
                            the current Swiss boundary file, used only for its
                            newer population figures (swisstopo)
  4. built_up_*.tif         how many square metres of each 100 m x 100 m cell
                            are covered by buildings (GHSL, from satellite images)
     built_volume_*.tif     how many cubic metres of building stand on each cell
  5. osm_industrial_*.json  outlines of industrial areas (OpenStreetMap)

Then it prints a short report, so we can see what we actually got before we
build anything on top of it.

Run it from the town-or-cow folder:

    python 01_get_data.py

Files that already exist are not downloaded again. Delete a file in data/raw/
if you want a fresh copy.
"""

import json
import shutil
import sqlite3
import time
import zipfile
from pathlib import Path

import requests

# --- Settings ---------------------------------------------------------------

COUNTRIES = ["CH", "HU"]  # Switzerland, Hungary

RAW_DIR = Path(__file__).parent / "data" / "raw"

# Boundaries and labels are both the 2024 edition, so the municipalities match.
BOUNDARIES_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/lau/gpkg/"
    "LAU_RG_01M_2024_3035.gpkg"
)
LABELS_URL = (
    "https://ec.europa.eu/eurostat/documents/345175/501971/"
    "EU-27-LAU-2024-NUTS-2024.xlsx/12971f56-c035-dbab-4d9f-ff1dcc617bb3"
)

# The built-up maps come in square tiles of 1000 km. One tile per country.
# There are two maps on the same grid:
#   S = surface: square metres of each cell covered by buildings (footprint)
#   V = volume:  cubic metres of building standing on each cell
BUILT_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_BUILT_{kind}_GLOBE_R2023A/GHS_BUILT_{kind}_E2020_GLOBE_R2023A_54009_100/V1-0/tiles/"
    "GHS_BUILT_{kind}_E2020_GLOBE_R2023A_54009_100_V1_0_{tile}.zip"
)
BUILT_TILES = {"CH": "R4_C19", "HU": "R4_C20"}
BUILT_FILES = {"S": "built_up", "V": "built_volume"}  # start of the file names

# The Swiss population figures in the Eurostat file are older than the
# Hungarian ones. The current swisstopo boundary file carries newer ones.
SWISS_URL = (
    "https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d/"
    "swissboundaries3d_2026-01/swissboundaries3d_2026-01_2056_5728.gpkg.zip"
)

# Industrial areas come from OpenStreetMap, through a service called Overpass.
# The query is written in its own small language: "give me every outline
# tagged landuse=industrial inside this rectangle, with its corner points".
# The rectangle is a bit bigger than the country; step 2 only uses what lies
# inside the municipalities, so the extra does no harm.
INDUSTRIAL_QUERY = """
[out:json][timeout:600];
(
  way["landuse"="industrial"]({south},{west},{north},{east});
  relation["landuse"="industrial"]({south},{west},{north},{east});
);
out geom;
"""
COUNTRY_BOXES = {  # south, west, north, east in degrees
    "CH": (45.80, 5.90, 47.85, 10.55),
    "HU": (45.70, 16.05, 48.65, 22.95),
}
# The main server is often too busy, so we try a few in turn, several times.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://ovp.dresel.systems/api/interpreter",
]
OVERPASS_ROUNDS = 3
OVERPASS_PAUSE = 30  # seconds to wait before the next round

# Public services like to know who is calling them.
HEADERS = {"User-Agent": "town-or-cow (30DayMapChallenge learning project)"}


# --- Downloading ------------------------------------------------------------

def download(url, target, form=None, timeout=900):
    """Download url into the file target and return target.

    form is for services that expect a question sent along (like Overpass).
    The download goes to a temporary ".part" file first, so a download that
    breaks half way never looks like a finished file.
    """
    if target.exists():
        print(f"  already there: {target.name}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")

    method = "GET" if form is None else "POST"
    with requests.request(method, url, data=form, headers=HEADERS,
                          stream=True, timeout=timeout) as response:
        if not response.ok:
            # Turn "404 not found", "429 too busy" etc. into an error that
            # includes what the server said about it.
            said = " ".join(response.text[:300].split())
            raise requests.HTTPError(f"{response.status_code} {response.reason}: {said}")
        megabytes = 0
        with open(part, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                file.write(chunk)
                megabytes += len(chunk) / 1024 / 1024
                print(f"  {target.name}: {megabytes:.0f} MB", end="\r")

    part.rename(target)
    print(f"\r  saved {target.name} ({megabytes:.1f} MB)          ")
    return target


def unzip_one(zip_path, ending, target):
    """Copy the one file inside zip_path whose name ends with ending to target."""
    with zipfile.ZipFile(zip_path) as archive:
        inside = [name for name in archive.namelist() if name.lower().endswith(ending)]
        if not inside:
            raise ValueError(f"no {ending} file inside {zip_path.name}: {archive.namelist()}")
        with archive.open(inside[0]) as source, open(target, "wb") as destination:
            shutil.copyfileobj(source, destination)
    print(f"  unpacked {target.name}")
    return target


def get_boundaries():
    return download(BOUNDARIES_URL, RAW_DIR / "lau_2024.gpkg")


def get_labels():
    return download(LABELS_URL, RAW_DIR / "lau_labels_2024.xlsx")


def get_swiss_population():
    gpkg = RAW_DIR / "swissboundaries3d_2026.gpkg"
    if gpkg.exists():
        print(f"  already there: {gpkg.name}")
        return gpkg
    zip_path = download(SWISS_URL, RAW_DIR / "swissboundaries3d_2026.gpkg.zip")
    return unzip_one(zip_path, ".gpkg", gpkg)


def industrial_problem(path):
    """Return what is wrong with a downloaded industrial file, or None if it is fine."""
    try:
        answer = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return "the answer was not the kind of file we asked for"
    if "remark" in answer:
        # A server that runs out of time still answers, with a "remark" and
        # an incomplete list.
        return f"the server gave up ({answer['remark']})"
    if not answer.get("elements"):
        return "the answer was empty"
    return None


def get_industrial(country):
    target = RAW_DIR / f"osm_industrial_{country}.json"
    if target.exists() and industrial_problem(target) is not None:
        print(f"  the existing {target.name} is no good "
              f"({industrial_problem(target)}), downloading it again")
        target.unlink()

    south, west, north, east = COUNTRY_BOXES[country]
    query = INDUSTRIAL_QUERY.format(south=south, west=west, north=north, east=east)

    for round_number in range(1, OVERPASS_ROUNDS + 1):
        for url in OVERPASS_URLS:
            server = url.split("/")[2]
            try:
                download(url, target, form={"data": query})
                problem = industrial_problem(target)
            except requests.RequestException as error:
                problem = " ".join(str(error).split())[:150]
            if problem is None:
                return target
            target.unlink(missing_ok=True)
            print(f"  {server} did not work: {problem}")
        if round_number < OVERPASS_ROUNDS:
            print(f"  waiting {OVERPASS_PAUSE} seconds, then trying again ...")
            time.sleep(OVERPASS_PAUSE)
    raise ValueError("none of the OpenStreetMap servers answered properly")


def get_built(country, kind):
    """kind is "S" for the surface map or "V" for the volume map."""
    name = f"{BUILT_FILES[kind]}_{country}"
    tif = RAW_DIR / f"{name}.tif"
    if tif.exists():
        print(f"  already there: {tif.name}")
        return tif
    url = BUILT_URL.format(kind=kind, tile=BUILT_TILES[country])
    zip_path = download(url, RAW_DIR / f"{name}.zip")
    return unzip_one(zip_path, ".tif", tif)


# --- Looking at what we got -------------------------------------------------

def report_boundaries(path):
    # A GeoPackage is an SQLite database, so we can ask it questions in SQL.
    connection = sqlite3.connect(path)
    tables = [row[0] for row in connection.execute("SELECT table_name FROM gpkg_contents")]
    print(f"\nBoundaries: layers {tables}")

    for table in tables:
        columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
        total = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        print(f"  {table}: {total} rows")
        print(f"    columns: {', '.join(columns)}")

        if "CNTR_CODE" not in columns:
            print("    no CNTR_CODE column, so I can't count by country here")
            continue

        # One line per country: how many municipalities, and for every
        # population column how many are filled in and what they add up to.
        population_columns = [c for c in columns if c.upper().startswith("POP_2")]
        for country in COUNTRIES:
            count = connection.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE CNTR_CODE = ?', (country,)
            ).fetchone()[0]
            print(f"    {country}: {count} municipalities")
            for column in population_columns:
                filled, total_people = connection.execute(
                    f'SELECT COUNT("{column}"), SUM("{column}") FROM "{table}" '
                    f'WHERE CNTR_CODE = ? AND "{column}" > 0', (country,)
                ).fetchone()
                print(f"        {column}: filled in for {filled}, sum {total_people}")

        example = connection.execute(
            f'SELECT * FROM "{table}" WHERE CNTR_CODE = ? LIMIT 1', (COUNTRIES[0],)
        ).fetchone()
        if example:
            shown = [f"{c}={v}" for c, v in zip(columns, example) if not isinstance(v, bytes)]
            print(f"    example row: {', '.join(shown)}")
    connection.close()


def report_labels(path):
    try:
        import openpyxl
    except ImportError:
        print("\nLabels: install openpyxl to look inside the Excel file "
              "(pip install -r requirements.txt)")
        return

    workbook = openpyxl.load_workbook(path, read_only=True)
    print(f"\nLabels: sheets {workbook.sheetnames}")
    for country in COUNTRIES:
        if country not in workbook.sheetnames:
            print(f"  {country}: NO SHEET for this country")
            continue
        rows = list(workbook[country].iter_rows(values_only=True))
        print(f"  {country}: {len(rows)} rows")
        for row in rows[:3]:
            print(f"    {[cell for cell in row if cell is not None]}")
    workbook.close()


def report_swiss_population(path):
    connection = sqlite3.connect(path)
    tables = [row[0] for row in connection.execute("SELECT table_name FROM gpkg_contents")]
    print(f"\nNewer Swiss population: layers {tables}")
    for table in tables:
        if "hoheitsgebiet" not in table.lower():
            continue
        columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
        total = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        print(f"  {table}: {total} rows")
        print(f"    columns: {', '.join(columns)}")
        example = connection.execute(f'SELECT * FROM "{table}" LIMIT 1').fetchone()
        shown = [f"{c}={v}" for c, v in zip(columns, example) if not isinstance(v, bytes)]
        print(f"    example row: {', '.join(shown)}")
    connection.close()


def report_industrial(path):
    answer = json.loads(path.read_text(encoding="utf-8"))
    kinds = [element["type"] for element in answer.get("elements", [])]
    megabytes = path.stat().st_size / 1024 / 1024
    print(f"\nIndustrial areas: {path.name}, {megabytes:.0f} MB, "
          f"{kinds.count('way')} outlines, {kinds.count('relation')} relations")


def report_built(path):
    megabytes = path.stat().st_size / 1024 / 1024
    print(f"\nBuilt-up map: {path.name}, {megabytes:.0f} MB")


# --- Main program -----------------------------------------------------------

def main():
    steps = [("Boundaries and population (Eurostat GISCO)", get_boundaries, report_boundaries),
             ("Urban / rural labels (Eurostat)", get_labels, report_labels),
             ("Newer Swiss population (swisstopo)", get_swiss_population,
              report_swiss_population)]
    for country in COUNTRIES:
        # country=country fixes the value now; without it every step would
        # use the last country in the list.
        steps.append((f"Built-up surface {country} (GHSL)",
                      lambda country=country: get_built(country, "S"), report_built))
        steps.append((f"Built-up volume {country} (GHSL)",
                      lambda country=country: get_built(country, "V"), report_built))
        steps.append((f"Industrial areas {country} (OpenStreetMap)",
                      lambda country=country: get_industrial(country), report_industrial))

    files = []
    problems = []

    print("Downloading into", RAW_DIR)
    for name, get, report in steps:
        print(f"\n{name}")
        # One failed download should not stop the others, so we catch the
        # error, remember it, and carry on.
        try:
            files.append((get(), report))
        except (requests.RequestException, OSError, ValueError) as error:
            print(f"  FAILED: {error}")
            problems.append(name)

    print("\n" + "=" * 70)
    for path, report in files:
        try:
            report(path)
        except Exception as error:  # the report must never hide the other reports
            print(f"\nCould not read {path.name}: {type(error).__name__}: {error}")
            problems.append(path.name)

    print("\n" + "=" * 70)
    if problems:
        print("Finished with problems:", ", ".join(problems))
    else:
        print("All datasets are in place.")


if __name__ == "__main__":
    main()
