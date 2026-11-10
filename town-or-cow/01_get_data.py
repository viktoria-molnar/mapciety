"""
Town or Cow - step 1: get the data.

Downloads four things for Switzerland and Hungary into data/raw/ :

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

Then it prints a short report, so we can see what we actually got before we
build anything on top of it.

Run it from the town-or-cow folder:

    python 01_get_data.py

Files that already exist are not downloaded again. Delete a file in data/raw/
if you want a fresh copy.
"""

import shutil
import sqlite3
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

# The built-up map comes in square tiles of 1000 km. One tile per country.
BUILT_UP_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100/V1-0/tiles/"
    "GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0_{tile}.zip"
)
BUILT_UP_TILES = {"CH": "R4_C19", "HU": "R4_C20"}

# The Swiss population figures in the Eurostat file are older than the
# Hungarian ones. The current swisstopo boundary file carries newer ones.
SWISS_URL = (
    "https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d/"
    "swissboundaries3d_2026-01/swissboundaries3d_2026-01_2056_5728.gpkg.zip"
)

# Public services like to know who is calling them.
HEADERS = {"User-Agent": "town-or-cow (30DayMapChallenge learning project)"}


# --- Downloading ------------------------------------------------------------

def download(url, target, timeout=600):
    """Download url into the file target and return target.

    The download goes to a temporary ".part" file first, so a download that
    breaks half way never looks like a finished file.
    """
    if target.exists():
        print(f"  already there: {target.name}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")

    with requests.get(url, headers=HEADERS, stream=True, timeout=timeout) as response:
        response.raise_for_status()  # turns "404 not found" etc. into an error
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


def get_built_up(country):
    tif = RAW_DIR / f"built_up_{country}.tif"
    if tif.exists():
        print(f"  already there: {tif.name}")
        return tif
    url = BUILT_UP_URL.format(tile=BUILT_UP_TILES[country])
    zip_path = download(url, RAW_DIR / f"built_up_{country}.zip")
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


def report_built_up(path):
    megabytes = path.stat().st_size / 1024 / 1024
    print(f"\nBuilt-up area: {path.name}, {megabytes:.0f} MB")


# --- Main program -----------------------------------------------------------

def main():
    steps = [("Boundaries and population (Eurostat GISCO)", get_boundaries, report_boundaries),
             ("Urban / rural labels (Eurostat)", get_labels, report_labels),
             ("Newer Swiss population (swisstopo)", get_swiss_population,
              report_swiss_population)]
    for country in COUNTRIES:
        # country=country fixes the value now; without it both steps would
        # use the last country in the list.
        steps.append((f"Built-up area {country} (GHSL)",
                      lambda country=country: get_built_up(country), report_built_up))

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
