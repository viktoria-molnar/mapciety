"""
Town or Cow - step 5: census data on homes.

Do houses (and owning them) explain why the satellite sees Hungary as more
urban than Eurostat does? This step fetches, per municipality:

  house_share  share of homes that are (probably) in a single-family house
  owner_share  share of households that own their home

and saves both to data/census.csv.

The sources are not alike, so read the "how" column before comparing:

  country      measure      year  how
  Switzerland  house_share  2025  counted: dwellings in single-family houses
                                  (register of buildings and dwellings)
  Hungary      house_share  2022  estimated: dwellings NOT heated from a
                                  building-wide boiler or district heating.
                                  Old blocks of flats where every flat has its
                                  own boiler count as houses, so this is too
                                  high in old city centres, Budapest above all.
  Switzerland  owner_share  2000  estimated from the 2000 census: occupied
                                  dwellings minus rented ones. The newest
                                  ownership figures per municipality.
  Hungary      owner_share  2022  counted: households living in a home they own

Run it from the town-or-cow folder:

    python3 05_get_census.py
"""

import csv
import importlib.util
import io
import json
from pathlib import Path

import pandas
import requests

HERE = Path(__file__).parent
CENSUS_DIR = HERE / "data" / "raw" / "census"
OUTPUT = HERE / "data" / "census.csv"

# --- Where the data comes from ---------------------------------------------

# Switzerland: dwellings per municipality by building category, newest year.
# "1021" is single-family house, "_T" is all buildings.
CH_DWELLINGS_URL = ("https://disseminate.stats.swiss/rest/data/CH1.GWS,DF_GWS_REG5,/"
                    "A._T.1021+_T._T.?lastNObservations=1&dimensionAtObservation=AllDimensions")
# Switzerland, 2000 census: dwellings by occupancy, and rented dwellings by rent
CH_OCCUPANCY_URL = ("https://www.pxweb.bfs.admin.ch/api/v1/de/px-x-0902020100_136/"
                    "px-x-0902020100_136.px")
CH_RENT_URL = ("https://www.pxweb.bfs.admin.ch/api/v1/de/px-x-0902020100_137/"
               "px-x-0902020100_137.px")
# Switzerland: which municipality became which, after mergers
CH_COMMUNES_URL = "https://www.agvchapp.bfs.admin.ch/api/communes/correspondances"

# Hungary, 2022 census, one row per settlement and indicator
HU_API = "https://nepszamlalas2022.ksh.hu/api/dataflows/{table}/V67/d/TIME_PERIOD:2022,TERUL_GEO5,{dimension}:{codes}"
HU_HEATING = HU_API.format(table="WBL003", dimension="INDICATORS2",
                           codes="FUTES_TOH_TOTAL+FUTES_TOH_HEAT112+FUTES_TOH_HEAT12")
HU_TENURE = HU_API.format(table="WBH002", dimension="TEL_H_ADAT", codes="HJC_TSH+OWN")

# Our municipalities are the 2024 ones (Eurostat). Budapest is one there, and
# its code in the census is "HU11".
REFERENCE_DATE = "01-01-2024"
BUDAPEST = {"HU11": "13578"}


# --- Downloading ------------------------------------------------------------

def load_step(file_name):
    spec = importlib.util.spec_from_file_location(file_name[:-3], HERE / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEADERS = {"User-Agent": "town-or-cow (30DayMapChallenge learning project)"}


def fetch(name, url, accept=None, query=None):
    """Download once into data/raw/census/name and return the text."""
    path = CENSUS_DIR / name
    if path.exists():
        print(f"  already there: {name}")
        return path.read_text(encoding="utf-8")
    CENSUS_DIR.mkdir(parents=True, exist_ok=True)
    headers = dict(HEADERS, **({"Accept": accept} if accept else {}))
    if query is None:
        response = requests.get(url, headers=headers, timeout=300)
    else:  # PxWeb tables are asked with a small JSON question
        response = requests.post(url, json=query, headers=headers, timeout=300)
    if not response.ok:
        said = " ".join(response.text[:200].split())
        raise requests.HTTPError(f"{response.status_code} {response.reason}: {said}")
    # PxWeb answers in an old Windows encoding, everything else in UTF-8
    text = response.content.decode("cp1252" if query is not None else "utf-8")
    path.write_text(text, encoding="utf-8")
    print(f"  saved {name} ({len(text) / 1024:,.0f} KB)")
    return text


def pxweb_all(url, variables):
    """A PxWeb question: every value of the given variables, others summed up."""
    return {"query": [{"code": code, "selection": {"filter": "all", "values": ["*"]}}
                      for code in variables],
            "response": {"format": "csv"}}


# --- Switzerland ------------------------------------------------------------

def commune_mapping(start, end=REFERENCE_DATE):
    """Municipality number on start -> municipality number on end."""
    text = fetch(f"ch_communes_{start[-4:]}_{end[-4:]}.csv",
                 f"{CH_COMMUNES_URL}?startPeriod={start}&endPeriod={end}&includeUnmodified=true")
    mapping = {}
    for row in csv.DictReader(io.StringIO(text)):
        # A municipality that was split goes wholly to its first successor.
        mapping.setdefault(int(row["InitialCode"]), int(row["TerminalCode"]))
    return mapping


def swiss_house_share():
    text = fetch("ch_dwellings.csv", CH_DWELLINGS_URL,
                 accept="application/vnd.sdmx.data+csv;version=1.0.0")
    rows = pandas.read_csv(io.StringIO(text), dtype=str)
    rows = rows[rows["GEMEINDENAME"].str.fullmatch(r"\d+")]  # municipalities only
    year = rows["TIME_PERIOD"].max()
    counts = rows.pivot_table(index="GEMEINDENAME", columns="GKATS", values="OBS_VALUE",
                              aggfunc=lambda v: float(v.iloc[0]))
    share = (counts["1021"] / counts["_T"]).rename("house_share")
    share.index = share.index.astype(int)

    # The data uses this year's municipalities, ours are from 2024. Where
    # municipalities merged since, each 2024 part gets the merged share.
    forward = commune_mapping(REFERENCE_DATE, f"01-01-{year}")
    result = {old: share.get(new) for old, new in forward.items()}
    print(f"  Switzerland, {year}: house share for "
          f"{sum(value is not None and value == value for value in result.values())} municipalities")
    return pandas.Series(result, name="house_share", dtype=float), int(year)


def read_pxweb_csv(text):
    """PxWeb CSV: first column "0261 Zürich", then one column per value."""
    table = pandas.read_csv(io.StringIO(text))
    first = table.columns[0]
    table = table[table[first].str.match(r"\d{4} ")]
    table.index = table[first].str[:4].astype(int)
    return table.drop(columns=first).apply(pandas.to_numeric, errors="coerce").fillna(0)


def swiss_owner_share():
    occupancy = read_pxweb_csv(fetch("ch_occupancy_2000.csv", CH_OCCUPANCY_URL,
                                     query=pxweb_all(CH_OCCUPANCY_URL, ["Gemeinde", "Belegungsart"])))
    rents = read_pxweb_csv(fetch("ch_rents_2000.csv", CH_RENT_URL,
                                 query=pxweb_all(CH_RENT_URL, ["Gemeinde", "Mietpreisklasse"])))
    lived_in = occupancy["Dauernd bewohnt"]
    unknown = rents["Angabe fehlt"]
    rented = rents.drop(columns=["Keine bewohnte Mietwohnung", "Angabe fehlt"]).sum(axis=1)

    # Add up the 2000 municipalities into the 2024 ones they became part of.
    mapping = commune_mapping("01-01-2000")
    counts = pandas.DataFrame({"lived_in": lived_in, "unknown": unknown, "rented": rented})
    counts["now"] = [mapping.get(number) for number in counts.index]
    counts = counts.dropna(subset=["now"]).groupby("now").sum()
    known = counts["lived_in"] - counts["unknown"]
    share = ((known - counts["rented"]) / known).clip(0, 1).where(known > 0)
    share.index = share.index.astype(int)
    print(f"  Switzerland, 2000: owner share for {share.notna().sum()} municipalities")
    return share.rename("owner_share")


# --- Hungary ----------------------------------------------------------------

def read_hungarian(text, dimension):
    """One row per settlement, one column per indicator. Settlement codes are
    the same 5-digit codes Eurostat uses. Values held back for privacy
    (status "Q", always small numbers) count as 0."""
    rows = pandas.DataFrame(json.loads(text))
    rows["value"] = pandas.to_numeric(rows["OBS_VALUE"], errors="coerce").fillna(0)
    rows["place"] = rows["TERUL_GEO5"].replace(BUDAPEST)
    rows = rows[rows["place"].str.fullmatch(r"\d{5}")]
    held_back = (rows["OBS_STATUS"] == "Q").sum()
    return rows.pivot_table(index="place", columns=dimension, values="value", aggfunc="sum"), held_back


def hungarian_shares():
    heating, held_back_1 = read_hungarian(fetch("hu_heating_2022.json", HU_HEATING), "INDICATORS2")
    tenure, held_back_2 = read_hungarian(fetch("hu_tenure_2022.json", HU_TENURE), "TEL_H_ADAT")
    shared = heating["FUTES_TOH_HEAT112"] + heating["FUTES_TOH_HEAT12"]
    house = (1 - shared / heating["FUTES_TOH_TOTAL"]).where(heating["FUTES_TOH_TOTAL"] > 0)
    owner = (tenure["OWN"] / tenure["HJC_TSH"]).where(tenure["HJC_TSH"] > 0)
    print(f"  Hungary, 2022: {held_back_1 + held_back_2} small values held back for privacy, "
          f"counted as 0")
    return house.rename("house_share"), owner.rename("owner_share")


# --- Main program -----------------------------------------------------------

def main():
    print("Switzerland")
    ch_house, ch_year = swiss_house_share()
    ch_owner = swiss_owner_share()
    swiss = pandas.concat([ch_house, ch_owner], axis=1)
    swiss.index = [f"CH_CH{number:04d}" for number in swiss.index]

    print("Hungary")
    hu_house, hu_owner = hungarian_shares()
    hungary = pandas.concat([hu_house, hu_owner], axis=1)
    hungary.index = [f"HU_{code}" for code in hungary.index]

    census = pandas.concat([swiss, hungary])
    census["house_year"] = [ch_year if i.startswith("CH") else 2022 for i in census.index]
    census["owner_year"] = [2000 if i.startswith("CH") else 2022 for i in census.index]
    census.index.name = "id"

    # Keep only our municipalities, and say how many have data.
    ours = load_step("04_train_and_swap.py").read_municipalities()[["id", "country", "name", "label_name"]]
    census = ours.merge(census, left_on="id", right_index=True, how="left")
    census.drop(columns=["name", "label_name"]).to_csv(OUTPUT, index=False)
    print(f"\nSaved data/{OUTPUT.name}")

    print("\nMunicipalities with data, and the typical value (median) per class")
    for country in ("CH", "HU"):
        rows = census[census["country"] == country]
        print(f"  {country}: house share for {rows['house_share'].notna().sum()} of {len(rows)}, "
              f"owner share for {rows['owner_share'].notna().sum()}")
        medians = rows.groupby("label_name")[["house_share", "owner_share"]].median()
        print("    " + (100 * medians).round(0).to_string().replace("\n", "\n    "))
    for name in ("Zürich", "Genève", "Budapest", "Debrecen", "Békéscsaba", "Újkígyós"):
        row = census[census["name"] == name]
        if len(row):
            row = row.iloc[0]
            print(f"  {name:<11} houses {100 * row['house_share']:4.0f}%   "
                  f"owners {100 * row['owner_share']:4.0f}%")


if __name__ == "__main__":
    main()
