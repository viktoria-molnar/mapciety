# Sound Shadow

A vertical sound map of Elephant & Castle, London: how loud is road traffic
on each floor of each building, and how much quieter is it behind a tall one?

## The idea

Official noise maps are calculated at one height, 4 m above ground. This
project asks what happens above that, in a place where 40-storey towers stand
next to two-storey terraces.

1. **Learn shielding from official data.** Defra's 2022 road noise map gives
   a value every 10 m at 4 m height. A model learns to predict it from
   geometry only: how far away the roads are, how big they are, and how much
   of them is hidden behind buildings.
2. **Test on streets the model never saw.**
3. **Go up.** Recompute the same geometry for every floor and predict again.

## What is and isn't checked

- Predictions at 4 m are checked against official values.
- Predictions on higher floors are **extrapolated**. No open London dataset
  has noise per floor, so nothing confirms them.

## Data

| Data | Source |
|---|---|
| Road noise at 4 m, 10 m grid, 2022 mapping (Lden, Lnight, 8 pitch bands) | Defra, Open Government Licence |
| Surface and ground height, 1 m grid, from laser scans | Environment Agency, Open Government Licence |
| Building outlines, roads, railways | OpenStreetMap contributors, ODbL |

## Steps

| File | Does |
|---|---|
| `01_get_data.py` | Downloads everything for a 1.6 km square and prints a report |
