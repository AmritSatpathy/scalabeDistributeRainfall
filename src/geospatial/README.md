# Geospatial scripts

All Task 1 map-generation code is kept in this directory. Run scripts from the
project root with the virtual-environment Python.

| Script | Purpose |
|---|---|
| `task1_geospatial.py` | Coverage-safe H3, seasonal, anomaly and storm-event figures |
| `task1_geospatial_choropleths.py` | Focused H3 comparisons and layered inspection map |
| `task1_august_geospatial_maps.py` | 2017/2024 rainfall and active-gauge maps |
| `task1_august_year_maps.py` | Comparable August maps for selected years |
| `task1_long_period_average_map.py` | 2017–2024 long-period H3 monthly average |
| `task1_region_monthly_choropleth.py` | Official-region 2017–2024 monthly average |
| `task1_region_2024_choropleths.py` | Official-region August and full-year 2024 maps |

Published PNG maps are written to `output/figures/geospatial`. Interactive HTML
maps remain in `output/interactive`, GeoJSON remains in `output/geojson`, and
supporting values remain in `output/tables`.
