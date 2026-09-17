# Data

Two small tables, shipped so that the worked examples have a real acquisition
geometry rather than a synthetic one. Clustered, non-uniform sampling is the
case an inference method has to survive, and uniformly scattered points quietly
make every problem easier than it is.

- `gsn_stations.csv` — Global Seismograph Network station coordinates, from
  IRIS. Name, latitude, longitude in degrees.
- `usgs_event_cache.csv` — a cached USGS earthquake catalogue. The columns are
  the USGS query format; only `latitude`, `longitude` and `mag` are read.

Read through `pygeoinf2.datasets.read_table`, which takes a copy in the
user's cache directory before the bundled one. The loaders never fetch;
`pygeoinf2.datasets.download_gsn_stations` and `download_usgs_earthquakes`
refresh the cache when asked.
