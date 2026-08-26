# Data

Two small tables, shipped so that the worked examples have a real acquisition
geometry rather than a synthetic one. Clustered, non-uniform sampling is the
case an inference method has to survive, and uniformly scattered points quietly
make every problem easier than it is.

- `gsn_stations.csv` — Global Seismograph Network station coordinates, from
  IRIS. Name, latitude, longitude in degrees.
- `usgs_event_cache.csv` — a cached USGS earthquake catalogue. The columns are
  the USGS query format; only `latitude`, `longitude` and `mag` are read.

Loaded by `pygeoinf2.symmetric_space.sphere`. Nothing is fetched at runtime.
