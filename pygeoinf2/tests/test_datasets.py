"""The shipped tables, the cache, and the explicit downloads, off the network."""

from __future__ import annotations

import numpy as np
import pytest

from pygeoinf2 import datasets

IRIS_TEXT = """#Network | Station | Latitude | Longitude | Elevation | SiteName | StartTime | EndTime
IU|AAK|42.6375|74.4942|1633.1|Ala Archa, Kyrgyzstan|1990-01-01|
II|ABKT|37.9304|58.1189|678.0|Alibek, Turkmenistan|1993-01-01|
"""

USGS_TEXT = """time,latitude,longitude,depth,mag,magType
2026-04-05T02:18:02.073Z,28.4548,142.6478,10,5.1,mb
2026-04-04T22:31:20.621Z,-31.5203,-177.9766,30.861,5,mb
2026-04-03T00:00:00.000Z,10.0,20.0,5,6.2,mw
"""


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PYGEOINF_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


@pytest.fixture
def stub(monkeypatch):
    """Replace the one network seam with a recorder."""
    calls = []

    def fetch(url, params, /, *, timeout):
        calls.append((url, dict(params), timeout))
        return IRIS_TEXT if "iris" in url else USGS_TEXT

    monkeypatch.setattr(datasets, "_fetch", fetch)
    return calls


class TestTheCacheDirectory:
    def test_the_variable_wins(self, cache):
        assert datasets.cache_directory() == cache

    def test_the_platform_default_otherwise(self, monkeypatch, tmp_path):
        monkeypatch.delenv("PYGEOINF_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        if datasets.os.name != "nt":
            assert datasets.cache_directory() == tmp_path / "pygeoinf"

    def test_the_bundled_table_is_read_when_the_cache_is_empty(self, cache):
        path = datasets.table_path(datasets.GSN_STATIONS)
        assert path is not None and "pygeoinf2" in str(path)
        table = datasets.read_table(datasets.GSN_STATIONS)
        assert table["Latitude"].dtype == float
        assert table["Station"].dtype == object

    def test_a_cached_copy_takes_precedence(self, cache):
        cache.mkdir()
        (cache / datasets.GSN_STATIONS).write_text(
            "Station,Latitude,Longitude\nXYZ,1.0,2.0\n", encoding="utf-8"
        )
        assert (
            datasets.table_path(datasets.GSN_STATIONS) == cache / datasets.GSN_STATIONS
        )
        assert datasets.read_table(datasets.GSN_STATIONS)["Latitude"].tolist() == [1.0]

    def test_a_missing_table_names_the_download(self, cache):
        with pytest.raises(FileNotFoundError, match="download_gsn_stations"):
            datasets.read_table("no_such_table.csv")


class TestTheDownloads:
    def test_stations_are_parsed_and_written(self, cache, stub):
        path = datasets.download_gsn_stations()
        assert path == cache / datasets.GSN_STATIONS
        table = datasets.read_table(datasets.GSN_STATIONS)
        assert table["Station"].tolist() == ["AAK", "ABKT"]
        assert np.allclose(table["Longitude"], [74.4942, 58.1189])
        assert stub[0][1] == {"network": "IU,II", "level": "station", "format": "text"}

    def test_an_existing_copy_is_kept_unless_forced(self, cache, stub):
        datasets.download_gsn_stations()
        datasets.download_gsn_stations()
        assert len(stub) == 1
        datasets.download_gsn_stations(force=True)
        assert len(stub) == 2

    def test_earthquakes_carry_the_filters(self, cache, stub):
        path = datasets.download_usgs_earthquakes(
            minimum_magnitude=6.0,
            start_time="2020-01-01",
            maximum_depth=70.0,
            bounding_box=(-10.0, 10.0, 100.0, 150.0),
            limit=50,
        )
        assert path == cache / datasets.USGS_EVENTS
        params = stub[0][1]
        assert params["minmagnitude"] == 6.0 and params["limit"] == 50
        assert params["starttime"] == "2020-01-01" and params["maxdepth"] == 70.0
        assert (params["minlatitude"], params["maxlongitude"]) == (-10.0, 150.0)
        assert "mindepth" not in params
        table = datasets.read_table(datasets.USGS_EVENTS)
        assert table["mag"].tolist() == [5.1, 5.0, 6.2]

    def test_a_refreshed_catalogue_reaches_the_sphere(self, cache, stub):
        from pygeoinf2.symmetric_space.sphere import Lebesgue

        sphere = Lebesgue(4)
        before = len(sphere.earthquakes())
        datasets.download_usgs_earthquakes()
        assert len(sphere.earthquakes()) == 3 != before
        assert len(sphere.earthquakes(minimum_magnitude=6.0)) == 1
        datasets.download_gsn_stations()
        assert len(sphere.stations()) == 2

    def test_too_many_names_the_download_rather_than_fetching(self, cache, stub):
        from pygeoinf2.symmetric_space.sphere import Lebesgue

        sphere = Lebesgue(4)
        with pytest.raises(ValueError, match="download_usgs_earthquakes"):
            sphere.earthquakes(count=10**6)
        with pytest.raises(ValueError, match="download_gsn_stations"):
            sphere.stations(count=10**6)
        assert stub == []

    def test_bad_arguments_and_failures_are_reported(self, cache, monkeypatch):
        with pytest.raises(ValueError, match="limit"):
            datasets.download_usgs_earthquakes(limit=0)
        with pytest.raises(ValueError, match="bounding box"):
            datasets.download_usgs_earthquakes(bounding_box=(1.0, 2.0))

        def failing(url, params, /, *, timeout):
            raise OSError("no route")

        monkeypatch.setattr(datasets, "_fetch", failing)
        with pytest.raises(RuntimeError, match="IRIS"):
            datasets.download_gsn_stations()
        with pytest.raises(RuntimeError, match="USGS"):
            datasets.download_usgs_earthquakes()
