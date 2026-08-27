"""
The examples must keep working.

An example that has quietly stopped running is worse than no example, so each
one is executed here. They are short by construction, so the whole set costs
little.
"""

import pathlib
import runpy

import matplotlib

matplotlib.use("Agg")

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def example_scripts() -> list[pathlib.Path]:
    """Every numbered example, in reading order."""
    return sorted(EXAMPLES.glob("[0-9][0-9]_*.py"))


def test_the_examples_are_discovered():
    """Guards against the glob silently matching nothing."""
    assert len(example_scripts()) >= 26


# Examples that need an optional dependency, and the module that provides it.
OPTIONAL = {
    "16_mfem_backend": "mfem",
    "19_observation": "pyshtools",
    "20_flexure": "cartopy",
    "21_tomography": "cartopy",
    "22_coupled_fields": "cartopy",
    "23_feasible_set": "cartopy",
    "24_preconditioning": "pyshtools",
    "25_distributions": "pyshtools",
}


# Examples that draw coastlines. Cartopy fetches the Natural Earth shapefile on
# first use, and a test suite that reaches the network is a test suite that
# fails for reasons unconnected to the code.
NEEDS_COASTLINES = {
    "20_flexure",
    "21_tomography",
    "22_coupled_fields",
    "23_feasible_set",
}


def _coastlines_are_cached() -> bool:
    """True when cartopy already has the land shapefile on disk."""
    try:
        from cartopy.io import shapereader
    except ImportError:
        return False
    import unittest.mock

    with unittest.mock.patch("cartopy.io.Downloader.acquire_resource") as blocked:
        blocked.side_effect = AssertionError("would download")
        try:
            shapereader.natural_earth(
                resolution="110m", category="physical", name="land"
            )
        except Exception:
            return False
    return True


# Examples whose *cost is the point*: an iteration count on a badly-scaled
# problem, a posterior covariance formed block by block. Shrinking them would
# make them demonstrate something else, so they are marked slow instead and run
# occasionally rather than on every invocation.
SLOW = {
    "22_coupled_fields",
    "24_preconditioning",
}


def _parametrised() -> list:
    """Every example, with the expensive ones carrying the slow marker.

    Marked at parametrisation rather than skipped inside the test, so that
    ``-m slow`` selects them and ``-m ""`` runs everything — which a skip
    written against the ``-m`` string cannot do, and got backwards the first
    time: ``"not slow".endswith("slow")`` is true.
    """
    return [
        pytest.param(
            script,
            id=script.stem,
            marks=pytest.mark.slow if script.stem in SLOW else (),
        )
        for script in example_scripts()
    ]


@pytest.mark.parametrize("script", _parametrised())
def test_the_example_runs(script, capsys):
    """Run the script and require that it produces output without raising."""
    if script.stem in OPTIONAL:
        pytest.importorskip(OPTIONAL[script.stem])
    if script.stem in NEEDS_COASTLINES and not _coastlines_are_cached():
        pytest.skip("the Natural Earth coastline data is not cached locally")
    runpy.run_path(str(script), run_name="__main__")
    assert capsys.readouterr().out.strip(), f"{script.name} printed nothing"


def test_every_example_is_listed_in_the_readme():
    """A script nobody is pointed at is a script nobody reads."""
    readme = (EXAMPLES / "README.md").read_text()
    missing = [
        script.stem
        for script in example_scripts()
        if script.stem.split("_", 1)[1] not in readme
    ]
    assert not missing, f"not mentioned in the README: {missing}"
