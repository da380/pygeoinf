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

from .conftest import coastlines_are_cached

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def example_scripts() -> list[pathlib.Path]:
    """Every numbered example, in reading order."""
    return sorted(EXAMPLES.glob("[0-9][0-9]_*.py"))


def test_the_examples_are_discovered():
    """Guards against the glob silently matching nothing."""
    assert len(example_scripts()) >= 36


# Examples that need an optional dependency, and the module that provides it,
# or the modules where there are several.
OPTIONAL = {
    "20_observation": "pyshtools",
    "21_flexure": "cartopy",
    "24_tomography": "cartopy",
    "26_distributions": "pyshtools",
    "27_coupled_fields": "cartopy",
    "28_preconditioning": "pyshtools",
    "30_feasible_set": "cartopy",
    "32_varying_length_scale": "planetmodel",
    "33_disk": "planetmodel",
    "34_mantle": ("planetmodel", "pyshtools", "cartopy"),
    "35_density_profile": "planetmodel",
    "36_layered_planet": "planetmodel",
}


# Examples that draw coastlines. Cartopy fetches the Natural Earth shapefile on
# first use, and a test suite that reaches the network is a test suite that
# fails for reasons unconnected to the code.
NEEDS_COASTLINES = {
    "21_flexure",
    "24_tomography",
    "27_coupled_fields",
    "30_feasible_set",
}


# Examples whose *cost is the point*: an iteration count on a badly-scaled
# problem, a posterior covariance formed block by block. Shrinking them would
# make them demonstrate something else, so they are marked slow instead and run
# occasionally rather than on every invocation.
SLOW = {
    "27_coupled_fields",
    "28_preconditioning",
}


def _parametrized() -> list:
    """Every example, with the expensive ones carrying the slow marker.

    Marked at parametrization rather than skipped inside the test, so that
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


@pytest.mark.parametrize("script", _parametrized())
def test_the_example_runs(script, capsys):
    """Run the script and require that it produces output without raising."""
    needed = OPTIONAL.get(script.stem, ())
    for module in (needed,) if isinstance(needed, str) else needed:
        pytest.importorskip(module)
    if script.stem in NEEDS_COASTLINES and not coastlines_are_cached():
        pytest.skip("the Natural Earth coastline data is not cached locally")
    runpy.run_path(str(script), run_name="__main__")
    assert capsys.readouterr().out.strip(), f"{script.name} printed nothing"


def test_every_example_is_listed_in_the_readme():
    """A script nobody is pointed at is a script nobody reads."""
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    missing = [
        script.stem
        for script in example_scripts()
        if script.stem.split("_", 1)[1] not in readme
    ]
    assert not missing, f"not mentioned in the README: {missing}"


def test_the_readme_lists_the_optional_dependencies():
    """The README's table of optional dependencies is this file's ``OPTIONAL``,
    module by module, so that what it tells a reader to install is what the
    examples actually need."""
    import re

    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    listed = {
        module: {int(n) for n in numbers.split(",")}
        for module, numbers in re.findall(r"^\| `(\w+)` \| ([\d, ]+) \|$", readme, re.M)
    }
    needed: dict[str, set[int]] = {}
    for stem, modules in OPTIONAL.items():
        for module in (modules,) if isinstance(modules, str) else modules:
            needed.setdefault(module, set()).add(int(stem[:2]))
    assert listed == needed
