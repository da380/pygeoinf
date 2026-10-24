"""
The tutorials must keep working.

The same contract as the examples: each one is executed, top to bottom, and
must print something. A tutorial that has quietly stopped running teaches the
reader that the library is broken, which is worse than teaching nothing.
"""

import pathlib
import runpy

import matplotlib

matplotlib.use("Agg")

import pytest

TUTORIALS = pathlib.Path(__file__).resolve().parent.parent / "tutorials"


def tutorial_scripts() -> list[pathlib.Path]:
    """Every numbered tutorial, in reading order."""
    return sorted(TUTORIALS.glob("[0-9][0-9]_*.py"))


def test_the_tutorials_are_discovered():
    """Guards against the glob silently matching nothing."""
    assert len(tutorial_scripts()) >= 3


# Tutorials that need an optional dependency, and the module that provides it,
# or the modules where there are several.
OPTIONAL: dict[str, str | tuple[str, ...]] = {
    "10_coupled_problems": "planetmodel",
    "11_varying_structure": "planetmodel",
    "12_writing_a_backend": "mfem",
}

# Tutorials whose cost is part of the lesson, run with the slow marker.
SLOW: set[str] = set()


def _parametrized() -> list:
    """Every tutorial, with the expensive ones carrying the slow marker."""
    return [
        pytest.param(
            script,
            id=script.stem,
            marks=pytest.mark.slow if script.stem in SLOW else (),
        )
        for script in tutorial_scripts()
    ]


@pytest.mark.parametrize("script", _parametrized())
def test_the_tutorial_runs(script, capsys):
    """Run the script and require that it produces output without raising."""
    needed = OPTIONAL.get(script.stem, ())
    for module in (needed,) if isinstance(needed, str) else needed:
        pytest.importorskip(module)
    runpy.run_path(str(script), run_name="__main__")
    assert capsys.readouterr().out.strip(), f"{script.name} printed nothing"


def test_every_tutorial_is_listed_in_the_readme():
    """A tutorial nobody is pointed at is a tutorial nobody reads."""
    readme = (TUTORIALS / "README.md").read_text()
    missing = [
        script.name for script in tutorial_scripts() if script.name not in readme
    ]
    assert not missing, f"not mentioned in the README: {missing}"
