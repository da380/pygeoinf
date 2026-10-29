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
}

# Tutorials whose cost is part of the lesson, run with the slow marker.
# Whole-notebook runs that take a minute or more at their own scale; they
# belong to the slow set, which the scheduled CI job runs, and not to the
# run every push waits for.
SLOW: set[str] = {"13_ray_tomography", "14_one_problem_three_answers"}


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
    readme = (TUTORIALS / "README.md").read_text(encoding="utf-8")
    missing = [
        script.name for script in tutorial_scripts() if script.name not in readme
    ]
    assert not missing, f"not mentioned in the README: {missing}"


def _make_notebooks():
    """The notebook generator, loaded from its script."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_notebooks", TUTORIALS / "make_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_notebooks_are_current():
    """Each tutorial has a notebook generated from it, and it is up to date.

    The notebooks are what Colab opens, so a stale one teaches an older
    tutorial than the script the suite ran. ``make_notebooks.py`` regenerates
    them.
    """
    changes = _make_notebooks().stale()
    assert not changes, (
        "the notebooks are stale; run pygeoinf2/tutorials/make_notebooks.py: "
        + ", ".join(changes)
    )


def test_the_badges_open_the_generated_notebooks():
    """Each Colab badge opens its own tutorial's notebook, on the branch the
    notebooks install the library from, so the two cannot drift apart."""
    branch = _make_notebooks().BRANCH
    readme = (TUTORIALS / "README.md").read_text(encoding="utf-8")
    missing = [
        script.name
        for script in tutorial_scripts()
        if f"/blob/{branch}/pygeoinf2/tutorials/notebooks/{script.stem}.ipynb)"
        not in readme
    ]
    assert not missing, f"no badge opening the generated notebook: {missing}"
