"""
The examples must keep working.

An example that has quietly stopped running is worse than no example, so each
one is executed here. They are short by construction, so the whole set costs
little.
"""

import pathlib
import runpy

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def example_scripts() -> list[pathlib.Path]:
    """Every numbered example, in reading order."""
    return sorted(EXAMPLES.glob("[0-9][0-9]_*.py"))


def test_the_examples_are_discovered():
    """Guards against the glob silently matching nothing."""
    assert len(example_scripts()) >= 15


@pytest.mark.parametrize("script", example_scripts(), ids=lambda p: p.stem)
def test_the_example_runs(script, capsys):
    """Run the script and require that it produces output without raising."""
    if "sphere" in script.stem:
        pytest.importorskip("pyshtools")
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
