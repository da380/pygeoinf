"""Generates the tutorial notebooks from the tutorial scripts.

The numbered scripts in this directory are the source of the tutorials.
They are written in cells, the ``# %%`` convention that editors such as
VS Code and Spyder run cell by cell, and the test suite runs them headless.
Each is converted here into a notebook under ``notebooks/``, for working in
Jupyter or trying the library online in Google Colab without installing
anything. A notebook gains one cell, before its first code cell, that
installs the library when it runs on Colab and does nothing elsewhere.

The conversion is mechanical and deterministic. A ``# %% [markdown]`` cell
becomes a markdown cell with its comment markers removed, any other ``# %%``
cell a code cell, and the cell ids are numbered as nbstripout numbers them,
so an unchanged script regenerates its notebook byte for byte. A notebook
whose script has gone is deleted. Run it by hand,

    poetry run python tutorials/make_notebooks.py

after editing a tutorial. It exits with status 1 when it has changed
anything, and the test suite checks that the notebooks are current, so a
commit with stale notebooks fails a test rather than shipping them.
"""

import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE / "notebooks"

MARKER = "# %%"

# The branch of the repository the Colab cell installs from, and the README's
# badges open the notebooks on.
BRANCH = "<release-branch>"

INSTALL = """\
# Generated from tutorials/{name}.
# Edit the script, not this notebook. On Google Colab the library is
# installed first, from GitHub, with the extras this tutorial needs{note};
# elsewhere this cell does nothing.
import subprocess
import sys

if "google.colab" in sys.modules:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "pygeoinf[{extras}] @ git+https://github.com/da380/pygeoinf@{branch}",
        ],
        check=True,
    )"""

# The extras each tutorial installs on Colab. Every one takes the sphere
# extra, since the plotting layer draws maps through it.
EXTRAS = {
    "10_coupled_problems": "sphere,planetmodel",
    "11_varying_structure": "sphere,planetmodel",
}
DEFAULT_EXTRAS = "sphere"

# What a Colab install cannot provide, said in the cell.
NOTES: dict[str, str] = {}

METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def scripts() -> list[Path]:
    """The numbered tutorial scripts, in reading order."""
    return sorted(HERE.glob("[0-9][0-9]_*.py"))


def _cells(script: Path) -> list[tuple[str, str]]:
    """The script's cells as (kind, source) pairs, kind 'markdown' or
    'code', with empty cells dropped."""
    blocks: list[tuple[str, list[str]]] = []
    for number, line in enumerate(
        script.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.startswith(MARKER):
            tag = line[len(MARKER) :].strip()
            if tag not in ("", "[markdown]"):
                raise ValueError(f"{script.name}:{number}: unknown cell tag {tag!r}")
            blocks.append(("markdown" if tag else "code", []))
        elif blocks:
            blocks[-1][1].append(line)
        elif line.strip():
            raise ValueError(f"{script.name}:{number}: text before the first cell")
    cells = []
    for kind, lines in blocks:
        if kind == "markdown":
            text = []
            for line in lines:
                if line.strip() and line != "#" and not line.startswith("# "):
                    raise ValueError(
                        f"{script.name}: markdown line not commented: {line!r}"
                    )
                text.append(line[2:] if line.startswith("# ") else "")
            lines = text
        source = "\n".join(lines).strip("\n")
        if source.strip():
            cells.append((kind, source))
    return cells


def notebook(script: Path) -> str:
    """The text of the notebook generated from a script."""
    cells = _cells(script)
    first_code = next((i for i, (kind, _) in enumerate(cells) if kind == "code"), 0)
    install = INSTALL.format(
        name=script.name,
        extras=EXTRAS.get(script.stem, DEFAULT_EXTRAS),
        note=NOTES.get(script.stem, ""),
        branch=BRANCH,
    )
    cells.insert(first_code, ("code", install))
    nb = nbformat.v4.new_notebook(metadata=METADATA)
    for i, (kind, source) in enumerate(cells):
        if kind == "markdown":
            cell = nbformat.v4.new_markdown_cell(source)
        else:
            cell = nbformat.v4.new_code_cell(source)
        cell["id"] = str(i)
        nb.cells.append(cell)
    nbformat.validate(nb)
    text = nbformat.writes(nb)
    return text if text.endswith("\n") else text + "\n"


def stale() -> list[str]:
    """What :func:`main` would change: notebooks to write or delete."""
    wanted = set()
    changes = []
    for script in scripts():
        target = NOTEBOOKS / (script.stem + ".ipynb")
        wanted.add(target)
        if not target.exists() or target.read_text(encoding="utf-8") != notebook(
            script
        ):
            changes.append(f"write {target.name}")
    for orphan in sorted(set(NOTEBOOKS.glob("*.ipynb")) - wanted):
        changes.append(f"delete {orphan.name}")
    return changes


def main() -> int:
    NOTEBOOKS.mkdir(exist_ok=True)
    changed = []
    wanted = set()
    for script in scripts():
        target = NOTEBOOKS / (script.stem + ".ipynb")
        wanted.add(target)
        text = notebook(script)
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
            changed.append(f"wrote {target.relative_to(HERE.parent)}")
    for orphan in sorted(set(NOTEBOOKS.glob("*.ipynb")) - wanted):
        orphan.unlink()
        changed.append(f"deleted {orphan.relative_to(HERE.parent)}")
    for line in changed:
        print(line)
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
