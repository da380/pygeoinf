# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# This line points Sphinx to the root directory of your project so it can find your library.
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pygeoinf"
copyright = "2025-2026, David Al-Attar, Dan Heathcote, Adrian Mag"
author = "David Al-Attar, Dan Heathcote, Adrian Mag"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add the Sphinx extensions necessary for a modern documentation site.
extensions = [
    "sphinx.ext.autodoc",  # Automatically generate docs from docstrings.
    "sphinx.ext.napoleon",  # Enables Sphinx to understand NumPy-style docstrings.
    "sphinx.ext.viewcode",  # Adds links to the source code from the documentation.
]

templates_path = ["_templates"]
exclude_patterns = []

# The docstrings use Markdown's convention of single backticks for inline code
# (`HilbertSpace`, `cartopy`). Left to itself reStructuredText reads those as
# "title reference" and renders them as italics; this makes them render as code,
# which is what they mean everywhere in this project.
default_role = "code"

# The result dataclasses (BundleResult, KKTResult, ...) describe their fields in
# an `Attributes:` docstring section *and* declare them as annotated fields.
# Napoleon's default turns that section into standalone `.. attribute::`
# directives, which autodoc then documents a second time from the annotations.
# Rendering them as info-field entries instead keeps the descriptions and leaves
# each field defined exactly once.
napoleon_use_ivar = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set the HTML theme to 'furo' for a clean, modern look.
html_theme = "furo"
html_static_path = ["_static"]


# -- Generated API reference -------------------------------------------------
# sphinx-apidoc runs from here rather than from .readthedocs.yaml, so that a
# local build and a Read the Docs build produce the same pages from the same
# settings. The .rst files it writes are build output, not source, and are
# gitignored; previously they were committed, went stale, and quietly hid whole
# modules from anyone building the docs locally.

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SOURCE_DIR, "..", ".."))


def _run_apidoc(_app):
    """Regenerate the per-module .rst stubs before the build reads them.

    Stubs left by an earlier build of a different tree are removed first:
    Sphinx builds every ``.rst`` it finds, and a stub for a module that no
    longer exists is an import error in the log and a dead page.
    """
    import glob

    from sphinx.ext.apidoc import main

    for stale in glob.glob(os.path.join(_SOURCE_DIR, "pygeoinf*.rst")) + [
        os.path.join(_SOURCE_DIR, "modules.rst")
    ]:
        if os.path.exists(stale):
            os.remove(stale)
    main(
        [
            "--force",
            "--templatedir",
            os.path.join(_REPO_ROOT, "docs", "apidoc_templates"),
            "--output-dir",
            _SOURCE_DIR,
            os.path.join(_REPO_ROOT, "pygeoinf"),
        ]
    )


def setup(app):
    app.connect("builder-inited", _run_apidoc)
