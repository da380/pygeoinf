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
copyright = "2025, David Al-Attar, Dan Heathcote, Adrian Mag"
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


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set the HTML theme to 'furo' for a clean, modern look.
html_theme = "furo"
html_static_path = ["_static"]
