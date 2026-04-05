"""
pygeoinf/config.py
Shared configuration constants for the pygeoinf library.
"""

from os.path import dirname, join as joinpath

# Define the path to the package's data directory
DATADIR = joinpath(dirname(__file__), "../data")
