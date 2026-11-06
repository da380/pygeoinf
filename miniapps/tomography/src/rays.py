"""The ray tracer: the mini-apps' shared one, loaded as an external code.

The tracer is not this app's own: it is ``raytracer.py`` in
``miniapps/raytracer``, beside the apps, which every app that traces rays
loads. Its directory is put on the path for the import and taken off again,
and the app's scripts and modules import the tracer from here, as
``src.rays``: the radial tracer and its rays, the closed form of the
flattened linear gradient, and the shooting tracer of the bent rays.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "raytracer"))
try:
    from raytracer import (
        CircularArcs,
        RadialProfile,
        Ray,
        Shots,
        Slowness,
        bend,
        flattened_linear_gradient,
        linear_gradient,
        polyline_weights,
        shoot,
        simpson_weights,
    )
finally:
    sys.path.pop(0)

__all__ = [
    "CircularArcs",
    "RadialProfile",
    "Ray",
    "Shots",
    "Slowness",
    "bend",
    "flattened_linear_gradient",
    "linear_gradient",
    "polyline_weights",
    "shoot",
    "simpson_weights",
]
