"""The machinery behind the spaces on a spectral-element mesh.

The symmetric spaces diagonalize the Laplacian of a homogeneous domain by a
fast transform. The spaces built here diagonalize ``A = 1 - div(L^2 grad)``
on a bounded one, with a length scale ``L`` that may vary from place to place,
by solving for its eigenfunctions on a Gauss-Lobatto-Legendre mesh. What that
buys is what the Fourier interval cannot do: a correlation length that
changes across the domain, a mesh refined where it needs to be, and -- in the
ball and the annulus, which no periodic box embeds -- the ``r^2 dr`` measure.

The numerics are those of ``planetmodel.randomfield``, an optional dependency
(``pip install pygeoinf[planetmodel]``); this package wraps them as spaces,
operators and Gaussian measures. The domain is reached by restriction: the mesh
extends past it by a padding, so that the boundary conditions ``A`` needs act
at a distance and leave no mark on the fields where they matter.

**A user reaches the spaces by geometry**, and never imports this package:

:mod:`pygeoinf2.interval`
    an interval of the *line*, under ``dx``, where the coordinate is a
    position and zero is nowhere special; and, as
    :mod:`pygeoinf2.interval.radial`, functions of *radius* alone in a ball
    or a shell, under ``r^2 dr``.
:mod:`pygeoinf2.disk`
    fields of position in a disk or an annulus of the plane, under ``r dr``
    and the circle's Fourier modes, the two-dimensional counterpart of the
    ball.
:mod:`pygeoinf2.ball`
    fields of position in a ball or an annulus, under ``r^2 dr`` and the area
    of the sphere.

Each exports ``Lebesgue`` and ``Sobolev``, and a ``Layered`` space of
piecewise-continuous fields over layers of its geometry that are adjacent in
space. What is here is what they share and what a backend author may build
on: :class:`SpectralElementSpace`, the Sobolev scale, the pointwise algebra,
point evaluation and the Gaussian measures of any such space;
:class:`AxisSpace`, a space of functions of one coordinate on a padded mesh;
the :class:`Layered` direct sum, which the geometries subclass; the
:data:`Boundary` and :data:`LengthScale` argument types; and
:class:`ShellTransform`, the batched harmonic transform of a ball's shells.

The mesh ends with a Robin condition matched to the operator, under which two
length scales of padding are enough; ``boundary=None`` is the natural condition
and wants four. A function handed in is fitted over the domain alone, which asks
nothing of the padding; ``extension="constant"`` or ``"odd"`` continue its own
nodal values across it, the first being what keeps a standard deviation
positive there.
"""

try:
    import planetmodel.randomfield as _randomfield
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "The spectral-element spaces need planetmodel, an optional dependency: "
        "pip install pygeoinf[planetmodel]."
    ) from error

import planetmodel.sampling as _sampling

# 1.3 is where a grid carries its kind, which the ball's transform checks.
_grid_fields = getattr(
    getattr(_sampling, "AngularGrid", None), "__dataclass_fields__", {}
)
if (
    not hasattr(_randomfield, "SpectralBasis") or "kind" not in _grid_fields
):  # pragma: no cover
    raise ImportError(
        "The spectral-element spaces need planetmodel 1.3 or later, whose grids "
        "carry their kind: pip install -U planetmodel."
    )
del _grid_fields

from ._transform import ShellTransform  # noqa: E402
from .axis import AxisSpace  # noqa: E402
from .base import Boundary, LengthScale, SpectralElementSpace  # noqa: E402
from .layered import Layered  # noqa: E402

__all__ = [
    "AxisSpace",
    "Boundary",
    "Layered",
    "LengthScale",
    "ShellTransform",
    "SpectralElementSpace",
]
