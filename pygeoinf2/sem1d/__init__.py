"""Spaces in the eigenbasis of an elliptic operator on a spectral-element mesh.

The symmetric spaces diagonalize the Laplacian of a homogeneous domain by a
fast transform. The spaces here diagonalize ``A = 1 - div(L^2 grad)`` on a
bounded one, with a length scale ``L`` that may vary from place to place, by
solving for its eigenfunctions on a Gauss-Lobatto-Legendre mesh. What that
buys is what the Fourier interval cannot do: a correlation length that
changes across the domain, a mesh refined where it needs to be, and -- in the
ball and the annulus, which no periodic box embeds -- the ``r^2 dr`` measure.

The numerics are those of ``planetmodel.randomfield``, an optional dependency
(``pip install pygeoinf[planetmodel]``); this package wraps them as spaces,
operators and Gaussian measures. The domain is reached by restriction: the mesh
extends past it by a padding, so that the boundary conditions ``A`` needs act
at a distance and leave no mark on the fields where they matter.

**One submodule per geometry**, each exporting ``Lebesgue`` and ``Sobolev``
(DECISIONS.md D-3), and the geometries differ first in their measure:

``interval``
    an interval of the *line*, under ``dx``. The coordinate is a position and
    zero is nowhere special.
``radial``
    functions of *radius* alone in a ball or a shell, under ``r^2 dr``: the
    part of degree zero of the ball, for a one-dimensional reference model.
``ball``
    fields of position in a ball or an annulus, under ``r^2 dr`` and the area
    of the sphere.

``layered``
    piecewise-continuous fields over layers of any one of these that are
    adjacent in space: their direct sum, with the points, jumps, integrals and
    layer-by-layer priors a direct sum alone cannot know.

So::

    from pygeoinf2.sem1d.radial import Sobolev
    from pygeoinf2.sem1d.layered import Layered

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
        "pygeoinf2.sem1d needs planetmodel, an optional dependency: "
        "pip install pygeoinf[planetmodel]."
    ) from error

if not hasattr(_randomfield, "SpectralBasis"):  # pragma: no cover
    raise ImportError(
        "pygeoinf2.sem1d needs planetmodel 1.2 or later, whose randomfield "
        "package has the spectral bases: pip install -U planetmodel."
    )
