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
(DECISIONS.md D-3)::

    from pygeoinf2.radial.interval import Sobolev

None of this is imported by the package itself, ``planetmodel`` being
optional, in the way the sphere is not.
"""

try:
    import planetmodel.randomfield as _randomfield
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "pygeoinf2.radial needs planetmodel, an optional dependency: "
        "pip install pygeoinf[planetmodel]."
    ) from error

if not hasattr(_randomfield, "SpectralBasis"):  # pragma: no cover
    raise ImportError(
        "pygeoinf2.radial needs planetmodel 1.2 or later, whose randomfield "
        "package has the spectral bases: pip install -U planetmodel."
    )
