"""A bounded interval of the line, in the eigenbasis of ``A = 1 - d/dx (L^2
d/dx)``.

**The measure is ``dx`` and the coordinate is a position on a line**, not a
radius: zero is a point like any other, the domain may straddle it, and the
mesh is padded at both ends. A function of radius in a ball or a shell, under
``r^2 dr``, is :mod:`~pygeoinf.interval.radial`.

The counterpart of :mod:`~pygeoinf.symmetric_space.line` for a length scale
``L(x)`` that varies, and for a mesh that is not uniform. The line embeds its
interval in a circle and diagonalizes the Laplacian there by an FFT; this one
embeds it in a longer interval, meshes that with Gauss-Lobatto-Legendre
spectral elements, and solves for the eigenfunctions of ``A`` on the mesh. The
padding plays the same part in both: ``A`` needs a boundary condition at each
end of its mesh, and whichever is chosen distorts a field within about a
correlation length of where it acts. Moving the ends away from ``[lower,
upper]`` moves the distortion out of the domain, and the Robin condition the
mesh ends with by default makes it smaller to begin with, so that half the
padding the natural condition wants is enough.

In the eigenbasis every power of ``A`` is diagonal, so the conveniences of a
symmetric space carry over with ``theta``, the eigenvalues of ``A``, in the
place of ``1 + L^2 k^2``: the ``H^s`` inner product is ``sum theta^s c^2``, a
Sobolev prior has covariance ``theta^-p``, and a draw is one synthesis. What
does *not* carry over is homogeneity. The pointwise variance of such a prior
depends on where it is asked for, which is why ``sobolev_measure`` calibrates
by a standard deviation *field* rather than by one number.

**The conventions are the package's.** A vector is the array of the field's
values at the nodes of the padded mesh, as a ``Line``'s is its values on the
padded grid; its components are the coefficients of the kept eigenfunctions;
a product is left on the nodes, untruncated; point evaluation is refused at
or below the order that admits it, which is one half.

The numerics are ``planetmodel.randomfield``'s: ``padded_mesh``,
``RadialOperatorFamily`` under its plain measure, and the ``SpectralBasis`` of
degree zero, which is reachable as ``basis``.
"""

from . import radial
from .layered import Layered
from .space import Interval, Lebesgue, Sobolev

__all__ = ["Interval", "Layered", "Lebesgue", "Sobolev", "radial"]
