"""A ball or an annulus, in the eigenbasis of ``A = 1 - div(L^2 grad)``.

Fields of position in ``inner_radius <= r <= radius``, which no periodic box
embeds with the right measure and no symmetric space here covers. The length
scale ``L`` may vary with radius, so ``A`` commutes with rotations and its
eigenfunctions separate: a real spherical harmonic in the angles times a
radial eigenfunction of the operator ``A_l`` of its degree, under the measure
``r^2 dr``. A field's components are its coefficients in those products,
ordered by degree ``l``, then the cosine harmonics of orders ``0 ... l``, then
the sine harmonics of orders ``1 ... l``, and within each harmonic the radial
modes in ascending eigenvalue; :attr:`Ball.degrees`, :attr:`Ball.orders` and
:attr:`Ball.radial_indices` label them.

The radial mesh is padded, as the interval's is and for the same reason:
``A`` needs a boundary condition at each end, and the padding keeps its mark
out of the domain. Outwards the padding is a fictitious shell; inwards it
stops at the centre, which is a regular point of the operator and needs
neither padding nor a condition, so a ball is padded on the outside only.

**The conventions are the package's.** A vector is the array of the field's
values on the padded grid, shape ``(radii, latitudes, longitudes)``, the
angular grid being the Gauss-Legendre one of the truncation degree; a product
is left on the grid, untruncated; a point is ``(radius, latitude, longitude)``
with the angles in degrees; the harmonics are orthonormal on the unit sphere
without the Condon-Shortley phase; and point evaluation is refused at or below
order three halves.

**A field regular at the centre grows there in a way its degree fixes.** The
part of degree ``l`` of a field analytic at the centre of a ball goes like
``r^l`` times a series in ``r^2``: ``u ~ r^l Y_lm``. The radial modes here do
so by construction -- for a constant ``L`` they are the spherical Bessel
functions ``j_l(k r)``, to the accuracy of the mesh -- so every draw of a
prior is regular there automatically. A function *handed in* has to be
regular itself, or the expansion converges slowly at the centre: a standard
deviation, a multiplier or a mean given as a profile in ``r`` is of degree
zero and must be even in ``r``. One linear in ``r`` is a cone, and converges
at the centre only at first order, where a profile in ``r^2`` converges
quickly. In an annulus there is no centre and nothing to it.

The numerics are ``planetmodel``'s: ``RadialOperatorFamily`` under the
``r^2`` measure, one ``SpectralBasis`` per degree held in a
``SphericalBasis`` (reachable as :attr:`Ball.basis`), and its harmonics. The
transform between coefficients and the grid is this package's, all the radial
shells at once; it agrees with ``planetmodel``'s shell-by-shell route to
rounding and is used in its place up to degree 128.
"""

from .layered import Layered
from .space import Ball, Lebesgue, Sobolev

__all__ = ["Ball", "Layered", "Lebesgue", "Sobolev"]
