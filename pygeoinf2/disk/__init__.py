"""A disk or an annulus, in the eigenbasis of ``A = 1 - div(L^2 grad)``.

Fields of position in the plane, in ``inner_radius <= r <= radius``: the
two-dimensional counterpart of :mod:`pygeoinf2.ball`, and built the same way.
The length scale ``L`` may vary with radius, so ``A`` commutes with rotations
about the centre and its eigenfunctions separate: a Fourier mode of the angle
times a radial eigenfunction of the operator ``A_m`` of its order, under the
measure ``r dr``. A field's components are its coefficients in those
products, ordered by order ``m``, then the cosine mode and, for ``m >= 1``,
the sine mode, and within each mode the radial modes in ascending
eigenvalue; :attr:`Disk.orders` and :attr:`Disk.radial_indices` label them.

The radial mesh is padded, as the ball's is and for the same reason: ``A``
needs a boundary condition at each end, and the padding keeps its mark out
of the domain. Outwards the padding is a fictitious ring; inwards it stops at
the centre, which is a regular point of the operator and needs neither
padding nor a condition, so a disk is padded on the outside only.

**The conventions are the package's.** A vector is the array of the field's
values on the padded grid, shape ``(radii, angles)``, the angles being the
``2 mmax + 1`` equally spaced ones of the truncation order, from zero
anticlockwise; a product is left on the grid, untruncated; a point is
``(radius, angle)``, the angle in degrees anticlockwise from the ``x`` axis,
as a longitude is east of Greenwich on the sphere; the Fourier modes are
orthonormal on the unit circle -- ``1 / sqrt(2 pi)``, ``cos(m t) / sqrt(pi)``,
``sin(m t) / sqrt(pi)`` -- so the Lebesgue inner product is the integral over
the area; and point evaluation is refused at or below order one.

**A field regular at the centre grows there in a way its order fixes.** The
part of order ``m`` of a field analytic at the centre of a disk goes like
``r^m`` times a series in ``r^2``: a polynomial in ``x`` and ``y`` is the
safe example. The radial modes here do so by construction -- for a constant
``L`` they are the Bessel functions ``J_m(k r)``, to the accuracy of the mesh
-- so every draw of a prior is regular there automatically. A function
*handed in* has to be regular itself, or the expansion converges slowly at
the centre: a standard deviation, a multiplier or a mean given as a profile
in ``r`` is of order zero and must be even in ``r``. In an annulus there is no
centre and nothing to it.

The numerics are ``planetmodel``'s spectral elements under the ``r dr``
measure, which ``planetmodel`` does not offer: its radial operator family
with the assembly changed, one ``planetmodel`` ``SpectralBasis`` per order
held in a :class:`~pygeoinf2.spectral_element._cylindrical.PolarBasis`
(reachable as :attr:`Disk.basis`), and a real FFT round each ring.
"""

from .layered import Layered
from .space import Disk, Lebesgue, Sobolev

__all__ = ["Disk", "Layered", "Lebesgue", "Sobolev"]
