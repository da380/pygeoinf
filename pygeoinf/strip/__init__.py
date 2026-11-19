"""A strip of the plane, in the eigenbasis of ``A = 1 - div(L^2 grad)``.

Fields of ``(depth, x)`` in a flat layer, ``top <= depth <= bottom``, over one
lateral axis: a vertical section through a flat earth, the Cartesian
counterpart of :mod:`pygeoinf.disk`, as :mod:`pygeoinf.slab` is of
:mod:`pygeoinf.ball`. The length scale ``L`` may vary with depth, so ``A``
commutes with translations along ``x`` and its eigenfunctions separate: a
Fourier mode of the lateral period times an eigenfunction in depth of the
operator ``A_k`` of its wavenumber, under the plain measure ``dz``. A field's
components are its coefficients in those products, the Fourier modes in the
order of ``planetmodel.fourier``'s packing -- those that are their own
conjugate first, the constant and the Nyquist cosine of an even grid, then
each cosine with its sine -- and within each the modes of depth in ascending
eigenvalue; :attr:`Strip.wavevectors`, :attr:`Strip.phases` and
:attr:`Strip.depth_indices` label them.

This is not :class:`pygeoinf.symmetric_space.box.Box`, which is Fourier in
every direction and has one length scale. Here only the lateral direction is
periodic; depth is a spectral-element mesh, on which the length scale may
vary and a layer has a top and a base.

**Depth is padded, as a radius is, and for the same reason**: ``A`` needs a
condition at each end of the mesh, and the padding keeps its mark out of the
domain. The mesh extends above the top, into negative depth under a free
surface, as below the base, and the condition is applied at its ends:
nothing is imposed at the top or the base of the layer itself.

**Laterally the field is periodic**, over the period of the grid. By default
the domain is the whole period, and a point is read at its image in it. A
``planetmodel.fourier.PaddedDomain`` given as ``lateral=`` makes the domain a
bounded interval inside a wider period, the padding keeping it clear of its
periodic images: then :attr:`Strip.interior_mask` leaves the lateral padding
out as it does the padding in depth, and a point outside the bounds is
refused.

**The conventions are the package's.** A vector is the array of the field's
values on the padded grid, shape ``(depths, x)``; a product is left on the
grid, untruncated; a point is ``(depth, x)``, the layered coordinate first as
a radius is in a disk, with depth positive downward; the Fourier modes are
orthonormal over the period, so the Lebesgue inner product is the integral
over the area; and point evaluation is refused at or below order one.

**The pointwise variance of a prior is exact at the grid's points.** On the
grid a cosine and its sine have squares summing to a constant, so the
variance of a prior that is a function of ``A`` depends on depth alone, and a
calibration by ``pointwise_std=`` is exact there. Between the points the
same holds of every mode but one: the cosine at the Nyquist wavenumber of a
grid of even size, which the grid holds without its sine. With an even grid
the variance between its points therefore dips below the calibrated value by
that mode's share, the smallest of the spectrum; a grid of odd size has no
such mode, and is exact everywhere across.

**Layers are stacked in depth** by :class:`Layered`: a strip in each layer
between consecutive depths, each with its own length scale and truncation in
depth, all over one lateral period and grid, the field smooth within a layer
and free to jump between them. Depth is positive downward and the sides of
an interface keep their physical sense: *above* is the shallower layer,
*below* the deeper, and the jump across an interface, a horizontal line, is
the value above less the value below, taken at given positions ``x``.

The numerics are ``planetmodel``'s: ``SeparableOperatorFamily`` under the
plain measure and indexed by the lateral eigenvalue, one ``SpectralBasis``
per distinct ``k^2`` held in a ``planetmodel.cartesian.LayerBasis``
(reachable as :attr:`Strip.basis`), and the lateral transform of
``planetmodel.fourier`` (:attr:`Strip.transform`).
"""

from .layered import Layered
from .space import Lebesgue, Sobolev, Strip

__all__ = ["Layered", "Lebesgue", "Sobolev", "Strip"]
