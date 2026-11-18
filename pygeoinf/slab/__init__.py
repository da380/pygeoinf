"""A slab, in the eigenbasis of ``A = 1 - div(L^2 grad)``.

Fields of ``(depth, x, y)`` in a flat layer, ``top <= depth <= bottom``, over
two lateral axes: a box of a flat earth, the Cartesian counterpart of
:mod:`pygeoinf.ball`, as :mod:`pygeoinf.strip` is of :mod:`pygeoinf.disk`.
The length scale ``L`` may vary with depth, so ``A`` commutes with lateral
translations and its eigenfunctions separate: a Fourier mode of the lateral
box times an eigenfunction in depth of the operator ``A_k`` of its
wavevector, under the plain measure ``dz``. ``A_k`` depends on ``k`` through
``|k|^2`` alone, so the wavevectors of one length share their modes of depth,
as the orders of a degree share their radial modes in a ball. A field's
components are its coefficients in those products, the Fourier modes in the
order of ``planetmodel.fourier``'s packing -- those that are their own
conjugate first, then each cosine with its sine -- and within each the modes
of depth in ascending eigenvalue; :attr:`Slab.wavevectors`,
:attr:`Slab.phases` and :attr:`Slab.depth_indices` label them.

This is not :class:`pygeoinf.symmetric_space.box.Box`, which is Fourier in
every direction and has one length scale. Here only the lateral directions
are periodic; depth is a spectral-element mesh, on which the length scale may
vary and a layer has a top and a base.

**Depth is padded, as a radius is, and for the same reason**: ``A`` needs a
condition at each end of the mesh, and the padding keeps its mark out of the
domain. The mesh extends above the top, into negative depth under a free
surface, as below the base, and the condition is applied at its ends:
nothing is imposed at the top or the base of the layer itself.

**Laterally the field is periodic**, over the periods of the grid. By default
the domain is the whole period, and a point is read at its image in it. A
``planetmodel.fourier.PaddedDomain`` given as ``lateral=`` makes the domain a
bounded rectangle inside a wider period, the padding keeping it clear of its
periodic images: then :attr:`Slab.interior_mask` leaves the lateral padding
out as it does the padding in depth, and a point outside the bounds is
refused.

**The conventions are the package's.** A vector is the array of the field's
values on the padded grid, shape ``(depths, x, y)``; a product is left on the
grid, untruncated; a point is ``(depth, x, y)``, the layered coordinate first
as a radius is in a ball, with depth positive downward; the Fourier modes
are orthonormal over the period, so the Lebesgue inner product is the
integral over the volume; and point evaluation is refused at or below order
three halves.

**The pointwise variance of a prior is exact at the grid's points.** On the
grid a cosine and its sine have squares summing to a constant, so the
variance of a prior that is a function of ``A`` depends on depth alone, and a
calibration by ``pointwise_std=`` is exact there. Between the points the
same holds of every mode but those that are their own conjugate: the cosines
whose every wavenumber is zero or the Nyquist one of an axis of even size,
at most three, which the grid holds without a sine. With an even axis the
variance between the grid's points therefore dips below the calibrated value
by those modes' share, the smallest of the spectrum; a grid of odd sizes has
none, and is exact everywhere across.

The numerics are ``planetmodel``'s: ``SeparableOperatorFamily`` under the
plain measure and indexed by the lateral eigenvalue, one ``SpectralBasis``
per distinct ``|k|^2`` held in a ``planetmodel.cartesian.LayerBasis``
(reachable as :attr:`Slab.basis`), and the lateral transform of
``planetmodel.fourier`` (:attr:`Slab.transform`).
"""

from .space import Lebesgue, Slab, Sobolev

__all__ = ["Lebesgue", "Slab", "Sobolev"]
