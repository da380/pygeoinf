"""The spectral-element disk and annulus: the family, the space, its priors.

Needs ``planetmodel`` 1.3. The independent routes are the cylindrical Bessel
functions, against which the radial family of a disk is held order by order;
a finite-volume solve of the same radial eigenproblem on an annulus, which
shares nothing with the spectral elements but the equation; the grid against
the point, values synthesized on the grid by FFT against the same field
evaluated through the element polynomials and closed-form cosines; and
``planetmodel``'s banded solve against the eigenbasis.
"""

import numpy as np
import pytest

randomfield = pytest.importorskip("planetmodel.randomfield")
if not hasattr(randomfield, "SpectralBasis"):  # pragma: no cover
    pytest.skip("needs planetmodel 1.3 or later", allow_module_level=True)

from planetmodel.randomfield import SpectralBasis, padded_mesh  # noqa: E402
from scipy.special import jnp_zeros, jv  # noqa: E402

from pygeoinf import Traits, spectral_element  # noqa: E402
from pygeoinf import disk as disk_module  # noqa: E402
from pygeoinf.algebra.direct_sum import DirectSum  # noqa: E402
from pygeoinf.disk import Disk, Lebesgue, Sobolev  # noqa: E402
from pygeoinf.spectral_element._cylindrical import (  # noqa: E402
    CylindricalOperatorFamily,
    RingTransform,
    circular_harmonics,
)
from pygeoinf.testing import (  # noqa: E402
    check_coordinates,
    check_measure,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)


def first_zeros(m, count, /):
    """The first zeros of ``J_m'``, the wavenumbers of a unit disk's modes of
    order ``m`` under the natural condition; zero first at order zero, whose
    first mode is the constant."""
    zeros = jnp_zeros(m, count)
    if m == 0:
        zeros = np.concatenate(([0.0], zeros[: count - 1]))
    return zeros


class TestTheFamily:
    @pytest.mark.parametrize("m", [0, 3])
    def test_the_eigenpairs_of_a_disk_are_bessel_functions(self, m):
        """On the unit disk with ``L`` constant and the natural condition the
        modes of order ``m`` are ``J_m(k r)``, ``k`` the zeros of ``J_m'``,
        and their eigenvalues ``1 + L^2 k^2``; closed forms from scipy."""
        length = 0.3
        mesh = padded_mesh(0.0, 1.0, pad=0.0, weight="r2", ngll=5, drmax=0.05)
        family = CylindricalOperatorFamily(mesh, kappa=length**2)
        zeros = first_zeros(m, 8)
        theta, _ = family.eig(m, count=8)
        assert np.allclose(theta, 1.0 + length**2 * zeros**2, rtol=1e-6)
        basis = SpectralBasis(family, m, nmodes=8)
        radii = np.linspace(0.0, 1.0, 201)
        values = basis.evaluate(radii)
        for j in range(8):
            exact = jv(m, zeros[j] * radii)
            scaled = exact * (values[:, j] @ exact) / (exact @ exact)
            assert np.allclose(
                values[:, j], scaled, atol=1e-4 * np.abs(values[:, j]).max()
            )

    @pytest.mark.parametrize("m", [0, 3])
    def test_the_modes_are_orthonormal_under_r_dr(self, m):
        """``int phi_i phi_j r dr = delta_ij`` by the quadrature of the
        domain's elements, which a mesh without padding has to itself."""
        mesh = padded_mesh(0.0, 1.0, pad=0.0, weight="r2", ngll=5, drmax=0.1)
        family = CylindricalOperatorFamily(mesh, kappa=0.04)
        basis = SpectralBasis(family, m, nmodes=6)
        nodal = basis.synthesise(np.identity(6))
        gram = nodal.T @ (basis.weights()[:, None] * nodal)
        assert np.allclose(gram, np.identity(6), atol=1e-12)
        # And the weights are those of r dr: they integrate r to 1/3.
        assert basis.weights() @ basis.r == pytest.approx(1.0 / 3.0, rel=1e-12)

    def test_the_centre_is_condensed_at_order_zero_and_dropped_above(self):
        mesh = padded_mesh(0.0, 1.0, pad=0.0, weight="r2", ngll=5, drmax=0.1)
        family = CylindricalOperatorFamily(mesh, kappa=0.04)
        assert family.is_ball
        assert family.ndof(0) == family.ndof(2) == mesh.nglob - 1
        zero, two = (
            SpectralBasis(family, 0, nmodes=3),
            SpectralBasis(family, 2, nmodes=3),
        )
        assert np.all(two.modes()[0] == 0.0)
        assert np.all(np.abs(zero.modes()[0]) > 0.1)
        shell = CylindricalOperatorFamily(
            padded_mesh(0.5, 1.0, pad=0.0, weight="r2", ngll=5, drmax=0.1), kappa=0.04
        )
        assert not shell.is_ball and shell.ndof(2) == shell.mesh.nglob

    def test_a_mode_of_order_m_goes_like_r_to_the_m(self):
        mesh = padded_mesh(0.0, 1.0, pad=0.0, weight="r2", ngll=5, drmax=0.05)
        family = CylindricalOperatorFamily(mesh, kappa=0.09)
        near = np.array([0.01, 0.04])
        for m in (1, 2, 3):
            first = SpectralBasis(family, m, nmodes=1).evaluate(near)[:, 0]
            assert np.log(first[1] / first[0]) / np.log(4.0) == pytest.approx(
                m, abs=0.01
            )

    def test_the_banded_solve_inverts_the_operator(self, rng):
        mesh = padded_mesh(0.3, 1.0, pad=(0.3, 0.4), weight="r2", ngll=5, drmax=0.1)
        family = CylindricalOperatorFamily(mesh, kappa=0.04, robin=(None, 3.0))
        for m in (0, 4):
            v = rng.normal(size=family.ndof(m))
            assert np.allclose(family.apply(m, family.solve(m, v)), v)
            assert np.allclose(family.apply_power(m, -1.0, v), family.solve(m, v))

    def test_bad_meshes_and_coefficients_are_refused(self):
        from planetmodel.mesh1d import Mesh1D

        mesh = padded_mesh(0.0, 1.0, pad=0.0, weight="r2", ngll=5, drmax=0.1)
        with pytest.raises(TypeError, match="Mesh1D"):
            CylindricalOperatorFamily(np.linspace(0.0, 1.0, 5))
        with pytest.raises(ValueError, match="r >= 0"):
            CylindricalOperatorFamily(Mesh1D([-1.0, 1.0], ngll=5))
        with pytest.raises(ValueError, match="kappa"):
            CylindricalOperatorFamily(mesh, kappa=-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            CylindricalOperatorFamily(mesh, robin=-1.0)


class TestTheRingTransform:
    @pytest.mark.parametrize("mmax", [0, 1, 6, 31])
    def test_synthesis_is_the_sum_of_the_modes(self, mmax, rng):
        transform = RingTransform(mmax)
        coefficients = rng.normal(size=(2, mmax + 1, 5))
        coefficients[1, 0] = 0.0
        values = transform.synthesise(coefficients)
        assert values.shape == (5, 2 * mmax + 1)
        modes = circular_harmonics(mmax, transform.angles)
        direct = np.einsum("smr,smj->rj", coefficients, modes)
        assert np.allclose(values, direct, atol=1e-12)
        assert np.allclose(transform.analyse(values), coefficients, atol=1e-12)

    @pytest.mark.parametrize("mmax", [0, 1, 6, 31])
    def test_analysis_is_the_adjoint_of_synthesis_under_the_ring_rule(self, mmax, rng):
        """``(S c, v)_ring == (c, A v)`` with ``(u, v)_ring = w sum u v``, for
        a ``v`` that is not in the span."""
        transform = RingTransform(mmax)
        c = rng.normal(size=(2, mmax + 1, 4))
        c[1, 0] = 0.0
        v = rng.normal(size=(4, 2 * mmax + 1))
        left = transform.weight * np.sum(transform.synthesise(c) * v)
        right = np.sum(c * transform.analyse(v))
        assert left == pytest.approx(right, rel=1e-12)

    def test_the_modes_are_orthonormal_on_the_unit_circle(self):
        """By a fine trapezoid rule, independent of the transform's own."""
        angles = 2.0 * np.pi * np.arange(400) / 400
        modes = circular_harmonics(5, angles).reshape(12, -1)
        kept = np.delete(modes, 6, axis=0)  # the empty sine slot of order zero
        gram = (2.0 * np.pi / 400) * kept @ kept.T
        assert np.allclose(gram, np.identity(11), atol=1e-12)

    def test_the_grid_and_the_shapes(self):
        transform = RingTransform(3)
        assert np.allclose(transform.angles, 2.0 * np.pi * np.arange(7) / 7)
        assert transform.weight == pytest.approx(2.0 * np.pi / 7)
        with pytest.raises(ValueError, match="shape"):
            transform.analyse(np.zeros((2, 6)))
        with pytest.raises(ValueError, match="shape"):
            transform.synthesise(np.zeros((2, 3, 2)))
        with pytest.raises(ValueError, match="non-negative"):
            RingTransform(-1)


# ------------------------------------------------------------------------- #
#                                The space                                  #
# ------------------------------------------------------------------------- #


def disk(order, /, **options):
    options = {"mmax": 6, "length_scale": 0.3, "element_length": 0.3, **options}
    return Disk(options.pop("mmax"), order=order, **options).with_order(order)


def annulus(order, /, **options):
    options = {
        "inner_radius": 0.5,
        "length_scale": 0.2,
        "element_length": 0.2,
        **options,
    }
    return disk(order, **options)


GEOMETRIES = [disk, annulus]


def grid_points(space, /, *, stride=3):
    """Some interior grid points, and where they sit in a grid array."""
    points, where = [], []
    for i in range(0, space.interior_radii.size, stride):
        for j in range(space.angles.size):
            points.append((space.interior_radii[i], space.angles[j]))
            where.append((i, j))
    return points, tuple(np.array(where).T)


def cartesian(function, /):
    """A function of ``(x, y)`` as one of a point ``(radius, angle)``."""

    def of_a_point(point):
        radius, angle = point
        t = np.radians(angle)
        return function(radius * np.cos(t), radius * np.sin(t))

    return of_a_point


def area_quadrature(space, /):
    """The weights of ``int f r dr dt`` over the domain's part of the grid:
    the mass of the domain's own radial elements times the ring's rule."""
    return space.basis[0].weights()[:, None] * space.grid.weight


class TestTheSpace:
    @pytest.mark.parametrize("build", GEOMETRIES)
    @pytest.mark.parametrize("order", [0.0, 2.0])
    def test_the_axioms(self, build, order, rng):
        space = build(order)
        check_space(space, rng=rng, rebuild=lambda: build(order))
        check_coordinates(space, rng=rng)

    def test_with_order_names_the_subclass_and_shares_the_basis(self):
        sobolev = disk(2.0)
        lebesgue = sobolev.with_order(0.0)
        assert type(sobolev) is Sobolev and type(lebesgue) is Lebesgue
        assert lebesgue.basis is sobolev.basis
        assert lebesgue == Lebesgue(6, length_scale=0.3, element_length=0.3)
        assert sobolev == Sobolev(6, 2.0, 0.3, element_length=0.3)
        assert sobolev.shares_vectors_with(lebesgue)
        assert not sobolev.shares_vectors_with(disk(2.0, mmax=5))
        assert disk(2.0) != annulus(2.0)
        assert repr(sobolev).startswith("Sobolev(mmax=6")

    def test_a_disk_is_padded_outwards_only_and_an_annulus_to_the_centre(self):
        assert disk(0.0).padding == pytest.approx((0.0, 0.6))
        assert disk(0.0, boundary=None).padding == pytest.approx((0.0, 1.2))
        assert disk(0.0).radii[0] == 0.0
        ring = annulus(0.0, padding=2.0)
        assert ring.padding == pytest.approx((0.5, 2.0))
        assert ring.radii[0] == 0.0
        assert ring.interior_radii[0] == 0.5 and ring.interior_radii[-1] == 1.0

    def test_the_grid(self):
        space = disk(0.0)
        assert space.grid_shape == (space.radii.size, 13)
        assert np.allclose(space.angles, 360.0 * np.arange(13) / 13)
        assert space.interior_mask.shape == space.grid_shape
        x = space.zero()
        assert space.interior_values(x).shape == (space.interior_radii.size, 13)
        assert space.spatial_dimension == 2
        assert space.domain_volume == pytest.approx(np.pi)
        assert annulus(0.0).domain_volume == pytest.approx(0.75 * np.pi)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_labels_are_the_packing(self, build):
        space = build(0.0)
        basis = space.basis
        for m in range(space.mmax + 1):
            for signed in (m, -m) if m else (0,):
                block = basis.block(int(signed < 0), m)
                assert np.all(space.orders[block] == signed)
                assert np.array_equal(
                    space.radial_indices[block], np.arange(basis.nmodes[m])
                )
                assert np.array_equal(space.eigenvalues[block], basis[m].theta)
        assert space.orders.size == space.dim
        assert space.dim == basis.nmodes[0] + 2 * basis.nmodes[1:].sum()
        with pytest.raises(IndexError):
            basis.block(1, 0)

    def test_the_default_truncation_is_isotropic_on_the_domain(self):
        """Every mode of the padded mesh below the first of order ``mmax`` on
        the domain *without* its padding, and no other: so the shortest
        wavelength kept is the angular one at the outer radius, whatever the
        padding is."""
        space = disk(0.0, mmax=8, boundary=None)
        bare = disk(0.0, mmax=8, padding=0.0, boundary=None)
        bound = bare.basis.family.eigvalsh(8)[0]
        family = space.basis.family
        for m in range(9):
            assert space.basis.nmodes[m] == np.sum(family.eigvalsh(m) <= bound * 1.0001)
        assert np.all(np.diff(space.basis.nmodes) <= 0)
        assert space.basis.nmodes[8] > 1
        shortest = 2.0 * np.pi * 0.3 / np.sqrt(space.eigenvalues.max() - 1.0)
        assert shortest == pytest.approx(2.0 * np.pi / 8, rel=0.25)
        wider = disk(0.0, mmax=8, padding=2.0, boundary=None)
        assert wider.eigenvalues.max() == pytest.approx(
            space.eigenvalues.max(), rel=0.05
        )
        assert wider.dim > space.dim

    def test_the_truncation_can_be_stated(self):
        assert np.all(disk(0.0, radial_modes=4).basis.nmodes == 4)
        assert disk(0.0, radial_modes=4).dim == 4 * 13
        bounded = disk(0.0, max_eigenvalue=30.0)
        assert bounded.eigenvalues.max() <= 30.0
        with pytest.raises(ValueError, match="not both"):
            disk(0.0, radial_modes=4, max_eigenvalue=30.0)
        with pytest.raises(ValueError, match="without a radial mode"):
            disk(0.0, max_eigenvalue=1.5)
        with pytest.raises(ValueError, match="mmax"):
            Disk(-1)
        with pytest.raises(ValueError, match="radii"):
            Disk(3, inner_radius=1.0)

    def test_the_default_element_length_resolves_the_modes_kept(self):
        """Without one given, a mesh fine enough that the modes kept are the
        lower part of what it holds: the largest eigenvalue kept is close to
        what a much finer mesh gives for the same mode."""
        space = Disk(12, length_scale=0.3)
        fine = Disk(12, length_scale=0.3, element_length=0.02)
        assert space.basis.nmodes.tolist() == fine.basis.nmodes.tolist()
        assert np.allclose(space.eigenvalues, fine.eigenvalues, rtol=1e-3)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_lebesgue_inner_product_is_the_area_quadrature(self, build, rng):
        """``int u v r dr dt`` over the padded mesh, summed on the grid with
        its own weights, for fields in the span -- whose product the grid
        integrates exactly."""
        space = build(0.0)
        u, v = space.random(rng=rng), space.random(rng=rng)
        weights = space.basis.family.element_mass()
        radial = np.zeros(space.radii.size)
        np.add.at(radial, space.basis.family.mesh.gmap, weights)
        quadrature = np.sum(radial[:, None] * space.grid.weight * u * v)
        assert space.inner_product(u, v) == pytest.approx(quadrature, rel=1e-10)

    def test_white_noise_components_carry_the_inverse_metric(self, rng):
        """``N(0, G^-1)``, stated on the components, where it costs nothing."""
        space = disk(2.0)
        draws = np.stack([space.white_noise_components(rng=rng) for _ in range(20000)])
        assert np.allclose(draws.var(axis=0) * space.metric_values, 1.0, atol=0.06)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_white_noise_has_identity_covariance(self, build, rng):
        check_white_noise(build(2.0, mmax=4), rng=rng, samples=6000, rtol=0.14)


class TestBoundary:
    def test_robin_halves_the_padding_and_carries_the_planes_curvature(self):
        natural = disk(0.0, mmax=8, max_eigenvalue=60.0, boundary=None)
        matched = disk(0.0, mmax=8, max_eigenvalue=60.0, boundary="robin")
        assert matched.padding == pytest.approx((0.0, 0.6))
        # Half the ball's curvature: K_0(r / L) goes like exp(-r / L) / sqrt(r).
        assert matched.robin == pytest.approx((0.0, 0.7 / 0.3 + 0.5 / 1.6))
        assert matched.dim < natural.dim and matched.radii.size < natural.radii.size
        assert matched != natural
        # The inner end of this mesh is at r = 0.1, where the curvature term
        # outweighs the other: held at zero, the natural condition.
        assert annulus(0.0, boundary="robin").robin[0] == 0.0
        thin = annulus(0.0, length_scale=0.05, boundary="robin")
        assert thin.robin[0] == pytest.approx(0.7 / 0.05 - 0.5 / 0.4)
        assert np.all(thin.basis.nmodes >= 1)
        assert disk(0.0, boundary=(None, 2.0)).robin == (0.0, 2.0)
        with pytest.raises(ValueError, match="boundary"):
            disk(0.0, boundary="dirichlet")

    @staticmethod
    def _variance(build, padding, boundary):
        space = build(
            0.0,
            mmax=8,
            length_scale=0.2,
            padding=padding,
            boundary=boundary,
            max_eigenvalue=160.0,
            element_length=0.05,
        )
        field = space.pointwise_variance(space.eigenvalues**-3.0)
        return space.interior_values(field)[:, 0]

    def test_two_length_scales_then_do_what_four_do_without_it(self):
        """On a disk, whose one end is the outer: measured, the error of two
        length scales under the matched condition is a quarter of that of
        four under the natural one."""
        reference = self._variance(disk, 2.0, None)

        def error(padding, boundary):
            return np.abs(
                self._variance(disk, padding, boundary) / reference - 1.0
            ).max()

        assert error(0.4, "robin") < 5e-3
        assert error(0.4, None) > 50.0 * error(0.4, "robin")
        assert error(0.4, "robin") < 0.5 * error(0.8, None)

    def test_an_inner_end_near_the_centre_is_held_natural(self):
        """An annulus of inner radius 0.5 padded by two length scales of 0.2
        ends its mesh at r = 0.1, half a length scale from the centre, where
        the matched coefficient ``0.7 / L - 1 / 2 r`` is negative and held at
        zero. Measured: the error there is about twice that of four length
        scales under the natural condition, and a tenth of two."""
        reference = self._variance(annulus, 2.0, None)

        def error(padding, boundary):
            return np.abs(
                self._variance(annulus, padding, boundary) / reference - 1.0
            ).max()

        assert error(0.4, "robin") < 2.5e-2
        assert error(0.4, None) > 8.0 * error(0.4, "robin")


class TestPointEvaluation:
    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_every_basis_function_is_its_own_grid_values(self, build):
        """One component at a time, so that an error in the order of the
        blocks cannot hide in a sum: closed-form cosines and the element
        polynomials against the FFT and the nodal values."""
        space = build(2.0)
        points, where = grid_points(space)
        rows = space.basis_matrix(points)
        for k in range(space.dim):
            unit = np.zeros(space.dim)
            unit[k] = 1.0
            on_grid = space.interior_values(space.from_components(unit))[where]
            assert np.allclose(
                rows[:, k], on_grid, atol=1e-10 * max(1.0, np.abs(on_grid).max())
            )

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_a_field_in_the_span_is_evaluated_exactly(self, build, rng):
        space = build(2.0)
        x = space.random(rng=rng)
        points, where = grid_points(space)
        values = space.interior_values(x)[where]
        assert np.allclose(
            space.evaluate(x, points), values, atol=1e-10 * np.abs(values).max()
        )
        assert space.basis_at(points[4]) == pytest.approx(space.basis_matrix(points)[4])

    def test_the_radial_modes_are_bessel_functions(self):
        """With ``L`` constant the modes of order ``m`` are ``J_m(k r)``, ``k``
        read off the eigenvalue -- whatever the padding and the matched
        condition do to which ``k`` they are."""
        space = disk(0.0, mmax=8, radial_modes=8, element_length=0.05)
        radii = np.linspace(0.0, 1.0, 201)
        for m in (0, 1, 2, 4, 8):
            radial = space.basis[m]
            values = radial.evaluate(radii)
            for j in (1, 4, 7):
                k = np.sqrt(radial.theta[j] - 1.0) / 0.3
                exact = jv(m, k * radii)
                scaled = exact * (values[:, j] @ exact) / (exact @ exact)
                assert np.allclose(
                    values[:, j], scaled, atol=2e-4 * np.abs(values[:, j]).max()
                )

    def test_the_angle_is_periodic_in_degrees(self, rng):
        space = disk(2.0)
        x = space.random(rng=rng)
        assert space.evaluate(x, [(0.6, -150.0)]) == pytest.approx(
            space.evaluate(x, [(0.6, 210.0)])
        )

    def test_the_angle_runs_anticlockwise_from_the_x_axis(self):
        """``y`` itself, projected, is ``r sin(angle)``: so the angle is
        measured from the ``x`` axis towards the ``y`` axis."""
        space = disk(2.0, radial_modes=10, element_length=0.1)
        field = space.project_function(cartesian(lambda x, y: y))
        assert space.evaluate(field, [(0.5, 90.0), (0.5, 0.0), (0.5, 210.0)]) == (
            pytest.approx([0.5, 0.0, -0.25], abs=1e-7)
        )

    def test_the_centre_has_one_value(self, rng):
        """Only order zero reaches the centre, so the angle there is idle."""
        space = disk(2.0)
        x = space.random(rng=rng)
        values = space.evaluate(x, [(0.0, 10.0), (0.0, 200.0)])
        assert values[0] == pytest.approx(values[1], abs=1e-12)

    def test_it_is_refused_at_and_below_order_one(self):
        for order in (0.0, 1.0):
            with pytest.raises(ValueError, match="above 1"):
                disk(order).dirac((0.5, 0.0))
            with pytest.raises(ValueError, match="above 1"):
                disk(order).point_evaluation_operator([(0.5, 0.0)])
        disk(0.0).dirac((0.5, 0.0), unsafe=True)
        disk(1.25).dirac((0.5, 0.0))

    def test_a_radius_outside_the_domain_is_refused(self):
        with pytest.raises(ValueError):
            disk(2.0).dirac((1.2, 0.0))
        with pytest.raises(ValueError):
            annulus(2.0).dirac((0.3, 0.0))

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_operator_and_its_adjoint(self, build, rng):
        space = build(2.0)
        points = [space.random_point(rng=rng) for _ in range(5)]
        operator = space.point_evaluation_operator(points)
        check_operator(operator, rng=rng)
        x = space.random(rng=rng)
        assert np.allclose(operator(x), space.evaluate(x, points))
        assert operator(x)[1] == pytest.approx(space.dirac(points[1])(x))

    def test_random_points_lie_in_the_domain_uniformly_by_area(self, rng):
        space = annulus(0.0)
        points = np.array([space.random_point(rng=rng) for _ in range(4000)])
        assert np.all((points[:, 0] >= 0.5) & (points[:, 0] <= 1.0))
        assert np.all((points[:, 1] >= 0.0) & (points[:, 1] < 360.0))
        # Uniform by area: the mean of r^2 is the midpoint of 0.25 and 1.
        assert np.mean(points[:, 0] ** 2) == pytest.approx(0.625, abs=0.01)


class TestProjection:
    @staticmethod
    def _truncation_error(space, function, extension):
        """Relative ``L2`` error over the domain, and the largest error by
        radius, of a sampled function once settled into the span, against the
        function's own values at the domain's grid points."""
        sampled = space.project_function(function, extension=extension)
        weights = area_quadrature(space)
        exact = space.interior_values(
            space.project_function(function, extension="constant")
        )
        missed = space.interior_values(space.truncate(sampled)) - exact
        norm = np.sqrt(np.sum(weights * missed**2) / np.sum(weights * exact**2))
        return norm, np.abs(missed).max(axis=1)

    #: A polynomial in x and y, regular at the centre by construction: its
    #: part of order m goes like r^m. Of order three at most.
    _polynomial = staticmethod(
        cartesian(lambda x, y: 1.0 + 0.5 * x - 0.3 * y**2 + 0.8 * x * y + 0.4 * x**3)
    )

    @staticmethod
    def _wavy(point):
        radius, angle = point
        return np.cos(3.0 * radius) * (1.0 + radius * np.sin(np.radians(angle)))

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_a_fit_over_the_domain_is_right_everywhere(self, build):
        """For a function regular at the centre, the centre included."""
        space = build(0.0, mmax=4, radial_modes=16, element_length=0.05)
        fitted, by_radius = self._truncation_error(space, self._polynomial, "fit")
        mirrored, _ = self._truncation_error(space, self._polynomial, "odd")
        assert fitted < 1e-8 and fitted < 1e-3 * mirrored
        assert by_radius.max() < 1e-6
        field = space.project_function(self._polynomial, extension="fit")
        assert np.allclose(space.truncate(field), field, atol=1e-10)
        points = [(0.0, 0.0), (0.7, 33.0), (1.0, 250.0)][int(build is annulus) :]
        assert np.allclose(
            space.evaluate(field, points),
            [self._polynomial(p) for p in points],
            atol=1e-7,
        )

    def test_more_radial_modes_bring_a_sampled_function_closer(self):
        """In the norm of the domain, which is the claim that holds; slowly,
        since a continued function has a kink, or the mesh's end condition
        unmet, somewhere on the padding."""
        errors = [
            self._truncation_error(
                annulus(
                    0.0,
                    mmax=4,
                    radial_modes=modes,
                    element_length=0.05,
                    boundary=None,
                ),
                self._wavy,
                "odd",
            )[0]
            for modes in (4, 8, 16, 32)
        ]
        assert all(a > b for a, b in zip(errors, errors[1:]))
        assert errors[-1] < 3e-3

    def test_the_reflection_mends_the_ends_of_an_annulus(self):
        """Held constant across the padding a function has a kink at each end
        of the domain, and the error sits there; reflected, it has none.
        Measured under the natural condition: the error at both ends falls by
        a factor of seven or more, and in the norm of the domain by a tenth."""
        space = annulus(
            0.0, mmax=4, radial_modes=32, element_length=0.05, boundary=None
        )
        held, by_radius_held = self._truncation_error(space, self._wavy, "constant")
        mirrored, by_radius = self._truncation_error(space, self._wavy, "odd")
        assert mirrored < held
        assert by_radius[0] < 0.2 * by_radius_held[0]
        assert by_radius[-1] < 0.2 * by_radius_held[-1]

    def test_an_extension_is_named(self):
        space = disk(0.0, mmax=2)
        with pytest.raises(ValueError, match="odd"):
            space.project_function(lambda p: 1.0, extension="even")
        positive = lambda p: 1.0 + 5.0 * (1.0 - p[0])  # noqa: E731
        assert np.any(space.project_function(positive, extension="odd") < 0.0)
        assert np.all(space.project_function(positive, extension="constant") > 0.0)


class TestPointwiseAlgebra:
    def test_a_product_is_left_on_the_grid(self, rng):
        space = disk(2.0)
        x, y = space.random(rng=rng), space.random(rng=rng)
        assert np.array_equal(space.multiply(x, y), x * y)
        assert not np.allclose(space.truncate(x * y), x * y)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_multiplication_is_self_adjoint_on_lebesgue_only(self, build, rng):
        lebesgue = build(0.0)
        f = lebesgue.project_function(cartesian(lambda x, y: 1.0 + 0.3 * x))
        on_lebesgue = lebesgue.multiplication_operator(f)
        assert Traits.SELF_ADJOINT in on_lebesgue.traits
        check_operator(on_lebesgue, rng=rng)
        check_traits(on_lebesgue, rng=rng)
        on_sobolev = build(2.0).multiplication_operator(f)
        assert Traits.SELF_ADJOINT not in on_sobolev.traits
        check_operator(on_sobolev, rng=rng)

    def test_the_support_projection_zeroes_the_padding(self, rng):
        space = disk(0.0)
        projection = space.support_projection()
        check_operator(projection, rng=rng)
        x = space.random(rng=rng)
        assert np.all(projection(x)[~space.interior_mask] == 0.0)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_integral_runs_over_the_domain_alone(self, build, rng):
        space = build(0.0)
        x = space.random(rng=rng)
        quadrature = np.sum(area_quadrature(space) * space.interior_values(x))
        assert space.integral_functional()(x) == pytest.approx(quadrature, rel=1e-10)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_integral_of_one_is_the_area_when_nothing_is_dropped(self, build):
        """With every radial mode kept the constant is in the span, and the
        quadrature of the domain's elements integrates it exactly. The centre
        of a disk is not a mode's own node at any order, so all is one fewer
        there; the annulus's padding stops short of the centre."""
        options = {"mmax": 2, "element_length": 0.25, "padding": 0.3}
        nodes = build(0.0, **options).radii.size
        full = build(0.0, radial_modes=nodes - int(build is disk), **options)
        assert full.integral_functional()(np.ones(full.grid_shape)) == pytest.approx(
            full.domain_volume, rel=1e-10
        )

    def test_the_integral_of_a_polynomial_is_its_closed_form(self):
        """``int (1 + x^2 + 3 x y) dA`` over the unit disk is ``pi + pi / 4``."""
        space = disk(0.0, mmax=4, radial_modes=10, element_length=0.1)
        field = space.project_function(cartesian(lambda x, y: 1.0 + x**2 + 3 * x * y))
        assert space.integral_functional()(field) == pytest.approx(
            1.25 * np.pi, rel=1e-9
        )


class TestMeasures:
    def test_the_sobolev_measure_is_the_banded_solve(self, rng):
        """Covariance ``A^-1`` on the Lebesgue space of an annulus with every
        radial mode kept, against a Cholesky solve of each order's pencil
        that never forms an eigenvector."""
        ring = annulus(0.0, mmax=3, element_length=0.25, padding=0.3)
        space = annulus(
            0.0, mmax=3, element_length=0.25, padding=0.3, radial_modes=ring.radii.size
        )
        family = space.basis.family
        # Order one, so that the covariance is one solve with the family; a
        # measure of that order is not one the space admits, hence unsafe.
        covariance = space.sobolev_measure(1.0, unsafe=True).covariance
        x = space.random(rng=rng)
        got = space.basis.synthesise(space.to_components(covariance(x)), physical=False)
        given = space.basis.synthesise(space.to_components(x), physical=False)
        for m in range(4):
            for part in (0, 1) if m else (0,):
                assert np.allclose(
                    got[part, m], family.solve(m, given[part, m]), atol=1e-9
                )

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_pointwise_variance_is_the_sum_of_squared_rows(self, build, rng):
        """At grid points, against ``basis_matrix``: once for a function of
        ``A``, where the angular sum is done in closed form, and once for
        variances that depend on the mode, where it cannot be."""
        space = build(2.0)
        points, where = grid_points(space)
        rows = space.basis_matrix(points)
        isotropic = space.eigenvalues**-1.5
        ragged = isotropic * rng.uniform(0.5, 1.5, size=space.dim)
        for variances in (isotropic, ragged):
            field = space.interior_values(space.pointwise_variance(variances))
            direct = (rows**2) @ (variances / space.metric_values)
            assert np.allclose(field[where], direct, rtol=1e-10)

    def test_the_variance_of_a_function_of_the_operator_is_radial(self):
        space = disk(0.0)
        field = space.pointwise_variance(space.eigenvalues**-2.0)
        assert np.allclose(field, field[:, :1])

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_pointwise_std_is_what_the_covariance_says(self, build):
        """``diag(E C E*)`` through the operator algebra, against a field
        even in ``r``, so regular at the centre."""
        space = build(0.0, radial_modes=12, element_length=0.1)
        target = 0.3 + 0.1 * space.radii[:, None] ** 2 + space.zero()
        measure = space.sobolev_measure(2.0, pointwise_std=target)
        radii = np.linspace(space.inner_radius, space.radius, 5)
        evaluation = space.point_evaluation_operator(
            [(r, 60.0) for r in radii], unsafe=True
        )
        matrix = (evaluation @ measure.covariance @ evaluation.adjoint).matrix(
            form="components"
        )
        assert np.allclose(np.sqrt(np.diag(matrix)), 0.3 + 0.1 * radii**2, rtol=5e-3)

    def test_the_diagonal_measure_has_a_precision(self):
        space = disk(2.0)
        assert space.sobolev_measure(2.0).precision is not None
        with pytest.raises(ValueError, match="positive"):
            space.sobolev_measure(2.0, pointwise_std=-1.0)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_moments_match_the_samples(self, build, rng):
        space = build(2.0, mmax=4, element_length=0.4)
        check_measure(
            space.sobolev_measure(2.0, amplitude=0.7), rng=rng, samples=4000, rtol=0.18
        )
        check_measure(
            space.sobolev_measure(2.0, pointwise_std=0.3),
            rng=rng,
            samples=4000,
            rtol=0.18,
        )

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_sampled_variances_are_the_stated_pointwise_variance(self, build, rng):
        """A Sobolev measure and a heat one, ``exp(-t (A - 1))``, each drawn
        two thousand times at once and the variance at every grid point held
        to what :meth:`pointwise_variance` states, to its standard error."""
        space = build(1.0, mmax=4, element_length=0.2)
        t = 0.02
        spectra = {
            "sobolev": 0.5 * space.eigenvalues**-2.0,
            "heat": np.exp(-t * (space.eigenvalues - 1.0)),
        }
        for variances in spectra.values():
            measure = space.spectral_measure(variances)
            stated = space.interior_values(space.pointwise_variance(variances))
            draws = np.stack(
                [space.interior_values(measure.sample(rng=rng)) for _ in range(2000)]
            )
            sampled = np.mean(draws**2, axis=0)
            # The relative standard error of a variance from n draws is
            # sqrt(2 / n), 0.03 at n = 2000; five of them is a safe bound at
            # every point at once.
            assert np.allclose(sampled, stated, rtol=0.16)


# ------------------------------------------------------------------------- #
#                              Layered disks                                #
# ------------------------------------------------------------------------- #

RADII = [0.0, 0.4, 1.0]
# A field that jumps at the interface, a polynomial in x and y in each layer.
PIECES = [
    cartesian(lambda x, y: 3.0 - x**2 - y**2 + 0.5 * x * y),
    cartesian(lambda x, y: 1.0 + 0.2 * (x**2 + y**2) - 0.3 * y),
]


def rings(order=2.0, /, **options):
    options = {
        "length_scale": [0.15, 0.25],
        "radial_modes": [12, 10],
        "element_length": 0.1,
        **options,
    }
    return disk_module.Layered(RADII, 4, order=order, **options)


class TestLayered:
    def test_the_axioms(self, rng):
        space = rings()
        check_space(space, rng=rng, rebuild=rings)
        check_coordinates(space, rng=rng)
        check_white_noise(space, rng=rng, samples=6000, rtol=0.14)

    def test_it_is_the_direct_sum_of_its_layers(self, rng):
        space = rings()
        plain = DirectSum(list(space.layers))
        assert space == plain and plain == space
        assert hash(space) == hash(plain)
        x, y = space.random(rng=rng), space.random(rng=rng)
        assert space.inner_product(x, y) == pytest.approx(plain.inner_product(x, y))
        assert isinstance(space, spectral_element.Layered)
        assert type(space.with_order(0.0)) is type(space)
        handed = disk_module.Layered.from_layers(space.layers, labels=space.labels)
        assert type(handed) is disk_module.Layered and handed == space

    def test_the_structure(self):
        space = rings()
        assert space.geometry is Disk and len(space) == 2
        assert space.layer_bounds == ((0.0, 0.4), (0.4, 1.0))
        assert space.interfaces == (0.4,)
        assert space.domain_volume == pytest.approx(np.pi)
        assert space.spatial_dimension == 2
        assert [layer.basis.nmodes.tolist() for layer in space.layers] == [
            [12] * 5,
            [10] * 5,
        ]
        assert space.layer_index((0.7, 10.0)) == 1
        assert space.layer_index((0.4, 10.0), side="below") == 0
        with pytest.raises(ValueError, match="interface"):
            space.layer_index((0.4, 10.0))

    def test_a_function_that_jumps_is_given_piece_by_piece(self):
        space = rings()
        field = space.project_function(PIECES)
        points = [(0.1, 30.0), (0.3, 200.0), (0.55, 10.0), (0.95, 300.0)]
        exact = [PIECES[space.layer_index(p)](p) for p in points]
        assert np.allclose(space.evaluate(field, points), exact, atol=1e-6)
        for side, piece in (("below", PIECES[0]), ("above", PIECES[1])):
            assert space.evaluate(field, [(0.4, 75.0)], side=side)[0] == (
                pytest.approx(piece((0.4, 75.0)), abs=1e-6)
            )

    def test_the_jump_is_taken_at_angles(self, rng):
        space = rings()
        angles = [0.0, 75.0, 230.0]
        jump = space.jump_operator(0, directions=angles)
        check_operator(jump, rng=rng)
        x = space.random(rng=rng)
        points = [(0.4, angle) for angle in angles]
        assert np.allclose(
            jump(x),
            space.evaluate(x, points, side="above")
            - space.evaluate(x, points, side="below"),
        )
        field = space.project_function(PIECES)
        assert np.allclose(
            jump(field),
            [PIECES[1](p) - PIECES[0](p) for p in points],
            atol=1e-6,
        )
        with pytest.raises(ValueError, match="directions"):
            space.jump_operator(0)
        with pytest.raises(IndexError):
            space.jump_operator(1, directions=angles)
        with pytest.raises(ValueError, match="above 1"):
            rings(0.0).jump_operator(0, directions=angles)

    def test_conditioning_on_no_jump_makes_the_medium_continuous_there(self, rng):
        space = rings()
        prior = space.sobolev_measure(2.0, pointwise_std=0.3)
        angles = list(np.linspace(0.0, 360.0, 9, endpoint=False))
        welded = prior.condition(space.jump_operator(0, directions=angles), np.zeros(9))
        draws = [welded.sample(rng=rng) for _ in range(20)]
        jumps = np.array([space.jump_operator(0, directions=angles)(x) for x in draws])
        assert np.abs(jumps).max() < 1e-8

    def test_the_integral_is_over_the_whole_medium(self):
        """``int (3 - r^2) dA`` over the inner disk and ``int (1 + 0.2 r^2)
        dA`` over the ring, the parts odd in an angle integrating to zero."""
        space = rings()
        field = space.project_function(PIECES)
        exact = (
            2.0
            * np.pi
            * (
                (3.0 * 0.4**2 / 2.0 - 0.4**4 / 4.0)
                + ((1.0**2 - 0.4**2) / 2.0 + 0.2 * (1.0**4 - 0.4**4) / 4.0)
            )
        )
        assert space.integral_functional()(field) == pytest.approx(exact, rel=1e-8)

    def test_the_prior_lives_on_the_space_with_independent_layers(self, rng):
        space = rings()
        measure = space.sobolev_measure([2.0, 2.5], pointwise_std=[0.2, 0.4])
        assert measure.domain == space
        check_measure(measure, rng=rng, samples=3000, rtol=0.2)
        a, b = (0.2, 40.0), (0.7, 40.0)
        look = space.point_evaluation_operator([a, b])
        covariance = (look @ measure.covariance @ look.adjoint).matrix(
            form="components"
        )
        assert covariance[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(np.sqrt(np.diag(covariance)), [0.2, 0.4], rtol=2e-2)


# ------------------------------------------------------------------------- #
#                                 Plotting                                  #
# ------------------------------------------------------------------------- #


class TestPlotting:
    """The pictures are checked through their numbers: the values handed to
    matplotlib against the space's own point evaluation."""

    @pytest.fixture(autouse=True)
    def _headless(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        yield
        pyplot.close("all")

    @staticmethod
    def _drawn(mappable):
        """The coordinates and values of a gouraud mesh, as flat arrays."""
        where = np.asarray(mappable.get_coordinates()).reshape(-1, 2)
        return where, np.asarray(mappable.get_array()).ravel()

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_what_is_drawn_is_the_field_at_the_points_drawn(self, build, rng):
        from pygeoinf import plotting

        space = build(2.0)
        x = space.random(rng=rng)
        ax, mappable = plotting.plot(space, x, angles=24, symmetric=True)
        where, values = self._drawn(mappable)
        radii = np.hypot(where[:, 0], where[:, 1])
        angles = np.degrees(np.arctan2(where[:, 1], where[:, 0]))
        assert np.allclose(np.unique(radii.round(12)), space.interior_radii)
        picked = slice(None, None, 7)
        points = list(zip(radii[picked], angles[picked]))
        assert np.allclose(values[picked], space.evaluate(x, points), atol=1e-10)
        low, high = mappable.get_clim()
        assert low == pytest.approx(-high)
        assert ax.get_aspect() == 1.0
        assert len(ax.lines) == (2 if space.inner_radius > 0.0 else 1)

    def test_the_padding_is_drawn_on_request(self, rng):
        from pygeoinf import plotting

        space = disk(2.0)
        _, mappable = plotting.plot(space, space.random(rng=rng), padding=True)
        where, _ = self._drawn(mappable)
        assert np.hypot(where[:, 0], where[:, 1]).max() == pytest.approx(
            space.radii[-1]
        )
        with pytest.raises(ValueError, match="shape"):
            plotting.plot(space, np.zeros(4))

    def test_points_go_where_they_are_in_the_plane(self, rng):
        from pygeoinf import plotting

        space = annulus(2.0)
        stations = [(1.0, 0.0), (0.7, 90.0), (0.5, 225.0)]
        ax, collection = plotting.plot_points(
            space, stations, data=[1.0, 2.0, 3.0], colorbar_label="value"
        )
        assert np.allclose(
            collection.get_offsets(),
            [[1.0, 0.0], [0.0, 0.7], [-0.5 / np.sqrt(2.0), -0.5 / np.sqrt(2.0)]],
        )
        with pytest.raises(ValueError, match="values"):
            plotting.plot_points(space, stations, data=[1.0])

    def test_a_path_is_a_polyline_and_a_pair_is_a_chord(self):
        from pygeoinf import plotting

        space = disk(2.0)
        chord = [(1.0, 180.0), (1.0, 0.0)]
        traced = [(1.0, 90.0), (0.3, 120.0), (0.3, 240.0), (1.0, 270.0)]
        ax, collection = plotting.plot_paths(space, [chord, traced], outline=True)
        first, second = collection.get_segments()
        assert np.allclose(first, [[-1.0, 0.0], [1.0, 0.0]])
        assert second.shape == (4, 2) and np.allclose(second[0], [0.0, 1.0])
        assert len(ax.lines) == 1
        _, sampled = plotting.plot_paths(space, [chord], count=3)
        assert np.allclose(sampled.get_segments()[0][:, 0], np.linspace(-1, 1, 5))
        with pytest.raises(ValueError, match="two points"):
            plotting.plot_paths(space, [[(0.5, 10.0)]])

    def test_a_network_marks_its_sources_and_receivers(self):
        from pygeoinf import plotting

        space = disk(2.0)
        paths = [((1.0, 0.0), (1.0, 120.0)), ((1.0, 0.0), (1.0, 240.0))]
        _, artists = plotting.plot_network(space, paths)
        assert len(artists["sources"].get_offsets()) == 1
        assert len(artists["receivers"].get_offsets()) == 2

    def test_subplots_have_an_equal_aspect(self):
        from pygeoinf import plotting

        _, axes = plotting.subplots(disk(2.0), columns=2)
        assert all(ax.get_aspect() == 1.0 for ax in axes)

    def test_a_layered_space_is_one_disk_on_one_scale(self, rng):
        from pygeoinf import plotting

        space = rings()
        x = space.random(rng=rng)
        ax, mappable = plotting.plot(space, x, angles=36, colorbar_label="field")
        meshes = [c for c in ax.collections if hasattr(c, "get_clim")]
        assert len(meshes) == 2
        assert len({mesh.get_clim() for mesh in meshes}) == 1
        assert len(ax.lines) == 2  # the interface and the outer boundary
        for mesh, layer, part in zip(meshes, space.layers, x):
            where, values = self._drawn(mesh)
            points = list(
                zip(
                    np.hypot(where[::11, 0], where[::11, 1]),
                    np.degrees(np.arctan2(where[::11, 1], where[::11, 0])),
                )
            )
            assert np.allclose(values[::11], layer.evaluate(part, points), atol=1e-10)
        with pytest.raises(ValueError, match="padding"):
            plotting.plot(space, x, padding=True)
        plotting.plot_points(space, [(0.2, 10.0), (0.9, 100.0)])
        plotting.plot_paths(space, [[(1.0, 0.0), (0.2, 90.0), (1.0, 180.0)]])


# ------------------------------------------------------------------------- #
#                         Parity with an independent solve                  #
# ------------------------------------------------------------------------- #


def finite_volume_modes(lower, upper, length, m, count, /, *, cells=20000):
    """The leading eigenpairs of ``u - L^2 (1/r)(r u')' + L^2 m^2 u / r^2`` on
    ``[lower, upper]`` under the natural condition, by cell-centred finite
    volumes: the flux ``r u'`` across each face, none across the two ends,
    and the symmetric tridiagonal problem that makes. Second order in the
    cell width, and nothing in common with the spectral elements but the
    equation.

    Returns the eigenvalues, the cell centres, and the modes there as
    columns, orthonormal under ``sum r h u^2``: the ``L^2(r dr)`` norm over
    the whole interval.
    """
    from scipy.linalg import eigh_tridiagonal

    h = (upper - lower) / cells
    centres = lower + (np.arange(cells) + 0.5) * h
    faces = lower + np.arange(1, cells) * h  # the interior faces
    flux = length**2 * faces / h
    diagonal = centres * h + length**2 * m**2 * h / centres
    diagonal[:-1] += flux
    diagonal[1:] += flux
    scale = 1.0 / np.sqrt(centres * h)
    theta, vectors = eigh_tridiagonal(
        diagonal * scale**2,
        -flux * scale[:-1] * scale[1:],
        select="i",
        select_range=(0, count - 1),
    )
    return theta, centres, scale[:, None] * vectors


class TestParity:
    @pytest.mark.parametrize("m", [0, 3])
    def test_the_modes_of_an_annulus_are_those_of_an_independent_solve(self, m):
        """An annulus ``[0.5, 1]`` padded to ``[0.2, 1.3]``, short of the
        centre, under the natural condition at both ends of the mesh: the
        radial modes of order ``m``, eigenvalues and functions, against the
        finite-volume solve on the same interval. The modes are normalized
        over the whole mesh in both, so they are compared as they come, up to
        sign, on the domain. Measured: the eigenvalues agree to 1e-7, the
        reference's own error; the modes to 2e-5 of their largest value at the
        eighth, the elements' error, which halving the elements or raising
        their order takes below 1e-6."""
        length, count = 0.2, 8
        space = Disk(
            3,
            inner_radius=0.5,
            length_scale=length,
            padding=0.3,
            boundary=None,
            radial_modes=count,
            element_length=0.05,
        )
        assert space.radii[0] == pytest.approx(0.2)
        assert space.radii[-1] == pytest.approx(1.3)
        radial = space.basis[m]
        theta, centres, modes = finite_volume_modes(0.2, 1.3, length, m, count)
        assert np.allclose(radial.theta, theta, rtol=3e-7)
        inside = (centres >= 0.5) & (centres <= 1.0)
        ours = radial.evaluate(centres[inside])
        theirs = modes[inside]
        signs = np.sign(np.sum(ours * theirs, axis=0))
        assert np.allclose(ours, signs * theirs, atol=5e-5 * np.abs(theirs).max())


class TestTheIntegralReadsThroughTheComponents:
    def test_a_composition_with_a_multiplication_is_self_consistent(self, rng):
        """A pointwise product is not in the span of the modes; the integral
        reads it through its components, like every operator on the space,
        which is what keeps the adjoint of a composition honest."""
        space = disk(2.0)
        integral = space.integral_functional()
        u, x = space.random(rng=rng), space.random(rng=rng)
        product = space.multiply(u, x)
        assert integral(product) == pytest.approx(
            integral.derivative_components @ space.to_components(product), rel=1e-10
        )
        check_operator(
            integral @ space.multiplication_operator(u),
            rng=rng,
            measure=space.sobolev_measure(2.0),
        )


class TestTheMeshDoesNotDependOnTheUnit:
    @pytest.mark.parametrize("unit", [1.0e-3, 7.3, 3480.0, 6371.0])
    @pytest.mark.parametrize("length_scale", [0.3, 0.15])
    def test_the_same_geometry_in_other_units(self, unit, length_scale):
        """A disk of radius one and one of radius ``unit``, with the length
        scale scaled alike, have the same mesh and so the same basis."""
        here = Lebesgue(4, radius=1.0, length_scale=length_scale)
        there = Lebesgue(4, radius=unit, length_scale=length_scale * unit)
        assert there.grid_shape == here.grid_shape
