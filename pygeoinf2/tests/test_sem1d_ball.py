"""The spectral-element ball and annulus: the space, its operators, its priors.

Needs ``planetmodel`` 1.2 and ``pyshtools``. The independent routes
(DECISIONS.md D-107) are the grid against the point: values synthesized on the
grid through ``pyshtools`` and the nodal eigenfunctions, against the same field
evaluated through scipy's harmonics and the element polynomials; and
``planetmodel``'s banded solve against the eigenbasis.

The two statistical checks take thousands of draws. They carried ``slow`` while
a draw cost a harmonic transform per radial shell, through ``pyshtools`` one at
a time, and took two and a half minutes between them; with the shells
transformed together they take five seconds and run with the rest.
"""

import numpy as np
import pytest

randomfield = pytest.importorskip("planetmodel.randomfield")
if not hasattr(randomfield, "SpectralBasis"):  # pragma: no cover
    pytest.skip("needs planetmodel 1.2 or later", allow_module_level=True)
pytest.importorskip("pyshtools")

from pygeoinf2 import Traits  # noqa: E402
from pygeoinf2.sem1d.ball import Ball, Lebesgue, Sobolev  # noqa: E402
from pygeoinf2.testing import (  # noqa: E402
    check_coordinates,
    check_measure,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)


def ball(order, /, **options):
    options = {"lmax": 5, "length_scale": 0.3, "element_length": 0.3, **options}
    return Ball(options.pop("lmax"), order=order, **options).with_order(order)


def annulus(order, /, **options):
    options = {
        "inner_radius": 0.5,
        "length_scale": 0.2,
        "element_length": 0.2,
        **options,
    }
    return ball(order, **options)


GEOMETRIES = [ball, annulus]


def grid_points(space, /, *, stride=3):
    """Some interior grid points, and where they sit in a grid array."""
    points, where = [], []
    for i in range(0, space.interior_radii.size, stride):
        for j in range(space.latitudes.size):
            for k in range(0, space.longitudes.size, 2):
                points.append(
                    (space.interior_radii[i], space.latitudes[j], space.longitudes[k])
                )
                where.append((i, j, k))
    return points, tuple(np.array(where).T)


class TestTheSpace:
    @pytest.mark.parametrize("build", GEOMETRIES)
    @pytest.mark.parametrize("order", [0.0, 2.0])
    def test_the_axioms(self, build, order, rng):
        space = build(order)
        check_space(space, rng=rng, rebuild=lambda: build(order))
        check_coordinates(space, rng=rng)

    def test_with_order_names_the_subclass_and_shares_the_basis(self):
        sobolev = ball(2.0)
        lebesgue = sobolev.with_order(0.0)
        assert type(sobolev) is Sobolev and type(lebesgue) is Lebesgue
        assert lebesgue.basis is sobolev.basis
        assert lebesgue == Lebesgue(5, length_scale=0.3, element_length=0.3)
        assert sobolev == Sobolev(5, 2.0, 0.3, element_length=0.3)
        assert sobolev.shares_vectors_with(lebesgue)
        assert not sobolev.shares_vectors_with(ball(2.0, lmax=4))
        assert ball(2.0) != annulus(2.0)

    def test_a_ball_is_padded_outwards_only_and_an_annulus_to_the_centre(self):
        assert ball(0.0).padding == pytest.approx((0.0, 0.6))
        assert ball(0.0, boundary=None).padding == pytest.approx((0.0, 1.2))
        assert ball(0.0).radii[0] == 0.0
        shell = annulus(0.0, padding=2.0)
        assert shell.padding == pytest.approx((0.5, 2.0))
        assert shell.radii[0] == 0.0
        assert shell.interior_radii[0] == 0.5 and shell.interior_radii[-1] == 1.0

    def test_the_grid(self):
        space = ball(0.0)
        assert space.grid_shape == (space.radii.size, 6, 11)
        assert np.all(np.diff(space.latitudes) < 0.0)
        assert space.interior_mask.shape == space.grid_shape
        x = space.zero()
        assert space.interior_values(x).shape == (space.interior_radii.size, 6, 11)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_labels_are_the_packing(self, build):
        space = build(0.0)
        basis = space.basis
        for l in range(space.lmax + 1):
            for order in range(-l, l + 1):
                block = basis.block(int(order < 0), l, abs(order))
                assert np.all(space.degrees[block] == l)
                assert np.all(space.orders[block] == order)
                assert np.array_equal(
                    space.radial_indices[block], np.arange(basis.nmodes[l])
                )
                assert np.array_equal(space.eigenvalues[block], basis[l].theta)
        assert space.degrees.size == space.dim

    def test_the_default_truncation_is_isotropic_on_the_domain(self):
        """Every mode of the padded mesh below the first of degree ``lmax`` on
        the domain *without* its padding, and no other: so the shortest
        wavelength kept is the angular one at the outer radius, whatever the
        padding is."""
        # Under the natural condition, whose bound the rule is stated for; a
        # Robin one can lift it, which TestBoundary covers.
        space = ball(0.0, lmax=8, boundary=None)
        bare = ball(0.0, lmax=8, padding=0.0, boundary=None)
        bound = bare.basis.family.eigvalsh(8)[0]
        family = space.basis.family
        for l in range(9):
            assert space.basis.nmodes[l] == np.sum(family.eigvalsh(l) <= bound * 1.0001)
        assert np.all(np.diff(space.basis.nmodes) <= 0)
        # The padded mesh's own first mode of degree lmax is longer, so the old
        # rule kept one mode there and this keeps more.
        assert space.basis.nmodes[8] > 1
        shortest = 2.0 * np.pi * 0.3 / np.sqrt(space.eigenvalues.max() - 1.0)
        assert shortest == pytest.approx(2.0 * np.pi / 8, rel=0.25)
        wider = ball(0.0, lmax=8, padding=2.0, boundary=None)
        assert wider.eigenvalues.max() == pytest.approx(
            space.eigenvalues.max(), rel=0.05
        )
        assert wider.dim > space.dim

    def test_the_truncation_can_be_stated(self):
        assert np.all(ball(0.0, radial_modes=4).basis.nmodes == 4)
        assert ball(0.0, radial_modes=4).dim == 4 * 36
        bounded = ball(0.0, max_eigenvalue=30.0)
        assert bounded.eigenvalues.max() <= 30.0
        with pytest.raises(ValueError, match="not both"):
            ball(0.0, radial_modes=4, max_eigenvalue=30.0)
        with pytest.raises(ValueError, match="without a radial mode"):
            ball(0.0, max_eigenvalue=1.5)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_lebesgue_inner_product_is_the_volume_quadrature(self, build, rng):
        """``int u v r^2 dr dOmega`` over the padded mesh, summed on the grid
        with its own weights, for fields in the span -- whose product the grid
        integrates exactly."""
        space = build(0.0)
        u, v = space.random(rng=rng), space.random(rng=rng)
        grid = space.grid
        weights = space.basis.family.element_mass()
        radial = np.zeros(space.radii.size)
        np.add.at(radial, space.basis.family.mesh.gmap, weights)
        angular = grid.weights[:, None] * (2.0 * np.pi / grid.nphi)
        quadrature = np.sum(radial[:, None, None] * angular[None] * u * v)
        assert space.inner_product(u, v) == pytest.approx(quadrature, rel=1e-10)

    def test_white_noise_components_carry_the_inverse_metric(self, rng):
        """``N(0, G^-1)``, stated on the components, where it costs nothing."""
        space = ball(2.0)
        draws = np.stack([space.white_noise_components(rng=rng) for _ in range(20000)])
        assert np.allclose(draws.var(axis=0) * space.metric_values, 1.0, atol=0.06)

    def test_white_noise_has_identity_covariance(self, rng):
        check_white_noise(
            ball(2.0, lmax=4, element_length=0.4), rng=rng, samples=6000, rtol=0.14
        )


class TestTransform:
    @pytest.mark.parametrize("lmax", [0, 1, 5, 24])
    def test_the_batched_transform_is_the_one_through_pyshtools(self, lmax, rng):
        """Every shell at once, against ``planetmodel``'s shell by shell."""
        from planetmodel import harmonics
        from planetmodel.sampling import gauss_legendre

        from pygeoinf2.sem1d._transform import ShellTransform

        grid = gauss_legendre(lmax)
        transform = ShellTransform(grid)
        coefficients = rng.normal(size=(2, lmax + 1, lmax + 1, 7))
        sine, degree, order = harmonics.packing(lmax)
        keep = np.zeros_like(coefficients, dtype=bool)
        keep[sine, degree, order] = True
        coefficients = np.where(keep, coefficients, 0.0)
        values = transform.synthesise(coefficients)
        assert values.shape == (7, lmax + 1, 2 * lmax + 1)
        assert np.allclose(
            values, harmonics.synthesise_grid(coefficients, grid), atol=1e-11
        )
        assert np.allclose(transform.analyse(values), coefficients, atol=1e-11)
        rough = rng.normal(size=values.shape)
        assert np.allclose(
            transform.analyse(rough),
            harmonics.analyse_grid(rough, grid, lmax=lmax),
            atol=1e-11,
        )

    def test_a_grid_that_is_not_gauss_legendre_is_refused(self):
        from planetmodel.sampling import equiangular, gauss_legendre

        from pygeoinf2.sem1d._transform import ShellTransform

        with pytest.raises(ValueError, match="Gauss-Legendre"):
            ShellTransform(equiangular(6, 11))
        with pytest.raises(ValueError, match="longitudes"):
            ShellTransform(gauss_legendre(5, nphi=16))

    def test_beyond_the_measured_degree_the_other_route_is_taken(
        self, monkeypatch, rng
    ):
        from pygeoinf2.sem1d import ball as module

        monkeypatch.setattr(module, "_BATCHED_LMAX", 3)
        slow, fast = ball(0.0, lmax=4), None
        monkeypatch.setattr(module, "_BATCHED_LMAX", 128)
        fast = ball(0.0, lmax=4)
        assert slow._transform is None and fast._transform is not None
        components = rng.normal(size=fast.dim)
        assert np.allclose(
            slow.from_components(components),
            fast.from_components(components),
            atol=1e-11,
        )
        field = rng.normal(size=fast.grid_shape)
        assert np.allclose(
            slow.to_components(field), fast.to_components(field), atol=1e-11
        )


class TestBoundary:
    def test_robin_halves_the_padding_and_shrinks_the_space(self):
        natural = ball(0.0, lmax=8, max_eigenvalue=60.0, boundary=None)
        matched = ball(0.0, lmax=8, max_eigenvalue=60.0, boundary="robin")
        assert matched.padding == pytest.approx((0.0, 0.6))
        assert matched.robin == pytest.approx((0.0, 0.7 / 0.3 + 1.0 / 1.6))
        assert matched.dim < natural.dim and matched.radii.size < natural.radii.size
        assert matched != natural
        # The inner end of this mesh is at r = 0.1, where the curvature term
        # outweighs the other and the coefficient would be negative: it is
        # held at zero, the natural condition. Further out it is not.
        assert annulus(0.0, boundary="robin").robin[0] == 0.0
        thin = annulus(0.0, length_scale=0.05, boundary="robin")
        assert thin.robin[0] == pytest.approx(0.7 / 0.05 - 1.0 / 0.4)
        # That condition lifts the long modes past the bound the domain alone
        # would set, which is raised to keep a mode at every degree.
        assert np.all(thin.basis.nmodes >= 1)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_two_length_scales_then_do_what_four_do_without_it(self, build):
        """With every degree sharing the one coefficient."""

        def variance(padding, boundary):
            space = build(
                0.0,
                lmax=8,
                length_scale=0.2,
                padding=padding,
                boundary=boundary,
                max_eigenvalue=160.0,
                element_length=0.05,
            )
            field = space.pointwise_variance(space.eigenvalues**-3.0)
            return space.interior_values(field)[:, 0, 0]

        reference = variance(2.0, None)

        def error(padding, boundary):
            return np.abs(variance(padding, boundary) / reference - 1.0).max()

        assert error(0.4, "robin") < 1e-2
        assert error(0.4, None) > 10.0 * error(0.4, "robin")
        assert error(0.4, "robin") < 2.0 * error(0.8, None)


class TestPointEvaluation:
    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_every_basis_function_is_its_own_grid_values(self, build):
        """One component at a time, so that an error in the order of the
        blocks cannot hide in a sum: scipy's harmonics and the element
        polynomials against ``pyshtools`` and the nodal values."""
        space = build(2.0)
        points, where = grid_points(space)
        rows = space.basis_matrix(points)
        for k in range(space.dim):
            unit = np.zeros(space.dim)
            unit[k] = 1.0
            on_grid = space.interior_values(space.from_components(unit))[where]
            assert np.allclose(rows[:, k], on_grid, atol=1e-12)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_a_field_in_the_span_is_evaluated_exactly(self, build, rng):
        space = build(2.0)
        x = space.random(rng=rng)
        points, where = grid_points(space)
        assert np.allclose(
            space.evaluate(x, points), space.interior_values(x)[where], atol=1e-11
        )
        assert space.basis_at(points[4]) == pytest.approx(space.basis_matrix(points)[4])

    def test_the_radial_modes_are_spherical_bessel_functions(self):
        """With ``L`` constant the modes of degree ``l`` are ``j_l(k r)``, ``k``
        read off the eigenvalue -- and so go like ``r^l`` at the centre, which
        is what makes every draw of a prior regular there for nothing. Held
        against the function and not against its exponent: below about
        ``r = 0.05`` a mode of degree four or more is smaller than the mesh's
        own error, and the slope measured there is the error's."""
        from scipy.special import spherical_jn

        space = ball(0.0, lmax=8, radial_modes=8, element_length=0.05)
        radii = np.linspace(0.0, 1.0, 201)
        for l in (0, 1, 2, 4, 8):
            radial = space.basis[l]
            values = radial.evaluate(radii)
            for j in (1, 4, 7):
                k = np.sqrt(radial.theta[j] - 1.0) / 0.3
                exact = spherical_jn(l, k * radii)
                scaled = exact * (values[:, j] @ exact) / (exact @ exact)
                assert np.allclose(
                    values[:, j], scaled, atol=2e-4 * np.abs(values[:, j]).max()
                )
        # Where the exponent can be seen it is the degree.
        near = np.array([0.01, 0.04])
        for l in (1, 2, 3):
            first = space.basis[l].evaluate(near)[:, 0]
            assert np.log(first[1] / first[0]) / np.log(4.0) == pytest.approx(
                l, abs=0.01
            )

    def test_longitude_is_an_angle(self, rng):
        space = ball(2.0)
        x = space.random(rng=rng)
        assert space.evaluate(x, [(0.6, 20.0, -150.0)]) == pytest.approx(
            space.evaluate(x, [(0.6, 20.0, 210.0)])
        )

    def test_the_centre_has_one_value(self, rng):
        """Only degree zero reaches the centre, so the angles there are idle."""
        space = ball(2.0)
        x = space.random(rng=rng)
        values = space.evaluate(x, [(0.0, 10.0, 20.0), (0.0, -70.0, 200.0)])
        assert values[0] == pytest.approx(values[1], abs=1e-12)

    def test_it_is_refused_at_and_below_order_three_halves(self):
        for order in (0.0, 1.5):
            with pytest.raises(ValueError, match="above 1.5"):
                ball(order).dirac((0.5, 0.0, 0.0))
            with pytest.raises(ValueError, match="above 1.5"):
                ball(order).point_evaluation_operator([(0.5, 0.0, 0.0)])
        ball(0.0).dirac((0.5, 0.0, 0.0), unsafe=True)
        ball(1.75).dirac((0.5, 0.0, 0.0))

    def test_a_radius_outside_the_domain_is_refused(self):
        with pytest.raises(ValueError):
            ball(2.0).dirac((1.2, 0.0, 0.0))
        with pytest.raises(ValueError):
            annulus(2.0).dirac((0.3, 0.0, 0.0))

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_operator_and_its_adjoint(self, build, rng):
        space = build(2.0)
        points = [space.random_point(rng=rng) for _ in range(5)]
        operator = space.point_evaluation_operator(points)
        check_operator(operator, rng=rng)
        x = space.random(rng=rng)
        assert np.allclose(operator(x), space.evaluate(x, points))
        assert operator(x)[1] == pytest.approx(space.dirac(points[1])(x))

    def test_random_points_lie_in_the_domain(self, rng):
        space = annulus(0.0)
        points = np.array([space.random_point(rng=rng) for _ in range(200)])
        assert np.all((points[:, 0] >= 0.5) & (points[:, 0] <= 1.0))
        assert np.all(np.abs(points[:, 1]) <= 90.0)

    @staticmethod
    def _truncation_error(space, function, extension):
        """Relative ``L2`` error over the domain, and the largest error by
        radius, of a sampled function once settled into the span, against the
        function's own values at the domain's grid points -- and not against
        the sample, which under ``"fit"`` is in the span already."""
        sampled = space.project_function(function, extension=extension)
        grid = space.grid
        weights = (
            space.basis[0].weights()[:, None, None]
            * (grid.weights[:, None] * 2.0 * np.pi / grid.nphi)[None]
        )
        exact = space.interior_values(
            space.project_function(function, extension="constant")
        )
        missed = space.interior_values(space.truncate(sampled)) - exact
        norm = np.sqrt(np.sum(weights * missed**2) / np.sum(weights * exact**2))
        return norm, np.abs(missed).max(axis=(1, 2))

    @staticmethod
    def _wavy(point):
        radius, latitude, _ = point
        return np.cos(3.0 * radius) * (1.0 + radius * np.sin(np.radians(latitude)))

    def test_more_radial_modes_bring_a_sampled_function_closer(self):
        """In the norm of the domain, which is the claim that holds: at any
        one point the error of a truncated expansion changes sign with the
        modes kept and can pass through zero by accident."""
        errors = [
            self._truncation_error(
                annulus(
                    0.0,
                    lmax=4,
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
        assert errors[-1] < 2e-3

    def test_the_reflection_mends_the_ends_of_an_annulus(self):
        """Held constant across the padding a function has a kink at each end
        of the domain, and the error sits there; reflected, it has none. A fact
        of the natural condition and its longer padding, under which it was
        measured, as the two tests beside it are; under the Robin default the
        two fills come out alike, and the fit is the one that does not care
        (DECISIONS.md D-124)."""
        space = annulus(
            0.0, lmax=4, radial_modes=32, element_length=0.05, boundary=None
        )
        held, by_radius_held = self._truncation_error(space, self._wavy, "constant")
        mirrored, by_radius = self._truncation_error(space, self._wavy, "odd")
        assert mirrored < 0.7 * held
        assert by_radius[0] < 0.2 * by_radius_held[0]
        assert by_radius[-1] < 0.5 * by_radius_held[-1]

    def test_in_a_ball_the_reflection_mends_the_surface_and_costs_the_centre(self):
        """Measured, and kept because a caller should know of it: the surface
        is an order of magnitude better reflected and the norm over the domain
        better too, while near the centre this function is the worse for it.
        The centre is badly controlled under either fill -- a radial mode's
        value there grows with its index -- and what the reflection adds is
        one more error focused on it; for a purely radial function the two
        fills are alike there (DECISIONS.md D-119)."""
        space = ball(0.0, lmax=4, radial_modes=32, element_length=0.05, boundary=None)
        held, by_radius_held = self._truncation_error(space, self._wavy, "constant")
        mirrored, by_radius = self._truncation_error(space, self._wavy, "odd")
        assert mirrored < held
        assert by_radius[-1] < 0.2 * by_radius_held[-1]
        near_centre = space.interior_radii < 0.25
        assert by_radius[near_centre].max() > by_radius_held[near_centre].max()

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_a_fit_over_the_domain_is_right_everywhere(self, build):
        """For a function regular at the centre, the centre included: what
        looked like the centre's trouble was the projection of a continued
        function over the whole padded ball, and a fit over the domain alone
        has none of it (DECISIONS.md D-124)."""
        space = build(0.0, lmax=4, radial_modes=16, element_length=0.05)
        fitted, by_radius = self._truncation_error(space, self._wavy, "fit")
        mirrored, _ = self._truncation_error(space, self._wavy, "odd")
        assert fitted < 1e-7 and fitted < 1e-3 * mirrored
        assert by_radius.max() < 1e-4
        field = space.project_function(self._wavy, extension="fit")
        assert np.allclose(space.truncate(field), field, atol=1e-10)

    def test_an_extension_is_named(self):
        space = ball(0.0, lmax=2)
        with pytest.raises(ValueError, match="odd"):
            space.project_function(lambda p: 1.0, extension="even")
        positive = lambda p: 1.0 + 5.0 * (1.0 - p[0])  # noqa: E731
        assert np.any(space.project_function(positive, extension="odd") < 0.0)
        assert np.all(space.project_function(positive, extension="constant") > 0.0)


class TestPointwiseAlgebra:
    def test_a_product_is_left_on_the_grid(self, rng):
        space = ball(2.0)
        x, y = space.random(rng=rng), space.random(rng=rng)
        assert np.array_equal(space.multiply(x, y), x * y)
        assert not np.allclose(space.truncate(x * y), x * y)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_multiplication_is_self_adjoint_on_lebesgue_only(self, build, rng):
        lebesgue = build(0.0)
        # 1 + 0.3 x, which is regular at the centre; r cos(longitude) is not.
        f = lebesgue.project_function(
            lambda p: 1.0
            + 0.3 * p[0] * np.cos(np.radians(p[1])) * np.cos(np.radians(p[2]))
        )
        on_lebesgue = lebesgue.multiplication_operator(f)
        assert Traits.SELF_ADJOINT in on_lebesgue.traits
        check_operator(on_lebesgue, rng=rng)
        check_traits(on_lebesgue, rng=rng)
        on_sobolev = build(2.0).multiplication_operator(f)
        assert Traits.SELF_ADJOINT not in on_sobolev.traits
        check_operator(on_sobolev, rng=rng)

    def test_the_support_projection_zeroes_the_padding(self, rng):
        space = ball(0.0)
        projection = space.support_projection()
        check_operator(projection, rng=rng)
        x = space.random(rng=rng)
        assert np.all(projection(x)[~space.interior_mask] == 0.0)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_integral_runs_over_the_domain_alone(self, build, rng):
        """Against a quadrature of the grid values themselves: the mass of the
        domain's own radial elements times the grid's angular weights, which
        integrates a field in the span exactly and gives the padding no
        weight."""
        space = build(0.0)
        x = space.random(rng=rng)
        grid = space.grid
        angular = grid.weights[:, None] * (2.0 * np.pi / grid.nphi)
        radial = space.basis[0].weights()
        quadrature = np.sum(
            radial[:, None, None] * angular[None] * space.interior_values(x)
        )
        assert space.integral_functional()(x) == pytest.approx(quadrature, rel=1e-10)

    def test_the_integral_of_one_is_the_volume_when_nothing_is_dropped(self):
        # A padding that stops short of the centre: a mesh that reaches it is
        # a ball's, which drops the axis node above degree zero, and then no
        # one count of modes is "all of them" at every degree.
        shell = annulus(0.0, lmax=2, element_length=0.25, padding=0.3)
        full = annulus(
            0.0, lmax=2, element_length=0.25, padding=0.3, radial_modes=shell.radii.size
        )
        assert full.integral_functional()(np.ones(full.grid_shape)) == pytest.approx(
            full.domain_volume, rel=1e-10
        )


class TestMeasures:
    def test_the_sobolev_measure_is_the_banded_solve(self, rng):
        """Covariance ``A^-1`` on the Lebesgue space of an annulus with every
        radial mode kept, against a Cholesky solve of each degree's pencil
        that never forms an eigenvector."""
        shell = annulus(0.0, lmax=3, element_length=0.25, padding=0.3)
        space = annulus(
            0.0, lmax=3, element_length=0.25, padding=0.3, radial_modes=shell.radii.size
        )
        family = space.basis.family
        covariance = space.sobolev_measure(1.0).covariance
        x = space.random(rng=rng)
        got = space.basis.synthesise(space.to_components(covariance(x)), physical=False)
        given = space.basis.synthesise(space.to_components(x), physical=False)
        for l in range(4):
            for part, order in ((0, 0), (0, l), (1, l)):
                if part == 1 and order == 0:
                    continue
                assert np.allclose(
                    got[part, l, order],
                    family.solve(l, given[part, l, order]),
                    atol=1e-9,
                )

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_pointwise_variance_is_the_sum_of_squared_rows(self, build, rng):
        """At grid points, against ``basis_matrix``: once for a function of
        ``A``, where the addition theorem does the angular sum, and once for
        variances that depend on the order, where it cannot."""
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
        space = ball(0.0)
        field = space.pointwise_variance(space.eigenvalues**-2.0)
        assert np.allclose(field, field[:, :1, :1])

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_pointwise_std_is_what_the_covariance_says(self, build):
        """``diag(E C E*)`` through the operator algebra, against a field
        asked for on the padding too, and smooth: what the covariance sees is
        the scaled draw's projection onto the kept modes. ``r^2`` and not
        ``r``, which is a cone at the centre of a ball; see the next test."""
        space = build(0.0, radial_modes=12, element_length=0.1)
        target = 0.3 + 0.1 * space.radii[:, None, None] ** 2 + space.zero()
        measure = space.sobolev_measure(2.5, pointwise_std=target)
        radii = np.linspace(space.inner_radius, space.radius, 5)
        evaluation = space.point_evaluation_operator(
            [(r, 30.0, 60.0) for r in radii], unsafe=True
        )
        matrix = (evaluation @ measure.covariance @ evaluation.adjoint).matrix(
            form="components"
        )
        assert np.allclose(np.sqrt(np.diag(matrix)), 0.3 + 0.1 * radii**2, rtol=5e-3)

    def test_a_profile_with_a_cusp_at_the_centre_converges_slowly_there(self):
        """The one target here that is *not* regular at the centre, to show
        what that costs. A field analytic there has a part of degree ``l``
        going like ``r^l`` times a series in ``r^2``, so a profile in ``r``
        alone, being of degree zero, must be even; one linear in ``r`` is a
        cone, and the modes come to it at first order: the error at the centre
        halves when they double, and is an order of magnitude above that of a
        profile in ``r^2``. Kept because a profile from a one-dimensional
        model is the thing a user has to hand."""

        def error(profile, modes):
            space = ball(0.0, radial_modes=modes, element_length=0.05, boundary=None)
            target = profile(space.radii)[:, None, None] + space.zero()
            measure = space.sobolev_measure(2.5, pointwise_std=target)
            evaluation = space.point_evaluation_operator([(0.0, 0.0, 0.0)], unsafe=True)
            variance = (evaluation @ measure.covariance @ evaluation.adjoint).matrix(
                form="components"
            )[0, 0]
            return abs(np.sqrt(variance) / profile(0.0) - 1.0)

        cone = [error(lambda r: 0.3 + 0.1 * r, modes) for modes in (12, 24)]
        smooth = error(lambda r: 0.3 + 0.1 * r**2, 24)
        assert 0.35 < cone[1] / cone[0] < 0.65
        assert cone[1] > 5.0 * smooth

    def test_the_diagonal_measure_has_a_precision(self):
        space = ball(2.0)
        assert space.sobolev_measure(1.0).precision is not None
        with pytest.raises(ValueError, match="positive"):
            space.sobolev_measure(1.0, pointwise_std=-1.0)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_moments_match_the_samples(self, build, rng):
        space = build(2.0, lmax=4, element_length=0.4)
        check_measure(
            space.sobolev_measure(1.0, amplitude=0.7), rng=rng, samples=4000, rtol=0.18
        )
        check_measure(
            space.sobolev_measure(1.0, pointwise_std=0.3),
            rng=rng,
            samples=4000,
            rtol=0.18,
        )
