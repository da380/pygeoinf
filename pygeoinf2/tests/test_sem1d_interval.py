"""The spectral-element interval: the space, its operators and its priors.

Needs ``planetmodel`` 1.2, whose ``randomfield`` surface does the numerics.
The independent routes (DECISIONS.md D-107) are that library's *direct*
methods -- the banded bilinear form and the banded Cholesky solve -- against
the eigenbasis this space works in, and closed forms where there is one.
"""

import numpy as np
import pytest

randomfield = pytest.importorskip("planetmodel.randomfield")
if not hasattr(randomfield, "SpectralBasis"):  # pragma: no cover
    pytest.skip("needs planetmodel 1.2 or later", allow_module_level=True)

from pygeoinf2 import Traits  # noqa: E402
from pygeoinf2.sem1d.interval import Interval, Lebesgue, Sobolev  # noqa: E402
from pygeoinf2.testing import (  # noqa: E402
    check_coordinates,
    check_measure,
    check_operator,
    check_representer,
    check_space,
    check_traits,
    check_white_noise,
)


def varying(x):
    """A length scale that halves across ``[-1, 2]``."""
    return 0.2 - 0.1 * (np.asarray(x) + 1.0) / 3.0


def build(order, /, *, modes=40, length_scale=varying, **options):
    if order == 0.0:
        return Lebesgue(
            modes, lower=-1.0, upper=2.0, length_scale=length_scale, **options
        )
    return Sobolev(modes, order, length_scale, lower=-1.0, upper=2.0, **options)


def complete(order, /, *, length_scale=0.3):
    """A space that keeps every mode of its mesh, so nothing is truncated."""
    options = dict(
        lower=-1.0,
        upper=2.0,
        order=order,
        length_scale=length_scale,
        element_length=0.5,
    )
    nodes = Interval(1, **options).nodes.size
    return Interval(nodes, **options)


class TestTheSpace:
    @pytest.mark.parametrize("order", [0.0, 1.0, 2.5])
    def test_the_axioms(self, order, rng):
        space = build(order)
        check_space(space, rng=rng, rebuild=lambda: build(order))
        check_coordinates(space, rng=rng)
        check_white_noise(space, rng=rng)
        check_representer(space, rng.normal(size=space.dim), rng=rng)

    def test_the_interval_may_start_below_zero(self):
        space = build(1.0)
        assert space.bounds == (-1.0, 2.0)
        assert space.nodes[0] < -1.0 and space.nodes[-1] > 2.0
        assert space.interior_nodes[0] == -1.0 and space.interior_nodes[-1] == 2.0
        assert np.array_equal(space.nodes[space.interior_mask], space.interior_nodes)

    def test_the_default_padding_is_two_length_scales_at_each_end(self):
        """Under the Robin condition, which is the default; four under the
        natural one, which needs them (DECISIONS.md D-122)."""
        assert build(1.0).padding == pytest.approx((0.4, 0.2))
        assert build(1.0, boundary=None).padding == pytest.approx((0.8, 0.4))

    def test_the_metric_is_a_power_of_the_eigenvalues(self):
        space = build(2.5)
        assert np.all(space.eigenvalues >= 1.0 - 1e-10)
        assert np.all(np.diff(space.eigenvalues) > 0.0)
        assert np.allclose(space.metric_values, space.eigenvalues**2.5)
        assert build(0.0).is_orthonormal

    def test_the_spectrum_without_padding_is_the_neumann_cosines(self):
        space = Lebesgue(
            12,
            upper=2.0,
            length_scale=0.5,
            padding=0.0,
            boundary=None,
            element_length=0.05,
        )
        n = np.arange(12)
        assert np.allclose(
            space.eigenvalues, 1.0 + (0.5 * n * np.pi / 2.0) ** 2, rtol=1e-8
        )

    def test_equality_is_structural_and_sees_the_length_scale(self):
        assert build(1.0) == build(1.0)
        assert build(1.0) != build(2.0)
        assert build(1.0) != build(1.0, length_scale=lambda x: 1.5 * varying(x))
        assert build(1.0, length_scale=0.2) == build(1.0, length_scale=0.2)

    def test_with_order_names_the_subclass_and_shares_the_basis(self):
        sobolev = build(2.0)
        lebesgue = sobolev.with_order(0.0)
        assert type(lebesgue) is Lebesgue and lebesgue == build(0.0)
        assert type(lebesgue.with_order(2.0)) is Sobolev
        assert lebesgue.with_order(2.0) == sobolev
        assert lebesgue.basis is sobolev.basis
        assert sobolev.shares_vectors_with(lebesgue)
        assert not sobolev.shares_vectors_with(build(2.0, modes=30))

    def test_a_mesh_too_coarse_for_the_modes_is_refused(self):
        with pytest.raises(ValueError, match="cannot hold"):
            Lebesgue(500, element_length=0.5)

    @pytest.mark.parametrize("order", [0.0, 1.0])
    def test_the_inner_product_is_the_banded_bilinear_form(self, order, rng):
        """``(u, v)_{H^1} = u^T K v``, with no eigenvector in sight."""
        space = complete(order)
        family = space.basis.family
        u, v = rng.normal(size=(2, space.nodes.size))
        direct = u @ (family.mass(0) * (family.apply(0, v) if order else v))
        assert space.inner_product(u, v) == pytest.approx(direct, rel=1e-10)

    def test_the_inclusion_between_orders(self, rng):
        sobolev = build(2.0)
        inclusion = sobolev.order_inclusion_operator(sobolev.with_order(0.0))
        check_operator(inclusion, rng=rng)
        x = sobolev.random(rng=rng)
        assert np.array_equal(inclusion(x), x)
        with pytest.raises(ValueError):
            sobolev.order_inclusion_operator(build(0.0, modes=30))


class TestBoundary:
    def test_robin_is_matched_to_the_length_scale_and_halves_the_padding(self):
        space = build(0.0, length_scale=0.1, boundary=None)
        assert space.robin == (0.0, 0.0) and space.padding == pytest.approx((0.4, 0.4))
        matched = Lebesgue(
            40, lower=-1.0, upper=2.0, length_scale=0.1, boundary="robin"
        )
        assert matched.padding == pytest.approx((0.2, 0.2))
        assert matched.robin == pytest.approx((7.0, 7.0))
        assert matched != space
        varied = Lebesgue(
            40, lower=-1.0, upper=2.0, length_scale=varying, boundary="robin"
        )
        assert varied.robin == pytest.approx((0.7 / 0.2, 0.7 / 0.1))

    def test_coefficients_can_be_given(self):
        assert Lebesgue(20, boundary=3.0).robin == (3.0, 3.0)
        assert Lebesgue(20, boundary=(None, 2.0)).robin == (0.0, 2.0)
        with pytest.raises(ValueError, match="robin"):
            Lebesgue(20, boundary="dirichlet")
        with pytest.raises(ValueError, match="non-negative"):
            Lebesgue(20, boundary=-1.0)

    @staticmethod
    def _variance_error(power, padding, boundary):
        def variance(padding, boundary):
            space = Lebesgue(
                400,
                upper=2.0,
                length_scale=0.1,
                padding=padding,
                boundary=boundary,
                element_length=0.02,
            )
            return space.interior_values(
                space.pointwise_variance(space.eigenvalues**-power)
            )

        return np.abs(variance(padding, boundary) / variance(1.2, None) - 1.0).max()

    def test_the_exact_condition_needs_no_padding_at_all(self):
        """``u' = -u / L`` is what the decaying solution of ``A u = 0`` does,
        so with it a mesh that stops at the domain gives the covariance
        ``A^-1`` of one that goes on; the natural condition doubles the
        variance there. What is left is the truncation's."""
        assert self._variance_error(1.0, 0.0, 1.0 / 0.1) < 5e-2
        assert self._variance_error(1.0, 0.0, None) > 0.9

    def test_two_length_scales_then_do_what_four_do_without_it(self):
        """For the higher powers priors are made of, with the coefficient
        ``0.7 / L`` that ``"robin"`` names."""
        for power in (2.0, 3.0):
            matched = self._variance_error(power, 0.2, "robin")
            assert matched < 1e-2
            assert self._variance_error(power, 0.2, None) > 10.0 * matched
            assert matched < 3.0 * self._variance_error(power, 0.4, None)


class TestPointwiseAlgebra:
    def test_a_product_is_left_on_the_nodes(self, rng):
        space = build(1.0)
        x, y = space.random(rng=rng), space.random(rng=rng)
        product = space.multiply(x, y)
        assert np.array_equal(product, x * y)
        assert not np.allclose(space.truncate(product), product)
        assert np.allclose(
            space.to_components(space.truncate(product)), space.to_components(product)
        )

    def test_multiplication_is_self_adjoint_on_lebesgue_only(self, rng):
        f = build(0.0).project_function(lambda x: 1.0 + 0.5 * np.sin(3.0 * x))
        on_lebesgue = build(0.0).multiplication_operator(f)
        assert Traits.SELF_ADJOINT in on_lebesgue.traits
        check_operator(on_lebesgue, rng=rng)
        check_traits(on_lebesgue, rng=rng)
        on_sobolev = build(2.0).multiplication_operator(f)
        assert Traits.SELF_ADJOINT not in on_sobolev.traits
        check_operator(on_sobolev, rng=rng)

    def test_the_support_projection_zeroes_the_padding(self, rng):
        space = build(0.0)
        projection = space.support_projection()
        check_operator(projection, rng=rng)
        x = space.random(rng=rng)
        assert np.all(projection(x)[~space.interior_mask] == 0.0)
        assert np.array_equal(
            space.interior_values(projection(x)), space.interior_values(x)
        )

    def test_the_integral_runs_over_the_domain_alone(self):
        space = complete(0.0)
        integral = space.integral_functional()
        assert integral(np.ones(space.nodes.size)) == pytest.approx(3.0, rel=1e-12)
        assert integral(space.project_function(np.cos)) == pytest.approx(
            np.sin(2.0) - np.sin(-1.0), rel=1e-8
        )


class TestSamplingAFunction:
    def test_the_padding_continues_the_function_with_its_slope(self):
        space = build(0.0)
        sampled = space.project_function(lambda x: 3.0 + 2.0 * x, extension="odd")
        assert np.allclose(sampled, 3.0 + 2.0 * space.nodes)
        held = space.project_function(lambda x: 3.0 + 2.0 * x, extension="constant")
        assert np.allclose(
            held[~space.interior_mask],
            3.0 + 2.0 * np.clip(space.nodes, -1.0, 2.0)[~space.interior_mask],
        )
        assert np.array_equal(
            space.interior_values(held), space.interior_values(sampled)
        )

    @pytest.mark.parametrize("extension", ["fit", "odd", "constant"])
    def test_the_function_is_never_asked_about_the_padding(self, extension):
        asked = []
        build(0.0).project_function(
            lambda x: asked.append(x) or 0.0, extension=extension
        )
        assert min(asked) >= -1.0 and max(asked) <= 2.0

    def test_the_reflection_mends_the_endpoints(self):
        """``cos`` on ``[-1, 2]``: held constant it has a kink at each
        endpoint, which the kept modes come to at first order; reflected, the
        error over the domain is that of knowing ``cos`` on the padding. Under
        the natural condition and its four length scales of padding, which is
        where this was measured; the next test is the default."""
        space = Lebesgue(
            64,
            lower=-1.0,
            upper=2.0,
            length_scale=0.1,
            boundary=None,
            element_length=0.02,
        )
        exact = np.cos(space.interior_nodes)
        weights = space.basis.weights()

        def errors(field):
            missed = space.interior_values(space.truncate(field)) - exact
            return np.sqrt(weights @ missed**2 / (weights @ exact**2)), abs(missed[-1])

        held = errors(space.project_function(np.cos, extension="constant"))
        mirrored = errors(space.project_function(np.cos, extension="odd"))
        known = errors(np.cos(space.nodes))
        assert mirrored[1] < 0.2 * held[1]
        assert mirrored[0] < 0.6 * held[0]
        assert mirrored[0] == pytest.approx(known[0], rel=0.15)

    def test_under_the_default_the_fills_are_alike_and_the_fit_is_not(self):
        """Under the Robin default a continued function does not satisfy the
        condition at the mesh's end and the padding is half as long, and the
        two fills come out alike over the domain; the fit asks nothing of the
        padding and is untouched."""
        space = Lebesgue(
            64, lower=-1.0, upper=2.0, length_scale=0.1, element_length=0.02
        )
        exact = np.cos(space.interior_nodes)
        weights = space.basis.weights()

        def error(extension):
            field = space.truncate(space.project_function(np.cos, extension=extension))
            missed = space.interior_values(field) - exact
            return np.sqrt(weights @ missed**2 / (weights @ exact**2))

        held, mirrored, fitted = error("constant"), error("odd"), error("fit")
        assert 0.5 < mirrored / held < 2.0
        assert fitted < 1e-4 * mirrored

    def test_an_extension_is_named(self):
        with pytest.raises(ValueError, match="odd"):
            build(0.0).project_function(np.cos, extension="even")

    def test_a_fit_over_the_domain_converges_as_the_function_is_smooth(self):
        """It asks nothing of the padding, so no continuation's kink, of any
        order, is there to slow it: orders of magnitude inside the reflection
        at the same modes, in the span of the kept modes, and with
        coefficients no larger."""
        space = Lebesgue(
            64, lower=-1.0, upper=2.0, length_scale=0.1, element_length=0.02
        )
        exact = np.cos(space.interior_nodes)
        fitted = space.project_function(np.cos)  # the default
        assert np.allclose(space.interior_values(fitted), exact, atol=1e-9)
        assert np.allclose(space.truncate(fitted), fitted, atol=1e-12)
        mirrored = space.truncate(space.project_function(np.cos, extension="odd"))
        assert np.abs(space.interior_values(mirrored) - exact).max() > 1e-4
        assert (
            np.abs(space.to_components(fitted)).max()
            < 1.2 * np.abs(space.to_components(mirrored)).max()
        )


class TestPointEvaluation:
    def test_it_is_refused_at_and_below_order_one_half(self):
        for order in (0.0, 0.5):
            with pytest.raises(ValueError, match="above 0.5"):
                build(order).dirac(0.3)
            with pytest.raises(ValueError, match="above 0.5"):
                build(order).point_evaluation_operator([0.3])
        build(0.0).dirac(0.3, unsafe=True)
        build(0.75).dirac(0.3)

    def test_points_outside_the_domain_are_refused(self):
        with pytest.raises(ValueError):
            build(1.0).dirac(2.5)

    def test_the_basis_reproduces_the_field_at_the_nodes(self, rng):
        space = build(1.0)
        x = space.random(rng=rng)
        nodes = space.interior_nodes
        assert np.allclose(
            space.evaluate(x, nodes), space.interior_values(x), atol=1e-12
        )
        assert np.allclose(space.basis_matrix(nodes[:3])[1], space.basis_at(nodes[1]))

    def test_between_the_nodes_it_is_the_element_polynomial(self):
        """A quartic is one polynomial per element, so it is evaluated exactly
        once every mode is kept."""
        space = complete(1.0)
        quartic = lambda x: 1.0 - x + 0.5 * x**4  # noqa: E731
        field = quartic(space.nodes)
        points = np.array([-1.0, -0.123, 0.4567, 1.999, 2.0])
        assert np.allclose(space.evaluate(field, points), quartic(points), atol=1e-10)

    def test_the_operator_and_its_adjoint(self, rng):
        space = build(1.5)
        points = [-0.9, 0.0, 0.3, 1.99]
        operator = space.point_evaluation_operator(points)
        check_operator(operator, rng=rng)
        x = space.random(rng=rng)
        assert np.allclose(operator(x), space.evaluate(x, points))
        assert operator(x)[2] == pytest.approx(space.dirac(0.3)(x))
        with pytest.raises(ValueError, match="one point"):
            space.point_evaluation_operator([])

    def test_the_representer_norm_converges_only_above_one_half(self):
        """The reason for the guard, seen: the squared norm of the Dirac's
        representer settles as modes are added at order one and keeps growing
        at order zero."""

        def squared_norm(order, modes):
            space = Interval(modes, order=order, length_scale=0.2, element_length=0.02)
            return space.dirac(0.5, unsafe=True).representer

        norms = {
            order: [
                Interval(
                    m, order=order, length_scale=0.2, element_length=0.02
                ).squared_norm(squared_norm(order, m))
                for m in (40, 80, 160)
            ]
            for order in (0.0, 1.0)
        }
        assert norms[1.0][2] / norms[1.0][1] < 1.05
        assert norms[0.0][2] / norms[0.0][1] > 1.8


class TestMeasures:
    def test_the_sobolev_measure_is_the_banded_solve(self, rng):
        """Covariance ``A^-1`` on the Lebesgue space, against a Cholesky solve
        of the pencil that never forms an eigenvector."""
        space = complete(0.0)
        covariance = space.sobolev_measure(1.0).covariance
        v = rng.normal(size=space.nodes.size)
        assert np.allclose(covariance(v), space.basis.family.solve(0, v), atol=1e-10)

    @pytest.mark.parametrize("order", [0.0, 1.0])
    def test_the_moments_match_the_samples(self, order, rng):
        space = build(order)
        check_measure(space.sobolev_measure(1.5, amplitude=0.7), rng=rng)
        check_measure(space.sobolev_measure(1.5, pointwise_std=0.3), rng=rng)

    def test_the_diagonal_measure_has_a_precision_and_the_scaled_one_none(self):
        space = build(1.0)
        assert space.sobolev_measure(1.0).precision is not None
        measure = space.sobolev_measure(1.0, pointwise_std=0.3)
        assert measure.covariance_factor is not None

    def test_the_raw_variance_follows_the_length_scale(self):
        """Nothing is homogeneous: where ``L`` is halved the variance of
        ``A^-p`` is about doubled, which is why one number cannot calibrate
        it."""
        space = build(1.0, modes=120)
        raw = space.interior_values(space.pointwise_variance(space.eigenvalues**-1.0))
        assert 1.6 < raw[-1] / raw[0] < 2.4

    @pytest.mark.parametrize("order", [0.0, 1.0])
    def test_pointwise_std_is_what_the_covariance_says(self, order):
        """``diag(E C E*)`` through the operator algebra, against the field
        asked for, with ``L`` varying underneath it. The field is given on the
        padding too, and smoothly: what the covariance sees is the scaled
        draw's projection onto the kept modes, and a kink at an endpoint is
        what that projection is worst at."""
        space = build(order, modes=120)
        target = 0.3 + 0.1 * space.nodes
        measure = space.sobolev_measure(1.5, pointwise_std=target)
        points = np.linspace(-1.0, 2.0, 13)
        evaluation = space.point_evaluation_operator(points, unsafe=True)
        matrix = (evaluation @ measure.covariance @ evaluation.adjoint).matrix(
            form="components"
        )
        assert np.allclose(np.sqrt(np.diag(matrix)), 0.3 + 0.1 * points, rtol=2e-3)

    def test_a_standard_deviation_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            build(1.0).sobolev_measure(1.0, pointwise_std=0.0)

    def test_the_padding_keeps_the_boundary_out_of_the_domain(self):
        """The variance on the domain does not care whether the ends of the
        mesh are four length scales away or eight, and does care when they are
        the ends of the domain."""

        def variance(padding):
            space = Lebesgue(
                100,
                upper=2.0,
                length_scale=0.1,
                padding=padding,
                boundary=None,
                element_length=0.05,
            )
            return space.interior_values(
                space.pointwise_variance(space.eigenvalues**-2.0)
            )

        far, farther, none = variance(0.4), variance(0.8), variance(0.0)
        assert np.allclose(far, farther, rtol=5e-3)
        # On the whole line the variance of A^-2 is 1 / (4 L); the modes
        # dropped account for what is missing.
        assert np.allclose(farther[20:-20], 2.5, rtol=1e-3)
        assert np.allclose(far[20:-20], farther[20:-20], rtol=1e-3)
        # A reflecting end doubles the variance there, by the method of images.
        assert none[0] / farther[0] == pytest.approx(2.0, rel=5e-3)
