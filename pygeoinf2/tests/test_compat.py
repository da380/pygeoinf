"""
Real v1 spaces driven through the v2 core.

This is the M2 acceptance: the existing sphere and circle machinery — Sobolev
spaces, point evaluation, invariant covariances — running against the new
algebra, with results checked against v1 where v1 is right and against the
mathematics where it is not.
"""

import numpy as np
import pytest

pygeoinf = pytest.importorskip("pygeoinf")

from pygeoinf.symmetric_space.circle import Sobolev as CircleSobolev  # noqa: E402

from pygeoinf2.algebra.operators import LinearOperator  # noqa: E402
from pygeoinf2.compat import (  # noqa: E402
    adapt_form,
    adapt_operator,
    adapt_space,
)
from pygeoinf2.testing import (  # noqa: E402
    check_coordinates,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)
from pygeoinf2.traits import Traits  # noqa: E402

KMAX, ORDER, SCALE = 8, 2.0, 0.2
POINTS = [0.4, 1.7, 3.0, 4.8]


def v1_space():
    return CircleSobolev(KMAX, ORDER, SCALE)


@pytest.fixture
def X():
    return adapt_space(v1_space())


class TestAdaptedSpace:
    def test_axioms(self, X, rng):
        check_space(X, rng=rng, rebuild=lambda: adapt_space(v1_space()))

    def test_coordinates(self, X, rng):
        check_coordinates(X, rng=rng)

    def test_it_is_not_orthonormal(self, X):
        """Otherwise this whole file would prove nothing about the metric."""
        assert not X.is_orthonormal
        gram = X.gram_matrix()
        assert not np.allclose(gram, np.identity(X.dim))

    def test_gram_is_v1s_riesz_map(self, X, rng):
        """The mass matrix was already there, under the name to_dual."""
        base = X.v1_space
        c = rng.normal(size=X.dim)
        expected = base.dual.to_components(base.to_dual(base.from_components(c)))
        assert np.allclose(X.apply_gram(c), expected)

    def test_inner_product_matches_v1(self, X, rng):
        base = X.v1_space
        x, y = X.random(rng), X.random(rng)
        assert X.inner_product(x, y) == pytest.approx(base.inner_product(x, y))

    def test_equality_survives_reconstruction(self):
        """v1 spaces are unhashable; the adapter must not inherit that."""
        a, b = adapt_space(v1_space()), adapt_space(v1_space())
        assert a is not b
        assert a == b and hash(a) == hash(b)
        assert {a: "value"}[b] == "value"

    def test_a_different_space_is_unequal(self):
        assert adapt_space(v1_space()) != adapt_space(CircleSobolev(8, 3.0, SCALE))

    def test_gram_diagonality_is_detected_without_forming_it(self, X):
        """A Sobolev space on the sphere at degree 128 has dim above 16000."""
        assert X._diagonal_gram() is not None
        assert np.allclose(X._diagonal_gram(), np.diag(X.gram_matrix()))


class TestWhiteNoiseIsFixed:
    """The v1 defect of DESIGN.md section 9, on a real workhorse space."""

    def test_adapted_white_noise_is_white(self, X, rng):
        check_white_noise(X, rng=rng, samples=30000, rtol=0.06)

    def test_v1_white_noise_is_not(self, rng):
        """random_range routes mass-weighted spaces here for safety."""
        base = v1_space()
        measure = pygeoinf.white_noise_measure(base)
        u = base.basis_vector(3)
        np.random.seed(11)
        squared = np.mean(
            [base.inner_product(measure.sample(), u) ** 2 for _ in range(4000)]
        )
        expected = base.inner_product(u, u)
        # Off by a factor of the metric, not by sampling noise.
        assert squared == pytest.approx(expected**2, rel=0.15)
        assert not (0.8 * expected < squared < 1.2 * expected)


class TestDiracAndRepresenters:
    """DESIGN.md 5.6, against the v1 idiom it was derived from."""

    def test_v1_dirac_becomes_a_linear_functional(self, X, rng):
        base = X.v1_space
        point = POINTS[0]
        f = adapt_form(base.dirac(point), domain=X)

        x = X.random(rng)
        assert f(x) == pytest.approx(base.dirac(point)(x))

    def test_the_representer_matches_v1s_from_dual(self, X):
        """v2's adjoint(1.0) is v1's from_dual(dirac)."""
        base = X.v1_space
        point = POINTS[1]
        f = adapt_form(base.dirac(point), domain=X)
        assert np.allclose(
            X.to_components(f.representer),
            X.to_components(base.dirac_representation(point)),
        )

    def test_the_matrix_is_the_derivative(self, X):
        base = X.v1_space
        point = POINTS[2]
        form = base.dirac(point)
        f = adapt_form(form, domain=X)
        assert np.allclose(f.matrix().ravel(), form.components)
        # And the two readings differ, on this space.
        assert not np.allclose(X.to_components(f.representer), form.components)


class TestAdaptedOperator:
    def test_point_evaluation_adjoint_identity(self, X, rng):
        """A hand-written adjoint on a mass-weighted space, verified."""
        A = adapt_operator(X.v1_space.point_evaluation_operator(POINTS), domain=X)
        check_operator(A, rng=rng)

    def test_forward_values_match_v1(self, X, rng):
        v1_op = X.v1_space.point_evaluation_operator(POINTS)
        A = adapt_operator(v1_op, domain=X)
        x = X.random(rng)
        assert np.allclose(A(x), v1_op(x))

    def test_adjoint_values_match_v1(self, X, rng):
        v1_op = X.v1_space.point_evaluation_operator(POINTS)
        A = adapt_operator(v1_op, domain=X)
        y = A.codomain.random(rng)
        assert np.allclose(
            X.to_components(A.adjoint(y)), X.to_components(v1_op.adjoint(y))
        )

    def test_adjoint_of_a_covector_is_a_dirac_representer(self, X):
        """Because the rows of the operator are Dirac derivative components."""
        base = X.v1_space
        A = adapt_operator(base.point_evaluation_operator(POINTS), domain=X)
        e0 = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(
            X.to_components(A.adjoint(e0)),
            X.to_components(base.dirac_representation(POINTS[0])),
        )

    def test_invariant_covariance_is_self_adjoint(self, X, rng):
        prior = X.v1_space.sobolev_kernel_gaussian_measure(2.0, 0.3)
        C = adapt_operator(
            prior.covariance, domain=X, codomain=X, traits=Traits.POSITIVE_DEFINITE
        )
        check_operator(C, rng=rng)
        check_traits(C, rng=rng)


class TestBayesianNormalOperator:
    """A whole forward problem assembled with the v2 algebra."""

    @pytest.fixture
    def problem(self, X, rng):
        base = X.v1_space
        A = adapt_operator(base.point_evaluation_operator(POINTS), domain=X)
        prior = base.sobolev_kernel_gaussian_measure(2.0, 0.3)
        Q = adapt_operator(
            prior.covariance, domain=X, codomain=X, traits=Traits.POSITIVE_DEFINITE
        )
        Y = A.codomain
        R = LinearOperator.from_component_matrix(
            Y, Y, 0.04 * np.identity(len(POINTS)), traits=Traits.POSITIVE_DEFINITE
        )
        return X, A, Q, R

    def test_traits_are_recovered_structurally(self, problem, rng):
        """A Q A* + R comes out positive definite with nothing asserted."""
        _, A, Q, R = problem
        normal = A @ Q @ A.adjoint + R
        assert Traits.SELF_ADJOINT & normal.traits
        assert Traits.POSITIVE_DEFINITE & normal.traits
        check_traits(normal, rng=rng)
        check_operator(normal, rng=rng)

    def test_the_matrix_is_symmetric(self, problem):
        """So it can be handed to a symmetric solver, with no galerkin flag."""
        _, A, Q, R = problem
        normal = A @ Q @ A.adjoint + R
        matrix = normal.matrix()  # form="auto", picked from the traits
        assert np.allclose(matrix, matrix.T)
        assert np.linalg.eigvalsh(matrix).min() > 0.0

    def test_it_agrees_with_v1_numerically(self, problem, rng):
        """The same operator assembled with v1's algebra, applied to the same data."""
        X, A, Q, R = problem
        base = X.v1_space
        v1_A = base.point_evaluation_operator(POINTS)
        v1_Q = base.sobolev_kernel_gaussian_measure(2.0, 0.3).covariance
        v1_R = 0.04 * pygeoinf.EuclideanSpace(len(POINTS)).identity_operator()
        v1_normal = v1_A @ v1_Q @ v1_A.adjoint + v1_R

        normal = A @ Q @ A.adjoint + R
        for _ in range(3):
            y = A.codomain.random(rng)
            assert np.allclose(normal(y), v1_normal(y))

    def test_posterior_mean_matches_v1(self, problem, rng):
        """One full solve, end to end, against v1's own answer."""
        X, A, Q, R = problem
        base = X.v1_space
        normal = A @ Q @ A.adjoint + R

        truth = X.random(rng)
        data = A(truth)

        # v2: solve the normal system densely, then map back through Q A*.
        weights = np.linalg.solve(normal.matrix(form="components"), data)
        posterior_mean = (Q @ A.adjoint)(weights)

        # v1: the same equations, assembled with v1 objects throughout.
        v1_A = base.point_evaluation_operator(POINTS)
        v1_Q = base.sobolev_kernel_gaussian_measure(2.0, 0.3).covariance
        v1_normal = (
            v1_A @ v1_Q @ v1_A.adjoint
            + 0.04 * pygeoinf.EuclideanSpace(len(POINTS)).identity_operator()
        )
        v1_weights = np.linalg.solve(v1_normal.matrix(dense=True), data)
        v1_mean = v1_Q(v1_A.adjoint(v1_weights))

        assert np.allclose(
            X.to_components(posterior_mean), base.to_components(v1_mean), atol=1e-10
        )


class TestSphere:
    """The same, on the space the library is really for."""

    @pytest.fixture
    def sphere(self):
        pytest.importorskip("pyshtools")
        from pygeoinf.symmetric_space.sphere import Sobolev

        return adapt_space(Sobolev(8, 2.0, 0.2))

    def test_axioms(self, sphere, rng):
        check_space(sphere, rng=rng)
        check_coordinates(sphere, rng=rng)

    def test_white_noise_is_white(self, sphere, rng):
        check_white_noise(sphere, rng=rng, samples=8000, rtol=0.12)

    def test_point_evaluation_adjoint(self, sphere, rng):
        points = sphere.v1_space.random_points(5)
        A = adapt_operator(
            sphere.v1_space.point_evaluation_operator(points), domain=sphere
        )
        check_operator(A, rng=rng)

    def test_normal_operator(self, sphere, rng):
        base = sphere.v1_space
        points = base.random_points(5)
        A = adapt_operator(base.point_evaluation_operator(points), domain=sphere)
        Q = adapt_operator(
            base.sobolev_kernel_gaussian_measure(2.0, 0.3).covariance,
            domain=sphere,
            codomain=sphere,
            traits=Traits.POSITIVE_DEFINITE,
        )
        Y = A.codomain
        R = LinearOperator.from_component_matrix(
            Y, Y, 0.01 * np.identity(len(points)), traits=Traits.POSITIVE_DEFINITE
        )
        normal = A @ Q @ A.adjoint + R
        assert Traits.POSITIVE_DEFINITE & normal.traits
        check_traits(normal, rng=rng)


class TestSolversOnAdaptedSpaces:
    """M3 acceptance: coordinate-free CG on a real Sobolev space, against v1."""

    @pytest.fixture
    def problem(self, X, rng):
        from pygeoinf2.numerics import CGSolver  # noqa: F401

        base = X.v1_space
        A = adapt_operator(base.point_evaluation_operator(POINTS), domain=X)
        Q = adapt_operator(
            base.sobolev_kernel_gaussian_measure(2.0, 0.3).covariance,
            domain=X,
            codomain=X,
            traits=Traits.POSITIVE_DEFINITE,
        )
        Y = A.codomain
        R = LinearOperator.from_component_matrix(
            Y, Y, 0.04 * np.identity(len(POINTS)), traits=Traits.POSITIVE_DEFINITE
        )
        return X, A, Q, R, A @ Q @ A.adjoint + R

    def test_cg_is_admitted_by_its_declared_precondition(self, problem):
        """No assertion needed: the traits were earned structurally."""
        from pygeoinf2.numerics import CGSolver, InverseOperator

        *_, normal = problem
        assert Traits.POSITIVE_DEFINITE & normal.traits
        assert isinstance(CGSolver()(normal), InverseOperator)

    def test_cg_matches_a_dense_solve(self, problem, rng):
        from pygeoinf2.numerics import CGSolver

        _, A, _, _, normal = problem
        b = A.codomain.random(rng)
        result = CGSolver(rtol=1e-12)(normal).solve(b)
        exact = np.linalg.solve(normal.matrix(form="components"), b)
        assert np.allclose(result.solution, exact, atol=1e-9)

    def test_cg_matches_v1s_own_solver(self, problem, rng):
        """The same system, solved by v1's CG and by v2's, to tolerance."""
        from pygeoinf2.numerics import CGSolver

        X, A, Q, _, normal = problem
        base = X.v1_space

        v1_A = base.point_evaluation_operator(POINTS)
        v1_Q = base.sobolev_kernel_gaussian_measure(2.0, 0.3).covariance
        v1_normal = (
            v1_A @ v1_Q @ v1_A.adjoint
            + 0.04 * pygeoinf.EuclideanSpace(len(POINTS)).identity_operator()
        )

        b = A.codomain.random(rng)
        v2_solution = CGSolver(rtol=1e-12)(normal).solve(b).solution
        v1_solution = pygeoinf.CGSolver(rtol=1e-12)(v1_normal)(b)
        assert np.allclose(v2_solution, v1_solution, atol=1e-8)

    def test_a_model_space_solve_on_the_sobolev_space_itself(self, X, rng):
        """CG in the model space, where the mass matrix actually bites.

        The operator acts on the Sobolev space, so every inner product inside
        CG carries the metric. Nothing here touches a component array.
        """
        from pygeoinf2.numerics import CGSolver

        base = X.v1_space
        Q = adapt_operator(
            base.sobolev_kernel_gaussian_measure(2.0, 0.3).covariance,
            domain=X,
            codomain=X,
            traits=Traits.POSITIVE_DEFINITE,
        )
        shifted = Q + 0.5 * LinearOperator.identity(X)
        assert Traits.POSITIVE_DEFINITE & shifted.traits

        b = X.random(rng)
        solution = CGSolver(rtol=1e-12)(shifted).solve(b).solution
        residual = X.norm(X.subtract(shifted(solution), b))
        assert residual < 1e-8 * X.norm(b)

    def test_the_posterior_mean_via_cg_matches_the_dense_route(self, problem, rng):
        from pygeoinf2.numerics import CGSolver

        X, A, Q, _, normal = problem
        truth = X.random(rng)
        data = A(truth)

        via_cg = (Q @ A.adjoint)(CGSolver(rtol=1e-12)(normal).solve(data).solution)
        via_dense = (Q @ A.adjoint)(
            np.linalg.solve(normal.matrix(form="components"), data)
        )
        assert np.allclose(
            X.to_components(via_cg), X.to_components(via_dense), atol=1e-9
        )


class TestMeasuresOnAdaptedSpaces:
    """M4 acceptance: sampling, pushforward and moments against v1."""

    @pytest.fixture
    def prior(self, X):
        """A v1 invariant covariance, wrapped and made into a v2 measure."""
        from pygeoinf2.probability import GaussianMeasure

        base = X.v1_space
        v1_measure = base.sobolev_kernel_gaussian_measure(2.0, 0.3)
        covariance = adapt_operator(
            v1_measure.covariance,
            domain=X,
            codomain=X,
            traits=Traits.POSITIVE_DEFINITE,
        )
        # An InvariantGaussianMeasure exposes no factor, but its covariance is
        # spectrally diagonal and carries a functional calculus, so the square
        # root is one: C == L L* with L self-adjoint. The factor's domain is the
        # Sobolev space itself, so sampling draws white noise with respect to
        # its inner product -- exactly the case DESIGN.md section 9 is about.
        factor = adapt_operator(v1_measure.covariance.sqrt, domain=X, codomain=X)
        return v1_measure, GaussianMeasure(X, covariance_factor=factor), covariance

    def test_the_covariance_matches_v1(self, X, prior, rng):
        v1_measure, mu, _ = prior
        x = X.random(rng)
        assert np.allclose(mu.covariance(x), v1_measure.covariance(x))

    def test_sampled_moments_match_the_declared_covariance(self, X, prior, rng):
        from pygeoinf2.testing import check_measure

        _, mu, _ = prior
        check_measure(mu, rng=rng, samples=6000, rtol=0.12, directions=2)

    def test_v1_agrees_on_this_measure(self, X, prior, rng):
        """Where v1 is right, v2 must match it, not merely be self-consistent.

        The symmetric-space measures compensate for the metric separately, with
        their own 1/sqrt(metric_values) factor, so this path in v1 is correct
        and is a fair reference.
        """
        v1_measure, mu, _ = prior
        base = X.v1_space
        u = base.basis_vector(2)

        np.random.seed(7)
        v1_moment = np.mean(
            [base.inner_product(v1_measure.sample(), u) ** 2 for _ in range(4000)]
        )
        v2_moment = np.mean(
            [X.inner_product(mu.sample(rng), u) ** 2 for _ in range(4000)]
        )
        exact = X.inner_product(mu.covariance(u), u)

        assert v1_moment == pytest.approx(exact, rel=0.12)
        assert v2_moment == pytest.approx(exact, rel=0.12)

    def test_pushforward_to_the_data_space(self, X, prior, rng):
        """A C A* on a real forward operator, recognised and verified."""
        from pygeoinf2.testing import check_measure, check_traits

        _, mu, _ = prior
        A = adapt_operator(X.v1_space.point_evaluation_operator(POINTS), domain=X)
        data_measure = A @ mu

        assert Traits.POSITIVE_SEMIDEFINITE & data_measure.covariance.traits
        check_traits(data_measure.covariance, rng=rng)
        check_measure(data_measure, rng=rng, samples=8000, rtol=0.12)

    def test_the_pushforward_matches_v1s_affine_mapping(self, X, prior, rng):
        v1_measure, mu, _ = prior
        base = X.v1_space
        v1_A = base.point_evaluation_operator(POINTS)

        v1_pushed = v1_measure.affine_mapping(operator=v1_A)
        v2_pushed = adapt_operator(v1_A, domain=X) @ mu

        y = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(v2_pushed.covariance(y), v1_pushed.covariance(y))

    def test_a_noisy_data_measure(self, X, prior, rng):
        """mu_d = A C A* + R, the object a Bayesian inversion starts from."""
        from pygeoinf2.probability import GaussianMeasure

        _, mu, _ = prior
        A = adapt_operator(X.v1_space.point_evaluation_operator(POINTS), domain=X)
        Y = A.codomain
        noise = GaussianMeasure.from_standard_deviation(Y, 0.2)

        total = (A @ mu) + noise
        assert Traits.POSITIVE_DEFINITE & total.covariance.traits
        assert total.can_sample
