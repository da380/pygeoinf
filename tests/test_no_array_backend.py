"""
The library over a backend whose vectors are not arrays.

Every other space in the suite, the opaque doubles included, holds an
ndarray somewhere inside its vectors, so a library path that reaches for
``np.asarray``, a shape, a length, an index or array arithmetic can work by
accident. :class:`ListSpace` removes the accident: its vectors are plain
Python lists wrapped in an object with no array protocol and no operators,
and its Gram matrix is not diagonal. Anything that treats a vector as an
array fails here, loudly.

Every assertion compares the library's answer with the same calculation
done by NumPy on the extracted lists. The helpers that do the extraction,
``arr`` and ``vec``, are the backend's own and the assertion's; the library
never sees them.

Every library path this file exercises is coordinate-free; what is not
refuses by name, and those refusals are tested too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.optimize import brentq, minimize
from scipy.stats import chi2

import pygeoinf as gi
from pygeoinf import (
    AffineSubspace,
    BackusGilbert,
    BackusGilbertParker,
    Ball,
    Ellipsoid,
    ForwardProblem,
    Functional,
    GaussianMeasure,
    LaplaceApproximation,
    LeastSquares,
    LinearForwardProblem,
    LinearGaussianInversion,
    LinearOperator,
    MaximumAPosteriori,
    Minimization,
    MinimumNorm,
    NonlinearLeastSquares,
    Occam,
    Operator,
    OrthogonalProjector,
    Traits,
)
from pygeoinf.numerics import (
    LBFGS,
    AugmentedLagrangian,
    CGSolver,
    CholeskySolver,
    GMRESSolver,
    LSQRSolver,
    LUSolver,
    MinResSolver,
    NewtonCG,
    NonlinearCG,
    SteepestDescent,
    TrustRegionNewton,
    log_determinant,
    random_diagonal,
    random_trace,
)
from pygeoinf.testing import (
    check_derivative,
    check_gradient,
    check_measure,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)

from .doubles import ListSpace, ListVector

NX, NY, NP, NQ = 6, 4, 3, 2
SPD = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE


# --------------------------------------------------------------------- #
#          The backend's side: NumPy on lists, never on vectors          #
# --------------------------------------------------------------------- #


def arr(v: ListVector) -> np.ndarray:
    """The entries as an array. For the backend's callables and assertions."""
    return np.asarray(v.values, dtype=float)


def vec(a) -> ListVector:
    """A vector from an array of entries. The backend's side of the fence."""
    return ListVector([float(t) for t in np.ravel(a)])


def gram(space: ListSpace) -> np.ndarray:
    return np.asarray(space.gram, dtype=float)


def spd_matrix(generator, n: int) -> np.ndarray:
    root = np.eye(n) + 0.3 * np.tril(generator.standard_normal((n, n)), -1)
    return root @ root.T


def make_space(generator, n: int) -> ListSpace:
    return ListSpace(spd_matrix(generator, n).tolist())


def matrix_operator(
    X: ListSpace, Y: ListSpace, M: np.ndarray, /, *, traits: Traits = Traits.NONE
) -> LinearOperator:
    """``A`` with component matrix ``M``, and its metric adjoint written by hand."""
    M = np.asarray(M, dtype=float)
    adjoint_matrix = np.linalg.solve(gram(X), M.T @ gram(Y))
    return LinearOperator.from_callables(
        X,
        Y,
        lambda x: vec(M @ arr(x)),
        adjoint=lambda y: vec(adjoint_matrix @ arr(y)),
        traits=traits,
    )


def adjoint_matrix(X: ListSpace, Y: ListSpace, M: np.ndarray) -> np.ndarray:
    return np.linalg.solve(gram(X), np.asarray(M).T @ gram(Y))


def spd_operator(X: ListSpace, S: np.ndarray, /) -> LinearOperator:
    """The self-adjoint positive operator whose Galerkin matrix is ``S``."""
    return matrix_operator(X, X, np.linalg.solve(gram(X), S), traits=SPD)


def gaussian(
    X: ListSpace, F: np.ndarray, /, *, expectation: ListVector | None = None
) -> GaussianMeasure:
    """``N(m, L L*)`` with ``L`` of component matrix ``F``, all from callables."""
    G = gram(X)
    F = np.asarray(F, dtype=float)
    C = F @ np.linalg.solve(G, F.T @ G)
    return GaussianMeasure(
        X,
        expectation=expectation,
        covariance=matrix_operator(X, X, C, traits=SPD),
        covariance_factor=matrix_operator(X, X, F),
        precision=matrix_operator(X, X, np.linalg.inv(C), traits=SPD),
        precision_factor=matrix_operator(X, X, np.linalg.inv(F)),
    )


def covariance_matrix(X: ListSpace, F: np.ndarray) -> np.ndarray:
    G = gram(X)
    return F @ np.linalg.solve(G, F.T @ G)


@dataclass
class Setting:
    X: ListSpace
    Y: ListSpace
    P: ListSpace
    Q: ListSpace
    M: np.ndarray
    Tm: np.ndarray
    Sm: np.ndarray
    F_noise: np.ndarray
    F_prior: np.ndarray
    m0: np.ndarray
    truth: np.ndarray
    data: np.ndarray

    @property
    def A(self) -> LinearOperator:
        return matrix_operator(self.X, self.Y, self.M)

    @property
    def T(self) -> LinearOperator:
        return matrix_operator(self.X, self.P, self.Tm)

    @property
    def S(self) -> LinearOperator:
        return matrix_operator(self.P, self.Q, self.Sm)

    @property
    def noise(self) -> GaussianMeasure:
        return gaussian(self.Y, self.F_noise)

    @property
    def prior(self) -> GaussianMeasure:
        return gaussian(self.X, self.F_prior, expectation=vec(self.m0))

    @property
    def R(self) -> np.ndarray:
        return covariance_matrix(self.Y, self.F_noise)

    @property
    def C(self) -> np.ndarray:
        return covariance_matrix(self.X, self.F_prior)

    @property
    def A_star(self) -> np.ndarray:
        return adjoint_matrix(self.X, self.Y, self.M)

    @property
    def problem(self) -> LinearForwardProblem:
        return LinearForwardProblem(self.A, error=self.noise)

    @property
    def nonlinear_problem(self) -> ForwardProblem:
        """The same linear map, presented as a nonlinear operator."""
        A = self.A
        return ForwardProblem(
            Operator.from_callables(self.X, self.Y, A, derivative=lambda m: A),
            error=self.noise,
        )

    @property
    def d(self) -> ListVector:
        return vec(self.data)

    # Closed forms, in components, of what the estimators compute.

    def tikhonov(self, damping: float, /, *, reference=None) -> np.ndarray:
        """``argmin |A u - d|^2_R + t |u - m_ref|^2``."""
        m_ref = np.zeros(NX) if reference is None else reference
        lhs = self.A_star @ np.linalg.solve(self.R, self.M) + damping * np.eye(NX)
        rhs = self.A_star @ np.linalg.solve(self.R, self.data) + damping * m_ref
        return np.linalg.solve(lhs, rhs)

    def chi_squared(self, u: np.ndarray) -> float:
        r = self.data - self.M @ u
        return float(r @ gram(self.Y) @ np.linalg.solve(self.R, r))

    def posterior(self) -> tuple[np.ndarray, np.ndarray]:
        precision = self.A_star @ np.linalg.solve(self.R, self.M) + np.linalg.inv(
            self.C
        )
        cov = np.linalg.inv(precision)
        mean = cov @ (
            self.A_star @ np.linalg.solve(self.R, self.data)
            + np.linalg.solve(self.C, self.m0)
        )
        return mean, cov


@pytest.fixture
def setting(rng) -> Setting:
    generator = np.random.default_rng(20261028)
    X, Y, P, Q = (make_space(generator, n) for n in (NX, NY, NP, NQ))
    M = generator.standard_normal((NY, NX))
    Tm = generator.standard_normal((NP, NX))
    Sm = generator.standard_normal((NQ, NP))
    F_noise = 0.3 * (np.eye(NY) + 0.2 * generator.standard_normal((NY, NY)))
    F_prior = np.eye(NX) + 0.2 * generator.standard_normal((NX, NX))
    m0 = 0.5 * generator.standard_normal(NX)
    truth = rng.standard_normal(NX)
    s = Setting(X, Y, P, Q, M, Tm, Sm, F_noise, F_prior, m0, truth, np.zeros(NY))
    s.data = M @ truth + arr(s.noise.sample(rng=rng))
    return s


def assert_close(space: ListSpace, v: ListVector, expected, /, *, rel=1e-8) -> None:
    """``v == expected`` in the space's own norm, relative to the expected size."""
    expected = np.asarray(expected, dtype=float)
    G = gram(space)
    diff = arr(v) - expected
    scale = max(float(np.sqrt(expected @ G @ expected)), 1e-300)
    assert float(np.sqrt(diff @ G @ diff)) <= rel * scale, (arr(v), expected)


# --------------------------------------------------------------------- #
#                          1. The double itself                         #
# --------------------------------------------------------------------- #


class TestTheDoubleRefusesArrays:
    def test_the_vector_cannot_be_treated_as_an_array(self):
        """If this passes trivially, the rest of the file proves nothing."""
        a, b = ListVector([1.0, 2.0]), ListVector([3.0, 4.0])
        assert np.asarray(a).dtype == object and np.asarray(a).shape == ()
        with pytest.raises(TypeError):
            np.asarray(a, dtype=float)
        with pytest.raises(TypeError):
            a + b
        with pytest.raises(TypeError):
            2.0 * a
        with pytest.raises(TypeError):
            -a
        with pytest.raises(TypeError):
            len(a)
        with pytest.raises(TypeError):
            a[0]
        with pytest.raises(TypeError):
            iter(a)
        assert not hasattr(a, "shape") and not hasattr(a, "copy")

    def test_the_space_has_no_coordinate_map(self, setting):
        from pygeoinf.algebra.spaces import CoordinateSpace

        assert not isinstance(setting.X, CoordinateSpace)
        assert not hasattr(setting.X, "to_components")

    def test_inner_product_is_the_gram_form(self, setting, rng):
        X = setting.X
        x, y = X.random(rng=rng), X.random(rng=rng)
        assert X.inner_product(x, y) == pytest.approx(arr(x) @ gram(X) @ arr(y))


class TestTheChecks:
    def test_check_space(self, setting, rng):
        generator = np.random.default_rng(20261028)
        report = check_space(
            setting.X, rng=rng, rebuild=lambda: make_space(generator, NX)
        )
        assert report.worst <= 1e-10

    def test_check_white_noise_with_a_measure(self, setting, rng):
        report = check_white_noise(
            setting.X, rng=rng, measure=setting.prior, samples=20000, rtol=0.06
        )
        assert report.worst <= 0.06

    def test_check_operator_and_traits(self, setting, rng):
        check_operator(setting.A, rng=rng, measure=setting.prior)
        H = spd_operator(setting.X, spd_matrix(np.random.default_rng(3), NX))
        check_operator(H, rng=rng)
        check_traits(H, rng=rng)
        check_traits(setting.A @ setting.A.adjoint, rng=rng)

    def test_check_derivative_on_a_nonlinear_operator(self, setting, rng):
        X, Y, M = setting.X, setting.Y, setting.M
        w = np.ones(NY)
        G = gram(X)

        def value(x):
            a = arr(x)
            return vec(M @ a + float(a @ G @ a) * w)

        def derivative(x):
            return matrix_operator(X, Y, M + 2.0 * np.outer(w, G @ arr(x)))

        F = Operator.from_callables(X, Y, value, derivative=derivative)
        check_derivative(F, X.random(rng=rng), rng=rng)

    def test_check_gradient_on_a_quadratic(self, setting, rng):
        X = setting.X
        S = spd_matrix(np.random.default_rng(5), NX)
        H = spd_operator(X, S)
        b = X.random(rng=rng)

        phi = Functional.from_callables(
            X,
            lambda x: 0.5 * X.inner_product(x, H(x)) - X.inner_product(b, x),
            gradient=lambda x: X.subtract(H(x), b),
        )
        check_gradient(phi, X.random(rng=rng), rng=rng)


# --------------------------------------------------------------------- #
#                            2. Operators                               #
# --------------------------------------------------------------------- #


class TestOperators:
    def test_algebra_matches_the_matrices(self, setting, rng):
        X, Y, A = setting.X, setting.Y, setting.A
        B = matrix_operator(X, Y, np.ones((NY, NX)))
        N = matrix_operator(Y, X, np.eye(NX, NY))
        x = X.random(rng=rng)

        combined = N @ (2.0 * A + B) - LinearOperator.identity(X)
        expected = np.eye(NX, NY) @ (2.0 * setting.M + np.ones((NY, NX))) - np.eye(NX)
        assert_close(X, combined(x), expected @ arr(x))
        y = Y.random(rng=rng)
        assert_close(X, A.adjoint(y), setting.A_star @ arr(y))
        assert_close(
            X,
            combined.adjoint(x),
            adjoint_matrix(X, X, expected) @ arr(x),
        )
        check_operator(combined, rng=rng)

    def test_assembled_is_refused_without_coordinates(self, setting):
        with pytest.raises(TypeError, match="coordinate"):
            setting.A.assembled()

    def test_probed_adjoint_is_refused_without_coordinates(self, setting, rng):
        X, Y = setting.X, setting.Y
        A = LinearOperator.from_callables(X, Y, lambda x: vec(setting.M @ arr(x)))
        with pytest.raises(TypeError, match="coordinate"):
            A.with_probed_adjoint().adjoint(Y.random(rng=rng))

    def test_the_adjoint_of_a_callable_without_one_is_refused(self, setting, rng):
        X, Y = setting.X, setting.Y
        A = LinearOperator.from_callables(X, Y, lambda x: vec(setting.M @ arr(x)))
        with pytest.raises(NotImplementedError):
            A.adjoint(Y.random(rng=rng))


# --------------------------------------------------------------------- #
#                              3. Solvers                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def spd_system(setting, rng):
    X = setting.X
    S = spd_matrix(np.random.default_rng(7), NX)
    H = spd_operator(X, S)
    b = X.random(rng=rng)
    expected = np.linalg.solve(np.linalg.solve(gram(X), S), arr(b))
    return X, H, b, expected


class TestSolvers:
    @pytest.mark.parametrize("solver", [CGSolver, MinResSolver, GMRESSolver])
    def test_krylov_solvers(self, spd_system, solver):
        X, H, b, expected = spd_system
        inverse = solver(rtol=1e-12)(H)
        assert_close(X, inverse(b), expected, rel=1e-8)
        result = inverse.solve(b)
        assert result.converged
        assert_close(X, result.solution, expected, rel=1e-8)

    def test_gmres_on_a_non_self_adjoint_operator(self, setting, rng):
        X = setting.X
        K = np.eye(NX) + 0.3 * np.random.default_rng(9).standard_normal((NX, NX))
        A = matrix_operator(X, X, K)
        b = X.random(rng=rng)
        assert_close(X, GMRESSolver(rtol=1e-12)(A)(b), np.linalg.solve(K, arr(b)))

    def test_lsqr_gives_the_minimum_norm_solution(self, setting, rng):
        X, Y, A = setting.X, setting.Y, setting.A
        y = Y.random(rng=rng)
        AAs = setting.M @ setting.A_star
        expected = setting.A_star @ np.linalg.solve(AAs, arr(y))
        assert_close(X, LSQRSolver(rtol=1e-12)(A)(y), expected, rel=1e-7)

    @pytest.mark.parametrize("solver", [CholeskySolver, LUSolver])
    def test_direct_solvers_are_refused(self, spd_system, solver):
        X, H, b, expected = spd_system
        with pytest.raises(TypeError, match="coordinate"):
            solver()(H)


# --------------------------------------------------------------------- #
#                         4. Gaussian measures                          #
# --------------------------------------------------------------------- #


class TestGaussianMeasures:
    def test_moments_and_samples(self, setting, rng):
        X, mu = setting.X, setting.prior
        u = X.random(rng=rng)
        assert_close(X, mu.expectation, setting.m0)
        assert_close(X, mu.covariance(u), setting.C @ arr(u))
        assert_close(X, mu.precision(u), np.linalg.solve(setting.C, arr(u)))
        assert isinstance(mu.sample(rng=rng), ListVector)
        check_measure(mu, rng=rng, samples=8000, rtol=0.15)

    def test_push_forward_through_a_callable_operator(self, setting, rng):
        Y, A = setting.Y, setting.A
        nu = setting.prior.push_forward(A)
        v = Y.random(rng=rng)
        assert_close(Y, nu.expectation, setting.M @ setting.m0)
        assert_close(
            Y, nu.covariance(v), setting.M @ setting.C @ setting.A_star @ arr(v)
        )
        assert isinstance(nu.sample(rng=rng), ListVector)

    def test_credible_set(self, setting, rng):
        X, mu = setting.X, setting.prior
        level = 0.9
        region = mu.credible_set(level=level)
        G, Cinv = gram(X), np.linalg.inv(setting.C)
        radius2 = chi2.ppf(level, NX)
        for _ in range(6):
            x = X.axpy(
                rng.uniform(0.5, 2.5), mu.sample(rng=rng), X.copy(mu.expectation)
            )
            d = arr(x) - setting.m0
            assert region.contains(x) == (d @ G @ Cinv @ d <= radius2)
        q = X.random(rng=rng)
        expected = arr(q) @ G @ setting.m0 + np.sqrt(
            radius2 * arr(q) @ G @ setting.C @ arr(q)
        )
        assert region.support_function()(q) == pytest.approx(expected, rel=1e-8)

    def test_ambient_ball(self, setting, rng):
        X, mu = setting.X, setting.prior
        ball = mu.ambient_ball(level=0.9, rng=rng)
        assert_close(X, ball.center, setting.m0)
        # The radius is a quantile of sum lambda_i Z_i^2 over the covariance's
        # eigenvalues in the space's metric; bracket it by simulation.
        G = gram(X)
        eigenvalues = np.linalg.eigvalsh(
            np.linalg.cholesky(G).T @ setting.C @ np.linalg.inv(np.linalg.cholesky(G).T)
        )
        draws = (rng.standard_normal((200000, NX)) ** 2) @ eigenvalues
        assert ball.radius == pytest.approx(np.sqrt(np.quantile(draws, 0.9)), rel=0.02)


# --------------------------------------------------------------------- #
#                            5. Estimators                              #
# --------------------------------------------------------------------- #


class TestLinearEstimators:
    def test_least_squares(self, setting):
        X, P = setting.X, setting.P
        estimator = LeastSquares(setting.problem, damping=0.7)
        expected = setting.tikhonov(0.7)
        assert_close(X, estimator(setting.d), expected, rel=1e-7)
        pushed = estimator.push_forward(setting.T)
        assert_close(P, pushed(setting.d), setting.Tm @ expected, rel=1e-7)

    def test_minimum_norm_with_an_error_measure(self, setting):
        X, P = setting.X, setting.P
        estimator = MinimumNorm(setting.problem, level=0.9, rtol=1e-10)
        u = estimator(setting.d)
        target = chi2.ppf(0.9, NY)
        assert setting.chi_squared(arr(u)) == pytest.approx(target, rel=1e-6)
        damping = estimator.for_data(setting.d).damping
        assert_close(X, u, setting.tikhonov(damping), rel=1e-7)
        pushed = estimator.push_forward(setting.T)
        assert_close(P, pushed(setting.d), setting.Tm @ arr(u), rel=1e-7)

    def test_minimum_norm_with_exact_data(self, setting):
        X, P = setting.X, setting.P
        estimator = MinimumNorm(LinearForwardProblem(setting.A))
        expected = setting.A_star @ np.linalg.solve(
            setting.M @ setting.A_star, setting.data
        )
        assert_close(X, estimator(setting.d), expected, rel=1e-7)
        pushed = estimator.push_forward(setting.T)
        assert_close(P, pushed(setting.d), setting.Tm @ expected, rel=1e-7)

    def test_linear_gaussian_inversion(self, setting, rng):
        X, P = setting.X, setting.P
        inversion = LinearGaussianInversion(setting.problem, setting.prior)
        posterior = inversion(setting.d)
        mean, cov = setting.posterior()
        u = X.random(rng=rng)
        assert_close(X, posterior.expectation, mean, rel=1e-7)
        assert_close(X, posterior.covariance(u), cov @ arr(u), rel=1e-7)
        assert isinstance(posterior.sample(rng=rng), ListVector)
        check_measure(posterior, rng=rng, samples=1500, rtol=0.35)
        pushed = inversion.push_forward(setting.T)(setting.d)
        v = P.random(rng=rng)
        assert_close(P, pushed.expectation, setting.Tm @ mean, rel=1e-7)
        assert_close(
            P,
            pushed.covariance(v),
            setting.Tm @ cov @ adjoint_matrix(X, P, setting.Tm) @ arr(v),
            rel=1e-7,
        )

    def test_backus_gilbert(self, setting, rng):
        P, Q = setting.P, setting.Q
        alpha = 0.5
        estimator = BackusGilbert(setting.problem, setting.T, alpha=alpha)
        Xc = (
            setting.Tm
            @ setting.A_star
            @ np.linalg.inv(setting.M @ setting.A_star + alpha * setting.R)
        )
        assert_close(P, estimator(setting.d), Xc @ setting.data, rel=1e-7)
        m = setting.X.random(rng=rng)
        assert_close(P, estimator.resolution()(m), Xc @ setting.M @ arr(m), rel=1e-7)
        pushed = estimator.push_forward(setting.S)
        assert_close(Q, pushed(setting.d), setting.Sm @ Xc @ setting.data, rel=1e-7)


def feasible_support(setting, direction, *, radius, epsilon):
    """``max (q, T m)_P`` over ``|m| <= rho`` and ``|A m - d|_Y <= eps``, by SLSQP."""
    GX, GY, GP = gram(setting.X), gram(setting.Y), gram(setting.P)
    c = (setting.Tm.T @ GP @ direction).astype(float)
    constraints = [
        {"type": "ineq", "fun": lambda m: radius**2 - m @ GX @ m},
        {
            "type": "ineq",
            "fun": lambda m: epsilon**2
            - (setting.M @ m - setting.data) @ GY @ (setting.M @ m - setting.data),
        },
    ]
    best = None
    for start in (np.zeros(NX), setting.truth * 0.5):
        result = minimize(
            lambda m: -c @ m,
            start,
            jac=lambda m: -c,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-14, "maxiter": 1000},
        )
        if best is None or -result.fun > best:
            best = -result.fun
    return best


@pytest.fixture
def feasible_setting(setting):
    X, Y = setting.X, setting.Y
    radius = 1.2 * float(np.sqrt(setting.truth @ gram(X) @ setting.truth))
    residual = setting.data - setting.M @ setting.truth
    epsilon = 1.5 * float(np.sqrt(residual @ gram(Y) @ residual))
    estimator = BackusGilbertParker(
        setting.problem,
        setting.T,
        Ball(X, radius=radius),
        noise=Ball(Y, radius=epsilon),
    )
    return estimator, radius, epsilon


class TestBackusGilbertParker:
    def test_support_fitting_model_and_polytope(self, setting, feasible_setting, rng):
        X, Y, P = setting.X, setting.Y, setting.P
        estimator, radius, epsilon = feasible_setting
        feasible = estimator(setting.d)
        assert feasible.form == "quadratic"
        assert not feasible.is_empty()

        fitting = feasible.fitting_model()
        assert float(np.sqrt(arr(fitting) @ gram(X) @ arr(fitting))) <= radius * (
            1 + 1e-8
        )
        misfit = setting.M @ arr(fitting) - setting.data
        assert float(np.sqrt(misfit @ gram(Y) @ misfit)) <= epsilon * (1 + 1e-6)

        directions = [P.random(rng=rng) for _ in range(3)]
        for q in directions:
            expected = feasible_support(setting, arr(q), radius=radius, epsilon=epsilon)
            assert feasible.support(q) == pytest.approx(expected, rel=1e-6)
        polytope = feasible.polytope(directions)
        assert polytope.contains(vec(setting.Tm @ setting.truth))

    def test_the_membership_side_refuses_by_name(self, setting, feasible_setting):
        """Membership forms matrices of the property's dimension, which needs
        coordinates on the property space; the refusal names that."""
        estimator, radius, epsilon = feasible_setting
        feasible = estimator(setting.d)
        with pytest.raises(TypeError, match="coordinate map on the property space"):
            feasible.contains(vec(setting.Tm @ setting.truth))

    def test_with_the_credible_ball_from_the_error_measure(self, setting, rng):
        """The ball is hardened by sampling, which is coordinate-free; the
        feasible set built on it is not."""
        X, P = setting.X, setting.P
        radius = 1.5 * float(np.sqrt(setting.truth @ gram(X) @ setting.truth))
        estimator = BackusGilbertParker(
            setting.problem, setting.T, Ball(X, radius=radius), level=0.9
        )
        assert isinstance(estimator.noise, Ball)
        epsilon = estimator.noise.radius
        feasible = estimator(setting.d)
        assert not feasible.is_empty()
        q = P.random(rng=rng)
        expected = feasible_support(setting, arr(q), radius=radius, epsilon=epsilon)
        assert feasible.support(q) == pytest.approx(expected, rel=1e-6)


class TestNonlinearEstimators:
    def test_maximum_a_posteriori(self, setting):
        X, P = setting.X, setting.P
        estimator = MaximumAPosteriori(
            setting.nonlinear_problem, setting.prior, optimizer=LBFGS(gtol=1e-10)
        )
        mean, _ = setting.posterior()
        assert_close(X, estimator(setting.d), mean, rel=1e-6)
        pushed = estimator.push_forward(setting.T)
        assert_close(P, pushed(setting.d), setting.Tm @ mean, rel=1e-6)

    def test_laplace_approximation(self, setting, rng):
        X, P = setting.X, setting.P
        laplace = LaplaceApproximation.from_problem(
            setting.nonlinear_problem, setting.prior, optimizer=LBFGS(gtol=1e-10)
        )
        measure = laplace(setting.d)
        mean, cov = setting.posterior()
        u = X.random(rng=rng)
        assert_close(X, measure.expectation, mean, rel=1e-6)
        assert_close(X, measure.covariance(u), cov @ arr(u), rel=1e-6)
        # The covariance is an inverse through a solver and carries no
        # factor, and the approximation draws by randomize-then-optimize,
        # which needs none, on any space.
        assert measure.can_sample
        assert np.isfinite(X.norm(measure.sample(rng=rng)))
        pushed = laplace.push_forward(setting.T)(setting.d)
        assert_close(P, pushed.expectation, setting.Tm @ mean, rel=1e-6)

    def test_nonlinear_least_squares(self, setting):
        X, P = setting.X, setting.P
        estimator = NonlinearLeastSquares(
            setting.nonlinear_problem,
            damping=0.7,
            reference=vec(setting.m0),
            optimizer=LBFGS(gtol=1e-10),
        )
        expected = setting.tikhonov(0.7, reference=setting.m0)
        assert_close(X, estimator(setting.d), expected, rel=1e-6)
        pushed = estimator.push_forward(setting.T)
        assert_close(P, pushed(setting.d), setting.Tm @ expected, rel=1e-6)

    def test_occam(self, setting):
        X, P = setting.X, setting.P
        level = 0.9
        estimator = Occam(
            setting.nonlinear_problem,
            level=level,
            reference=vec(setting.m0),
            tolerance=1e-8,
            optimizer=LBFGS(gtol=1e-10),
        )
        u = estimator(setting.d)
        target = chi2.ppf(level, NY)
        assert setting.chi_squared(arr(u)) == pytest.approx(target, rel=1e-5)
        damping = brentq(
            lambda t: setting.chi_squared(setting.tikhonov(t, reference=setting.m0))
            - target,
            1e-8,
            1e4,
        )
        assert_close(X, u, setting.tikhonov(damping, reference=setting.m0), rel=1e-4)
        pushed = estimator.push_forward(setting.T)
        assert_close(P, pushed(setting.d), setting.Tm @ arr(u), rel=1e-6)

    def test_minimization_with_a_hand_written_objective(self, setting):
        X, Y, P = setting.X, setting.Y, setting.P
        S = spd_matrix(np.random.default_rng(11), NX)
        H = spd_operator(X, S)
        A_star = setting.A.adjoint

        def objective(data):
            center = A_star(data)

            def value(m):
                dm = X.subtract(m, center)
                return 0.5 * X.inner_product(dm, H(dm))

            return Functional.from_callables(
                X,
                value,
                gradient=lambda m: H(X.subtract(m, center)),
                hessian=lambda m: H,
            )

        estimator = Minimization(Y, X, objective, optimizer=NewtonCG(gtol=1e-10))
        expected = setting.A_star @ setting.data
        assert_close(X, estimator(setting.d), expected, rel=1e-7)
        pushed = estimator.push_forward(setting.T)
        assert_close(P, pushed(setting.d), setting.Tm @ expected, rel=1e-7)


# --------------------------------------------------------------------- #
#                            6. Optimizers                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def quadratic(setting, rng):
    X = setting.X
    S = spd_matrix(np.random.default_rng(13), NX)
    H = spd_operator(X, S)
    b = X.random(rng=rng)

    phi = Functional.from_callables(
        X,
        lambda x: 0.5 * X.inner_product(x, H(x)) - X.inner_product(b, x),
        gradient=lambda x: X.subtract(H(x), b),
        hessian=lambda x: H,
    )
    expected = np.linalg.solve(np.linalg.solve(gram(X), S), arr(b))
    return X, phi, b, S, expected


class TestOptimizers:
    @pytest.mark.parametrize(
        "optimizer",
        [
            # The line-search methods stop where rounding in the value stops
            # them, which on this problem is a gradient between 1e-10 and
            # 1e-7; the Newton methods reach the arithmetic's own floor.
            LBFGS(gtol=1e-8, ftol=0.0),
            NonlinearCG(gtol=1e-8, ftol=0.0),
            SteepestDescent(gtol=1e-8, ftol=0.0, max_iterations=5000),
            NewtonCG(gtol=1e-10, ftol=0.0),
            TrustRegionNewton(gtol=1e-10, ftol=0.0),
        ],
        ids=lambda o: type(o).__name__,
    )
    def test_unconstrained(self, quadratic, optimizer):
        X, phi, b, S, expected = quadratic
        result = optimizer.minimize(phi, X.zero())
        assert result.converged, result
        assert_close(X, result.minimizer, expected, rel=1e-6)

    def test_augmented_lagrangian_with_a_linear_equality(self, quadratic, rng):
        X, phi, b, S, expected = quadratic
        a = X.random(rng=rng)
        constraint = Functional.from_callables(
            X,
            lambda x: X.inner_product(a, x) - 1.0,
            gradient=lambda x: X.copy(a),
            hessian=lambda x: LinearOperator.zero(X),
        )
        method = AugmentedLagrangian(
            NewtonCG(gtol=1e-10, ftol=0.0),
            equalities=[constraint],
            tolerance=1e-10,
            gtol=1e-10,
        )
        result = method.minimize(phi, X.zero())
        assert result.converged, result
        G = gram(X)
        ga = G @ arr(a)
        kkt = np.block([[S, ga[:, None]], [ga[None, :], np.zeros((1, 1))]])
        solution = np.linalg.solve(kkt, np.concatenate([G @ arr(b), [1.0]]))
        assert_close(X, result.minimizer, solution[:NX], rel=1e-6)


# --------------------------------------------------------------------- #
#                      7. Randomized linear algebra                     #
# --------------------------------------------------------------------- #


class TestRandomized:
    def test_random_trace(self, spd_system, rng):
        X, H, b, expected = spd_system
        S = spd_matrix(np.random.default_rng(7), NX)
        trace = np.trace(np.linalg.solve(gram(X), S))
        estimate = random_trace(H, samples=4000, rng=rng)
        assert abs(estimate.value - trace) <= 4.0 * estimate.standard_error

    def test_random_diagonal_is_refused_without_coordinates(self, spd_system, rng):
        X, H, b, expected = spd_system
        with pytest.raises(TypeError, match="coordinate"):
            random_diagonal(H, samples=10, rng=rng)

    def test_log_determinant_by_stochastic_lanczos(self, spd_system, rng):
        X, H, b, expected = spd_system
        S = spd_matrix(np.random.default_rng(7), NX)
        exact = np.linalg.slogdet(np.linalg.solve(gram(X), S))[1]
        estimate = log_determinant(H, method="stochastic", samples=2000, rng=rng)
        assert abs(estimate.value - exact) <= 4.0 * estimate.standard_error + 1e-6

    def test_kl_divergence_estimate(self, setting, rng):
        X = setting.X
        first = setting.prior
        second = gaussian(X, 1.5 * np.eye(NX))
        C1, C2 = setting.C, 2.25 * np.eye(NX)
        d = setting.m0
        exact = 0.5 * (
            np.trace(np.linalg.solve(C2, C1))
            + d @ gram(X) @ np.linalg.solve(C2, d)
            - NX
            + np.linalg.slogdet(C2)[1]
            - np.linalg.slogdet(C1)[1]
        )
        estimate = first.kl_divergence_estimate(
            second, method="stochastic", samples=2000, rng=rng
        )
        assert abs(estimate.value - exact) <= 4.0 * estimate.standard_error + 1e-6

    def test_kl_divergence_auto_refuses_to_guess(self, setting, rng):
        second = gaussian(setting.X, 1.5 * np.eye(NX))
        with pytest.raises((TypeError, ValueError)):
            setting.prior.kl_divergence_estimate(second, rng=rng)

    def test_hilbert_schmidt_norm_by_hutchinson(self, setting, rng):
        norm = setting.prior.hilbert_schmidt_norm(
            method="stochastic", samples=4000, rng=rng
        )
        exact = np.sqrt(np.trace(setting.C @ setting.C))
        assert norm == pytest.approx(exact, rel=0.15)

    def test_hilbert_schmidt_norm_by_default_is_not_an_estimate(self, setting):
        """Without coordinates there is no exact route, and ``"auto"`` says
        so rather than return an estimate as though it were exact."""
        with pytest.raises(TypeError, match="coordinate"):
            setting.prior.hilbert_schmidt_norm(
                samples=400, rng=np.random.default_rng(3)
            )


# --------------------------------------------------------------------- #
#                            8. Convex sets                             #
# --------------------------------------------------------------------- #


class TestConvexSets:
    def test_ball(self, setting, rng):
        X = setting.X
        G = gram(X)
        center = X.random(rng=rng)
        ball = Ball(X, radius=0.7, center=center)
        x = X.scale(3.0, X.random(rng=rng))
        d = arr(x) - arr(center)
        distance = np.sqrt(d @ G @ d)
        assert not ball.contains(x)
        assert_close(X, ball.project(x), arr(center) + 0.7 * d / distance)
        inside = X.axpy(0.1, X.random(rng=rng), X.copy(center))
        assert ball.contains(inside)
        assert_close(X, ball.project(inside), arr(inside))
        q = X.random(rng=rng)
        expected = arr(q) @ G @ arr(center) + 0.7 * np.sqrt(arr(q) @ G @ arr(q))
        assert ball.support_function()(q) == pytest.approx(expected, rel=1e-10)

    def test_ellipsoid(self, setting, rng):
        X = setting.X
        G = gram(X)
        S = spd_matrix(np.random.default_rng(17), NX)
        Pc = np.linalg.solve(G, S)
        Cc = np.linalg.inv(Pc)
        center = X.random(rng=rng)
        ellipsoid = Ellipsoid(
            X,
            matrix_operator(X, X, Pc, traits=SPD),
            center=center,
            covariance=matrix_operator(X, X, Cc, traits=SPD),
        )
        x = X.scale(3.0, X.random(rng=rng))
        d = arr(x) - arr(center)
        assert ellipsoid.contains(x) == (d @ G @ Pc @ d <= 1.0)
        q = X.random(rng=rng)
        expected = arr(q) @ G @ arr(center) + np.sqrt(arr(q) @ G @ Cc @ arr(q))
        assert ellipsoid.support_function()(q) == pytest.approx(expected, rel=1e-8)

        projection = ellipsoid.project(x)
        result = minimize(
            lambda m: (m - arr(x)) @ G @ (m - arr(x)),
            arr(center),
            jac=lambda m: 2.0 * G @ (m - arr(x)),
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda m: 1.0
                    - (m - arr(center)) @ G @ Pc @ (m - arr(center)),
                }
            ],
            method="SLSQP",
            options={"ftol": 1e-15, "maxiter": 1000},
        )
        assert_close(X, projection, result.x, rel=1e-5)

    def test_affine_subspace_from_a_linear_equation(self, setting, rng):
        X, Y, A = setting.X, setting.Y, setting.A
        value = Y.random(rng=rng)
        subspace = AffineSubspace.from_linear_equation(A, value)
        AAs = setting.M @ setting.A_star
        x = X.random(rng=rng)
        expected = arr(x) - setting.A_star @ np.linalg.solve(
            AAs, setting.M @ arr(x) - arr(value)
        )
        assert_close(X, subspace.project(x), expected, rel=1e-8)
        assert_close(
            X,
            subspace.translation,
            setting.A_star @ np.linalg.solve(AAs, arr(value)),
            rel=1e-8,
        )
        assert subspace.contains(subspace.project(x))

    def test_documented_coordinate_needs_are_refused_by_name(self, setting, rng):
        """What says it needs coordinates refuses with a TypeError that says so."""
        A = setting.A
        subspace = AffineSubspace.from_linear_equation(A, setting.Y.random(rng=rng))
        with pytest.raises(TypeError, match="coordinate"):
            subspace.dimension()
        with pytest.raises(TypeError, match="coordinate"):
            subspace.projector.basis()
        with pytest.raises(TypeError, match="coordinate"):
            A.matrix()
        with pytest.raises(TypeError, match="coordinate"):
            setting.prior.as_multivariate_normal()

    def test_projector_onto_the_kernel(self, setting, rng):
        X, A = setting.X, setting.A
        projector = OrthogonalProjector.onto_kernel(A)
        AAs = setting.M @ setting.A_star
        x = X.random(rng=rng)
        expected = arr(x) - setting.A_star @ np.linalg.solve(AAs, setting.M @ arr(x))
        assert_close(X, projector(x), expected, rel=1e-8)
        assert setting.Y.norm(A(projector(x))) <= 1e-8 * setting.X.norm(x)
        check_operator(projector, rng=rng)
        check_traits(projector, rng=rng)
        assert gi.Traits.IDEMPOTENT & projector.traits
