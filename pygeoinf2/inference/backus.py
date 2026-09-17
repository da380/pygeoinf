"""
Set-valued inference: a constraint set in, a constraint set out.

The feasible model set is ``S_M == S_M^0 ∩ A^-1(d - S_eta)`` and the answer is
its image ``T(S_M)``. Four routes compute that image, and they are the same set
— which is what makes them testable against each other. See DESIGN.md §18.3.

Here: route (a), the closed form for error-free data and a ball prior, and
route (b), the linear certificate, which is where Backus-Gilbert lives. Routes
(c) and (d) — the primal bisection and the dual with bundle methods — are
§18.3's expensive general cases.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Sequence

import numpy as np
import scipy.linalg

from ..algebra.operators import Functional, LinearOperator
from ..algebra.spaces import HilbertSpace
from ..geometry.convex import Ball, ConvexSet, Ellipsoid, HalfSpace, Polytope
from ..geometry.sets import SublevelSet
from ..geometry.subspaces import OrthogonalProjector
from ..numerics.root_find import Evaluation, monotone_root
from ..probability.base import ProbabilityMeasure
from ..numerics.convex import SupportFunction
from ..numerics.solvers import CGSolver, CholeskySolver, LinearSolver
from ..traits import Traits
from .estimators import LinearPointEstimator, SetEstimator
from .problem import LinearForwardProblem

# Relative floor below which an eigenvalue of A A* is treated as zero, so
# that a null direction contributes nothing rather than dividing by noise.
_SPECTRUM_FLOOR = 1.0e-13

# Below this, a non-converged dual infimum is read as unbounded rather than as
# an answer. Support values of a bounded set are of the size of the set.
_UNBOUNDED = 1.0e6


def _self_adjoint_spectrum(operator: LinearOperator) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues, and eigenvectors orthonormal in the space's own metric.

    The *Galerkin* matrix ``G N_c`` is the symmetric one (§5.6), not the
    component matrix, so the eigenproblem is the generalised ``M v == lambda G
    v``. Its vectors satisfy ``v_j^T G v_k == delta_jk`` — orthonormal in the
    inner product of the space rather than in whichever coordinates happen to
    be in use. On a Euclidean space this is the ordinary symmetric
    eigenproblem, which is why the distinction stays invisible until it isn't.

    Used where a self-adjoint operator is *singular* and a solver cannot be:
    with the spectrum in hand, the range and the kernel can be told apart, so
    "no solution exists" becomes an answer rather than a breakdown.
    """
    space = operator.domain
    gram = space.gram_matrix()
    galerkin = operator.matrix(form="galerkin")
    values, vectors = scipy.linalg.eigh(0.5 * (galerkin + galerkin.T), gram)
    return np.clip(values, 0.0, None), vectors


def _spectral_components(
    space: HilbertSpace, vectors: np.ndarray, vector: Any
) -> np.ndarray:
    """The coefficients of *vector* in a metric-orthonormal eigenbasis.

    ``beta_k == (v_k, x)`` in the space's inner product, which in components is
    ``v_k^T G x_c``.
    """
    return vectors.T @ space.apply_gram(space.to_components(vector))


__all__ = [
    "BackusGilbert",
    "BackusGilbertParker",
    "FeasiblePropertySet",
    "harden_error",
]


def _ball_radius(candidate: Any, name: str) -> float:
    """The radius of a ball, or a complaint naming what was wanted."""
    if not isinstance(candidate, Ball):
        raise TypeError(
            f"{name} must be a Ball for this route; got a "
            f"{type(candidate).__name__}. A general convex set needs the dual route."
        )
    return float(candidate.radius)


def harden_error(problem: LinearForwardProblem, /, *, level: float) -> Ball:
    """The problem's error as a ball, hardening a measure if need be.

    The bridge from a probabilistic error to the set-theoretic one the Backus
    routes need: the smallest ball about the mean carrying probability *level*,
    which is :meth:`~pygeoinf2.probability.GaussianMeasure.ambient_ball`.

    The radius is a quantile of ``sum_i lambda_i Z_i^2`` in the *space's* norm.
    The rule this replaces was ``sqrt(chi2_crit * mean diagonal of the
    component matrix)``, which is not the radius of any credible ball and is
    not even a variance on a weighted data space — the covariance of the
    components is ``C_c G^-1``, so the component matrix's diagonal is the
    variance only when the basis is orthonormal. It also formed a dense matrix
    to read one scalar off it.

    Args:
        problem: the forward problem, whose error may be a measure or a set.
        level: the probability the ball is to carry.

    Returns:
        A ball in the data space. Radius zero when the problem has no error,
        which is the error-free case saying "exactly this".

    Raises:
        TypeError: if the error is a convex set that is not a ball.
    """
    if not problem.has_error:
        return Ball(problem.data_space, radius=0.0)
    if isinstance(problem.error, Ball):
        return problem.error
    if isinstance(problem.error, ConvexSet):
        raise TypeError(
            "This route needs the noise set to be a ball; a general convex "
            "one needs route (d). Pass noise= explicitly to bound it."
        )
    return problem.error_measure.ambient_ball(level=level)


class BackusGilbert(LinearPointEstimator):
    """The optimally-averaged estimate of a property, with its error set.

    A *linear certificate*, in the sense of §18.3(b): the estimator is a fixed
    operator ``X`` applied to the data, chosen by minimising a quadratic
    surrogate for the width of the resulting bound,

    .. code-block:: text

        M^2 ||T - X A||_HS^2  +  D^2 ||X||_HS^2      ->      X = T A* (A A* + alpha)^-1

    with ``alpha == D^2 / M^2``. **Any** ``X`` gives a *valid* bound — that is
    weak duality, and it is why this route can never be wrong, only loose. This
    particular one is the least loose in an average sense.

    What comes out is not a number but a set. The estimate is ``X d``; the
    uncertainty is a Minkowski sum of two ellipsoids, one for what the data
    cannot resolve and one for the noise, and the two are reported separately
    because they respond to different remedies.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        target: LinearOperator,
        prior: Ball,
        /,
        *,
        noise: Ball | None = None,
        level: float = 0.95,
        solver: LinearSolver | None = None,
    ) -> None:
        """
        Args:
            problem: the forward problem.
            target: the property operator ``T``.
            prior: a norm ball on the model space.
            noise: a norm ball on the data space. Taken from the problem's
                error if omitted, hardening a Gaussian one at ``level``.
            level: the confidence level used if a Gaussian error is hardened.
            solver: how to invert the data-space normal operator.
        """
        if target.domain != problem.model_space:
            raise ValueError("The property operator must act on the model space.")
        forward = problem.forward_operator
        model_radius = _ball_radius(prior, "The prior")

        if noise is None:
            noise = harden_error(problem, level=level)
        noise_radius = _ball_radius(noise, "The noise")

        if model_radius <= 0.0:
            raise ValueError("The prior ball must have a positive radius.")
        alpha = (noise_radius / model_radius) ** 2
        normal = forward @ forward.adjoint
        if alpha > 0.0:
            normal = normal + alpha * LinearOperator.identity(problem.data_space)
        inverse = (solver or CGSolver(rtol=1e-12))(
            normal.with_traits(Traits.POSITIVE_DEFINITE)
        )
        operator = target @ forward.adjoint @ inverse

        super().__init__(
            operator,
            forward_operator=forward,
            error=(
                problem.error
                if problem.has_error and not isinstance(problem.error, ConvexSet)
                else None
            ),
        )
        self._problem = problem
        self._target = target
        self._prior_radius = model_radius
        self._noise_radius = noise_radius

    @property
    def unresolved(self) -> LinearOperator:
        """``T - X A``: what the estimate cannot see.

        The complement of the resolution, and the operator whose norm sets the
        first half of the error bound. An estimate is only as good as this is
        small.
        """
        return self._target - self.resolution

    def uncertainty(self, data: Any, /) -> ConvexSet:
        """The set of property values consistent with the data.

        The Minkowski sum ``X d + (T - X A) S_M + (-X) S_eta``, whose support
        function is
        ``(q, X d) + M ||(T - X A)* q|| + D ||X* q||`` — a resolution term and
        a noise term, added. Both are needed: shrinking one at the expense of
        the other is exactly what the choice of ``X`` trades.
        """
        centre = self(data)
        unresolved = self.unresolved
        prior_radius, noise_radius = self._prior_radius, self._noise_radius
        space = self.target_space
        operator = self.operator

        def support(direction: Any) -> float:
            resolution_term = prior_radius * self._problem.model_space.norm(
                unresolved.adjoint(direction)
            )
            noise_term = noise_radius * self._problem.data_space.norm(
                operator.adjoint(direction)
            )
            return space.inner_product(centre, direction) + resolution_term + noise_term

        return ConvexSet.from_support_function(space, support)

    def error_bars(self, data: Any, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Componentwise bounds, and the two contributions separately.

        Returns ``(estimate, resolution_half_width, noise_half_width)``. The
        split is the useful part: more data narrows the second and better
        coverage narrows the first, and a single number cannot say which is
        needed.
        """
        space = self.target_space
        estimate = self(data)
        unresolved, operator = self.unresolved, self.operator
        resolution, noise = [], []
        for index in range(space.dim):
            direction = space.basis_vector(index)
            resolution.append(
                self._prior_radius
                * self._problem.model_space.norm(unresolved.adjoint(direction))
            )
            noise.append(
                self._noise_radius
                * self._problem.data_space.norm(operator.adjoint(direction))
            )
        return estimate, np.array(resolution), np.array(noise)


class _ClosedFormRoute(SetEstimator):
    """The exact feasible property set, where a closed form exists.

    Route (a) of §18.3: error-free data and a ball prior. Al-Attar (2021)
    eq. (2.84) gives the answer as an *ellipsoid*,

    .. code-block:: text

        { p : ((T P T*)^-1 (p - p~), p - p~) <= r^2 - ||m~||^2 }

    with ``m~`` the minimum-norm model fitting the data, ``p~ == T m~``, and
    ``P`` the projection onto the kernel of the forward operator. Every piece
    of that is already here: ``A A*`` is positive semidefinite by the
    palindrome rule, ``onto_kernel`` is example 18's subject, and ``Ellipsoid``
    carries the support function the bound's directional form needs.

    Costs ``dim(P) + 1`` minimum-norm solves, and nothing else.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        target: LinearOperator,
        prior: Ball,
        /,
        *,
        solver: LinearSolver | None = None,
    ) -> None:
        """
        Args:
            problem: the forward problem. Its data are treated as exact.
            target: the property operator ``T``.
            prior: a norm ball on the model space, centred at the origin.
            solver: how to invert ``A A*`` and the property Gram.
        """
        if target.domain != problem.model_space:
            raise ValueError("The property operator must act on the model space.")
        self._problem = problem
        self._target = target
        self._radius = _ball_radius(prior, "The prior")
        self._solver = solver or CGSolver(rtol=1e-12)

        forward = problem.forward_operator
        normal = (forward @ forward.adjoint).with_traits(Traits.POSITIVE_DEFINITE)
        self._normal_inverse = self._solver(normal)
        self._kernel = OrthogonalProjector.onto_kernel(forward, solver=self._solver)
        # T P T*, the shape of the answer. Positive definite whenever the
        # property is not determined by the data alone.
        self._shape = (target @ self._kernel @ target.adjoint).with_traits(
            Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )

    @property
    def data_space(self) -> HilbertSpace:
        """The problem's data space."""
        return self._problem.data_space

    @property
    def target_space(self) -> HilbertSpace:
        """The property space."""
        return self._target.codomain

    @property
    def shape(self) -> LinearOperator:
        """``T P T*``: the shape of the answer, independent of the data.

        Only the centre and the size depend on the data. That is the same
        structure as a Gaussian estimator's data-independent covariance, and it
        arrives for the same reason.
        """
        return self._shape

    def minimum_norm_model(self, data: Any, /) -> Any:
        """``A* (A A*)^-1 d``, the smallest model fitting the data exactly."""
        return self._problem.forward_operator.adjoint(self._normal_inverse(data))

    def budget(self, data: Any, /) -> float:
        """``r^2 - ||m~||^2``: what the prior has left after fitting the data.

        Negative when no model within the prior ball fits the data at all,
        which is a statement about the data and the prior together and is
        reported rather than clipped.
        """
        model = self.minimum_norm_model(data)
        return self._radius**2 - self._problem.model_space.squared_norm(model)

    def is_feasible(self, data: Any, /) -> bool:
        """Whether any model within the prior bound fits these data at all.

        v1's ``test_data_compatibility``, and the question every other method
        here assumes has been answered: the smallest model reproducing the data
        exactly is ``A* (A A*)^-1 d``, and if even that lies outside the prior
        ball then no model does and the feasible set is empty. That is a
        statement about the data and the prior together -- usually that the
        prior bound is too tight, or the data too noisy to be fitted exactly.

        A predicate, so a caller can ask *before* being told by an exception.
        :meth:`__call__` raises on the same condition, which is right when a
        set was asked for and unhelpful when the question was whether one
        exists.

        Args:
            data: the observations.

        Returns:
            Whether the feasible set is non-empty.
        """
        return self.budget(data) >= 0.0

    def __call__(self, data: Any) -> ConvexSet:
        """The feasible property set, as an ellipsoid.

        Raises:
            ValueError: if no model within the prior bound fits the data. Ask
                :meth:`is_feasible` first to test that without an exception.
        """
        budget = self.budget(data)
        if budget < 0.0:
            raise ValueError(
                f"No model within the prior ball fits these data: the smallest "
                f"one has norm {np.sqrt(self._radius**2 - budget):.4g} against "
                f"a bound of {self._radius:.4g}."
            )
        centre = self._target(self.minimum_norm_model(data))
        covariance = budget * self._shape
        return Ellipsoid(
            self.target_space,
            self._solver(covariance.with_traits(Traits.POSITIVE_DEFINITE)),
            centre=centre,
            covariance=covariance,
        )

    @cached_property
    def _joint_spectrum(self) -> tuple[Any, np.ndarray, np.ndarray]:
        """The spectrum of ``C C*`` for Parker's joint map ``C == (A, T)``.

        Depends only on the problem and the target, not on the data or the
        value -- so it is formed once, as
        :attr:`_BisectionRoute._reduced` already is for the reduced problem.
        Rebuilding it per call made every sweep over property values pay for a
        full eigendecomposition of a ``dim(D) + dim(P)`` operator.
        """
        from ..algebra.direct_sum import ColumnLinearOperator

        joint = ColumnLinearOperator([self._problem.forward_operator, self._target])
        values, vectors = _self_adjoint_spectrum(joint @ joint.adjoint)
        return joint.codomain, values, vectors

    def inclusion_norm(self, value: Any, data: Any, /) -> float:
        """``min { ||m|| : A m == d, T m == p }``, the cost of a property value.

        Parker's joint data-property map ``C == (A, T)``: the smallest model
        reproducing *both* the data and the proposed property. A value is
        admissible exactly when this is within the prior bound, which is
        Al-Attar (2021) eq. (2.46).

        ``C C*`` acts on ``D (+) P``, so the solve is Parker's square system of
        size ``dim(D) + dim(P)``. It is positive *semi*definite — the palindrome
        rule gives no more than that, and once ``dim(D) + dim(P)`` exceeds
        ``dim(M)`` it is genuinely singular, because a model cannot generically
        match more numbers than it has. So this goes through the spectrum
        rather than a solver: the unreachable part of the joint target is the
        part in the kernel, and if it is non-zero the answer is ``inf``, which
        is a statement about the value and not a failure to converge.
        """
        space, values, vectors = self._joint_spectrum
        target = space.from_components(
            np.concatenate(
                [
                    self.data_space.to_components(data),
                    self.target_space.to_components(value),
                ]
            )
        )
        beta = _spectral_components(space, vectors, target)
        live = values > _SPECTRUM_FLOOR * max(values.max(initial=0.0), 1.0)
        if float(np.linalg.norm(beta[~live])) > _SPECTRUM_FLOOR * max(
            float(np.linalg.norm(beta)), 1.0
        ):
            return float("inf")
        # ||m||^2 == (C C* x, x) == (t, x) == sum beta_k^2 / lambda_k.
        return float(np.sqrt(np.sum(beta[live] ** 2 / values[live])))

    def admits(self, value: Any, data: Any, /, *, rtol: float = 1e-8) -> bool:
        """Whether a property value is consistent with the data and the prior.

        The primal membership test, run without ever forming the feasible set.
        It agrees with ``self(data).contains(value)`` — the two are different
        computations of the same statement, and the test suite checks that they
        do.

        Args:
            value: the property value to test.
            data: the observations.
            rtol: how far outside the bound still counts as admissible. A
                value exactly on the boundary is admissible in exact
                arithmetic and would fail without this.

        Returns:
            Whether the value is admissible.
        """
        return self.inclusion_norm(value, data) <= self._radius * (1.0 + rtol)

    def push_forward(self, operator: LinearOperator, /) -> "_ClosedFormRoute":
        """The same inference about a further property of the model."""
        return _ClosedFormRoute(
            self._problem,
            operator @ self._target,
            Ball(self._problem.model_space, radius=self._radius),
            solver=self._solver,
        )

    def prior_only(self) -> ConvexSet:
        """What the prior alone says, before any data.

        ``((T T*)^-1 p, p) <= r^2``, Al-Attar (2021) eq. (2.85). The bracket
        the data are supposed to improve on, and worth reporting beside the
        answer for exactly that reason.
        """
        covariance = self._radius**2 * (
            self._target @ self._target.adjoint
        ).with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE)
        return Ellipsoid(
            self.target_space,
            self._solver(covariance),
            covariance=covariance,
        )


def _minimum_norm_fits(
    problem: LinearForwardProblem,
    data: Any,
    /,
    *,
    noise_radius: float,
    prior_radius: float,
    solver: LinearSolver | None = None,
    iterations: int = 60,
    rtol: float = 1e-6,
) -> bool:
    """Whether a model within the prior ball fits the data within the noise ball.

    v1's ``test_data_compatibility``, matrix-free, and the same search as
    the discrepancy principle's: the smallest model whose misfit
    ``||d - A m||`` is within the noise radius is the damped minimum-norm
    model at the damping where the misfit reaches it, because the misfit
    rises with the damping while the model's norm falls.
    :func:`~pygeoinf2.inference.point.misfit_search` finds that damping,
    one warm-started Krylov solve in the data space per probe, each an
    application of ``A`` and ``A*``, and never forms ``A A*`` -- which at a
    datum per column is the one thing a data space of any size cannot
    afford (DESIGN §50). Only the misfit differs from the discrepancy
    principle's: the plain data norm here, the chi-squared there, which is
    the difference between a noise ball and a credible ellipsoid.

    The misfit is measured on the model the solve gives, so it carries the
    solve's residual: a noise radius below the solver's tolerance is one no
    fit can be certified to, and the answer is then that nothing fits.

    Args:
        problem: the forward problem; its error measure is not used, the
            noise being given as a radius in the data space's own norm.
        data: the observations.
        noise_radius: how far a fit may miss the data. Must exceed the
            solver's residual floor to be answerable.
        prior_radius: how large a model may be.
        solver: for the data-space solves. Conjugate gradients by default.
        iterations: the root search's budget.
        rtol: the root search's bracket tolerance.

    Returns:
        Whether such a model exists.
    """
    from .point import misfit_search
    from .tikhonov import TikhonovFamily

    forward = problem.forward_operator
    data_space, model_space = forward.codomain, forward.domain
    if data_space.norm(data) <= noise_radius:
        return True  # the zero model already fits
    family = TikhonovFamily(forward, solver=solver, formalism="data_space")
    found = misfit_search(
        family,
        family.right_hand_side(data),
        lambda model: data_space.norm(data_space.subtract(data, forward(model))),
        noise_radius,
        iterations=iterations,
        rtol=rtol,
    )
    if found.value > noise_radius * (1.0 + rtol) and found.exhausted is not None:
        # The misfit stayed above the radius at the smallest damping tried:
        # the data are not fitted to the noise by any model at all.
        return False
    return model_space.norm(family.model_from(found.solution)) <= prior_radius


class _Quadratic:
    """A ball or an ellipsoid read as ``{ x : (Q (x - a), x - a) <= 1 }``.

    The primal route works in the inner product a set defines, and a
    quadratic level function defines one: ``Q`` its precision, ``C == Q^-1``
    its covariance, ``a`` its centre. A ball of radius ``r`` has ``Q == I /
    r^2``; its distances are reported in the space's own norm, so
    :attr:`scale` is ``r`` there and one for an ellipsoid, whose level
    function is already normalised. An ellipsoid built without its
    covariance gets one through the solver, an inverse applied by conjugate
    gradients wherever the route needs ``C``.
    """

    def __init__(self, subset: Any, solver: LinearSolver, name: str) -> None:
        space = subset.domain
        self.subset = subset
        if isinstance(subset, Ball):
            radius = float(subset.radius)
            if radius <= 0.0:
                raise ValueError(f"{name} needs a positive radius for this route.")
            self.is_ball = True
            self.scale = radius
            self.centre = subset.centre
            self.precision = (
                LinearOperator.identity(space) * (1.0 / radius**2)
            ).with_traits(Traits.POSITIVE_DEFINITE)
            self.covariance = (LinearOperator.identity(space) * radius**2).with_traits(
                Traits.POSITIVE_DEFINITE
            )
        elif isinstance(subset, Ellipsoid):
            self.is_ball = False
            self.scale = 1.0
            self.centre = subset.centre
            self.precision = subset.precision.with_traits(Traits.POSITIVE_DEFINITE)
            covariance = subset.covariance
            if covariance is None:
                covariance = solver(self.precision)
            self.covariance = covariance
        else:
            raise TypeError(
                f"{name} must be a Ball or an Ellipsoid for this route; got a "
                f"{type(subset).__name__}. A general convex set needs the dual route."
            )


class _BisectionRoute(SetEstimator):
    """The exact feasible property set for noisy data, by the primal route.

    Route (c) of §18.3, BGP's primal route, for a prior and a confidence set
    that are each a ball or an ellipsoid -- sets with a quadratic level
    function. The support value in a direction is a concave maximisation
    over the intersection of the two, and attaching multipliers to the two
    constraints turns its stationarity condition into

    .. code-block:: text

        (s Q + t A* P A) u == T* q + t A* P d'

    with ``Q`` the prior's precision, ``P`` the confidence set's, ``u`` the
    model relative to the prior's centre and ``d'`` the data relative to
    both centres: a damped least-squares solve in the two sets' own inner
    products, the same primitive as §18.6. The multipliers are fixed by the
    two level functions reaching their levels, both monotone in their own
    multiplier, so nested bisection converges. A ball is the case ``Q == I
    / M^2``; nothing here needs it to be.

    The route is written in those inner products rather than by whitening
    the problem: the reduction to the data space diagonalises ``A C A*`` in
    the inner product ``P`` defines, through the Cholesky factor of ``P``
    on the data components, and the model side carries ``C == Q^-1`` as an
    operator. For balls both are scalings and the arithmetic is v1's.

    Two things this has that the dual route does not: it reuses solvers that
    already exist, and it produces the **extremal model** attaining each
    bound. What it lacks is generality: a set without a quadratic level
    function is the dual's.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        target: LinearOperator,
        prior: ConvexSet,
        /,
        *,
        noise: ConvexSet | None = None,
        level: float = 0.95,
        solver: LinearSolver | None = None,
        iterations: int = 60,
    ) -> None:
        """
        Args:
            problem: the forward problem.
            target: the property operator ``T``.
            prior: a ball or an ellipsoid on the model space.
            noise: a ball or an ellipsoid on the data space; taken from the
                problem if omitted.
            level: the level at which a Gaussian error is hardened.
            solver: how to invert the damped normal operator, and the prior's
                precision when its covariance is not given.
            iterations: bisection steps, on each of the two multipliers.
        """
        if target.domain != problem.model_space:
            raise ValueError("The property operator must act on the model space.")
        self._problem = problem
        self._target = target
        self._solver = solver or CGSolver(rtol=1e-12)
        self._prior = _Quadratic(prior, self._solver, "The prior")
        self._noise = _Quadratic(
            harden_error(problem, level=level) if noise is None else noise,
            self._solver,
            "The noise",
        )
        self._iterations = iterations
        # Reported in the prior's units: a ball's radius, an ellipsoid's one.
        self._radius = self._prior.scale
        self._noise_radius = self._noise.scale

    @property
    def data_space(self) -> HilbertSpace:
        """The problem's data space."""
        return self._problem.data_space

    @property
    def target_space(self) -> HilbertSpace:
        """The property space."""
        return self._target.codomain

    # ----------------------------------------------------------------- #
    #                      The two inner products                       #
    # ----------------------------------------------------------------- #

    @cached_property
    def _whitening(self) -> np.ndarray | None:
        """``W`` with ``W^T W == P_c`` on the data components, or None for a ball.

        The confidence set's inner product on components, as a triangular
        factor: ``(P r, r) == ||W r_c||^2``. A ball's ``P`` is a scaling and
        needs no factor.
        """
        if self._noise.is_ball:
            return None
        if not self.data_space.is_orthonormal:
            raise NotImplementedError(
                "The primal route reduces to the data space, which needs that "
                "space to be orthonormal. Every forward problem here has a "
                "Euclidean data space; if yours does not, use the dual route."
            )
        matrix = self._noise.precision.matrix(form="components")
        return np.linalg.cholesky(0.5 * (matrix + matrix.T)).T

    def _whiten(self, vector: Any) -> np.ndarray:
        """A data vector's components in the confidence set's inner product."""
        components = self.data_space.to_components(vector)
        if self._whitening is None:
            return components / self._noise.scale
        return self._whitening @ components

    def _unwhiten(self, components: np.ndarray) -> Any:
        """The adjoint of :meth:`_whiten`: whitened components back to a data vector."""
        if self._whitening is None:
            return self.data_space.from_components(components / self._noise.scale)
        return self.data_space.from_components(self._whitening.T @ components)

    def _shifted(self, data: Any) -> Any:
        """``d - A a - b``: the data relative to both centres."""
        space = self.data_space
        shifted = space.subtract(
            data, self._problem.forward_operator(self._prior.centre)
        )
        return space.subtract(shifted, self._noise.centre)

    def _misfit(self, residual: Any) -> float:
        """``sqrt((P r, r))``, the confidence set's level function's root."""
        return float(np.linalg.norm(self._whiten(residual)))

    # ----------------------------------------------------------------- #
    #                            The reduction                           #
    # ----------------------------------------------------------------- #

    @cached_property
    def _data_gram(self) -> tuple[np.ndarray, np.ndarray]:
        """The spectrum of ``A C A*`` in the confidence set's inner product.

        BGP §2.6's reduction, and the thing that makes the bisection
        affordable. Woodbury turns the model-space solve

        .. code-block:: text

            (s Q + t A* P A)^-1

        into one on the data space, so every quantity the bisection tests --
        the model's level function and the misfit -- becomes an ``O(dim(D))``
        expression once ``W (A C A*)_c W^T`` is diagonalised, ``W`` the
        factor of :attr:`_whitening`. Without it each of the four thousand
        bisection steps per direction would be a fresh Krylov solve.

        Costs ``dim(D)`` applications of the forward operator, its adjoint
        and the prior's covariance, once per estimator.
        """
        if not self.data_space.is_orthonormal:
            raise NotImplementedError(
                "The primal route reduces to the data space, which needs that "
                "space to be orthonormal. Every forward problem here has a "
                "Euclidean data space; if yours does not, use the dual route."
            )
        forward = self._problem.forward_operator
        gram = (forward @ self._prior.covariance @ forward.adjoint).matrix(
            form="components"
        )
        return self._reduce(gram)

    def _reduce(self, gram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Diagonalise a data-space Gram in the confidence set's inner product."""
        gram = 0.5 * (gram + gram.T)
        if self._whitening is None:
            gram = gram / self._noise.scale**2
        else:
            gram = self._whitening @ gram @ self._whitening.T
        values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
        return np.clip(values, 0.0, None), vectors

    @staticmethod
    def _live(values: np.ndarray) -> np.ndarray:
        return values > _SPECTRUM_FLOOR * max(values.max(initial=0.0), 1.0)

    def _kernel_part(self, dual: Any, image: np.ndarray) -> float:
        """``(C v, v) - sum image^2 / lambda``: what of a dual vector lies off the range.

        Computed once per direction and at the natural scale, because the
        alternative -- subtracting the range part from the whole at each
        bisection step -- cancels: at a data weight of ``1e8`` the kernel
        term is ``1e-16`` of two quantities of order one, and comes out as
        noise.
        """
        values, _ = self._data_gram
        live = self._live(values)
        whole = self._problem.model_space.inner_product(
            self._prior.covariance(dual), dual
        )
        return max(whole - float(np.sum(image[live] ** 2 / values[live])), 0.0)

    def _prepare(self, direction: Any, data: Any) -> dict:
        forward = self._problem.forward_operator
        pulled = self._target.adjoint(direction)
        shifted = self._shifted(data)
        adjoint_data = forward.adjoint(self._noise.precision(shifted))
        whitened = self._whiten(shifted)
        values, vectors = self._data_gram
        forward_pulled = self._whiten(forward(self._prior.covariance(pulled)))
        return {
            "pulled": pulled,
            "adjoint_data": adjoint_data,
            "data": whitened,
            "forward_pulled": forward_pulled,
            "gram_data": vectors @ (values * (vectors.T @ whitened)),
            "pulled_kernel_squared": self._kernel_part(
                pulled, vectors.T @ forward_pulled
            ),
        }

    def _state(self, prepared: dict, damping: float, weight: float) -> tuple:
        """The model's level function's root and the misfit at ``(gamma, t)``.

        Parameterised by ``gamma == s / t`` rather than by ``s``, which is
        what BGP §2.6 calls it and which is the only stable choice: with
        ``s``, the norm is a ratio of two large numbers as ``t`` grows and
        the whole expression cancels. Here

        .. code-block:: text

            u* == C (gamma I + A* P A C)^-1 ( (1/t) T* q + A* P d' )

        stays bounded as ``t -> infinity``, tending to the damped
        least-squares solution. Both quantities expand into sums over the
        reduced spectrum.
        """
        values, vectors = self._data_gram
        inverse_weight = 1.0 / weight
        forward_w = inverse_weight * prepared["forward_pulled"] + prepared["gram_data"]
        projected = vectors.T @ forward_w
        image = vectors @ (projected / (damping + values))
        kernel_squared = inverse_weight**2 * prepared["pulled_kernel_squared"]
        live = self._live(values)
        model_squared = kernel_squared / damping**2 + float(
            np.sum(
                projected[live] ** 2 / (values[live] * (damping + values[live]) ** 2)
            )
        )
        residual = prepared["data"] - image
        return np.sqrt(model_squared), float(np.linalg.norm(residual))

    def _model(self, prepared: dict, damping: float, weight: float) -> Any:
        """The model at ``(gamma, t)``, relative to the prior's centre."""
        space = self._problem.model_space
        values, vectors = self._data_gram
        inverse_weight = 1.0 / weight
        forward_w = inverse_weight * prepared["forward_pulled"] + prepared["gram_data"]
        projected = vectors.T @ forward_w
        adjoint = self._problem.forward_operator.adjoint
        live = self._live(values)
        w = space.add(
            space.scale(inverse_weight, prepared["pulled"]), prepared["adjoint_data"]
        )
        pseudo = np.zeros_like(projected)
        pseudo[live] = projected[live] / values[live]
        kernel = space.subtract(w, adjoint(self._unwhiten(vectors @ pseudo)))
        weighted = np.zeros_like(projected)
        weighted[live] = projected[live] / (values[live] * (damping + values[live]))
        dual = space.add(
            space.scale(1.0 / damping, kernel),
            adjoint(self._unwhiten(vectors @ weighted)),
        )
        return self._prior.covariance(dual)

    def _bisect(
        self, quantity: Any, target: float, /, *, decreasing: bool = True
    ) -> float:
        """The positive multiplier at which a monotone quantity hits a target.

        Delegates to :func:`~pygeoinf2.numerics.root_find.monotone_root`,
        DESIGN §18.6's one kernel. The probes here are closed form -- the
        spectral reduction has already turned each into an ``O(dim(D))``
        expression -- so there is no solve to warm start. The tolerance is
        zero so that the full iteration count is always taken: these
        searches are nested, and an inner search that stopped early would
        put a step in the outer one's function.
        """
        result = monotone_root(
            lambda multiplier, _: Evaluation(quantity(multiplier)),
            target,
            decreasing=decreasing,
            iterations=self._iterations,
            rtol=0.0,
            atol=0.0,
            warm_start=False,
        )
        if result.exhausted is not None:
            raise ValueError("The bisection could not bracket its target.")
        return result.argument

    def _fit_norm(self, prepared: dict, weight: float) -> float:
        """The damping at which the model reaches the prior's level."""
        return self._bisect(
            lambda damping: self._state(prepared, damping, weight)[0], 1.0
        )

    # ----------------------------------------------------------------- #
    #                          The support side                         #
    # ----------------------------------------------------------------- #

    def extremal_model(self, direction: Any, data: Any, /) -> Any:
        """The model of the feasible set furthest along a direction.

        What the dual route leaves implicit. Not generally unique -- when the
        prior constraint is slack the null-space components are free -- but
        the bound it attains is.
        """
        space = self._problem.model_space
        pulled = self._target.adjoint(direction)
        image = self._prior.covariance(pulled)
        length = float(np.sqrt(max(space.inner_product(image, pulled), 0.0)))
        if length == 0.0:
            return space.copy(self._prior.centre)

        # Prior-only: if the prior's own support point already fits the data,
        # the data constraint is slack and there is nothing to solve.
        flat = space.axpy(1.0 / length, image, space.copy(self._prior.centre))
        residual = self.data_space.subtract(
            self._shifted(data),
            self._problem.forward_operator(space.subtract(flat, self._prior.centre)),
        )
        if self._misfit(residual) <= 1.0:
            return flat

        prepared = self._prepare(direction, data)

        def misfit(weight: float) -> float:
            return self._state(prepared, self._fit_norm(prepared, weight), weight)[1]

        weight = self._bisect(misfit, 1.0)
        relative = self._model(prepared, self._fit_norm(prepared, weight), weight)
        return space.add(self._prior.centre, relative)

    def support(self, direction: Any, data: Any, /) -> float:
        """The support value of the feasible property set in one direction."""
        model = self.extremal_model(direction, data)
        return self._problem.model_space.inner_product(
            self._target.adjoint(direction), model
        )

    # ----------------------------------------------------------------- #
    #                            Feasibility                             #
    # ----------------------------------------------------------------- #

    def fitting_model(self, data: Any, /) -> Any | None:
        """The model nearest the prior's centre that fits the data, or None.

        In the prior's own inner product. For two balls this is v1's
        ``test_data_compatibility`` computation, matrix-free: the damped
        minimum-norm search of :func:`_minimum_norm_fits`, a few
        warm-started Krylov solves in the data space (DESIGN §50). For an
        ellipsoid on either side it comes from the reduction, whose cost
        the route pays anyway for its support values: with ``xi`` the
        shifted data in the reduced spectrum, the damped fit has misfit
        ``gamma ||xi / (gamma + lambda)||`` and level function ``sum lambda
        xi^2 / (gamma + lambda)^2``, monotone in ``gamma``.
        """
        space = self._problem.model_space
        shifted = self._shifted(data)
        if self._prior.is_ball and self._noise.is_ball:
            from .point import misfit_search
            from .tikhonov import TikhonovFamily

            forward = self._problem.forward_operator
            if self.data_space.norm(shifted) <= self._noise.scale:
                return space.copy(self._prior.centre)
            family = TikhonovFamily(
                forward, solver=self._solver, formalism="data_space"
            )
            found = misfit_search(
                family,
                family.right_hand_side(shifted),
                lambda model: self.data_space.norm(
                    self.data_space.subtract(shifted, forward(model))
                ),
                self._noise.scale,
                iterations=self._iterations,
            )
            if (
                found.value > self._noise.scale * (1.0 + 1e-6)
                and found.exhausted is not None
            ):
                return None
            model = family.model_from(found.solution)
            if space.norm(model) > self._prior.scale:
                return None
            return space.add(self._prior.centre, model)

        values, vectors = self._data_gram
        whitened = self._whiten(shifted)
        if float(np.linalg.norm(whitened)) <= 1.0:
            return space.copy(self._prior.centre)
        projected = vectors.T @ whitened
        live = self._live(values)
        unreachable = float(np.linalg.norm(projected[~live]))
        if unreachable > 1.0:
            return None

        def misfit(damping: float) -> float:
            return float(damping * np.linalg.norm(projected / (damping + values)))

        damping = self._bisect(misfit, 1.0, decreasing=False)
        level = float(
            np.sum(values[live] * projected[live] ** 2 / (damping + values[live]) ** 2)
        )
        if level > 1.0:
            return None
        weighted = np.zeros_like(projected)
        weighted[live] = projected[live] / (damping + values[live])
        dual = self._problem.forward_operator.adjoint(
            self._unwhiten(vectors @ weighted)
        )
        return space.add(self._prior.centre, self._prior.covariance(dual))

    def is_feasible(self, data: Any, /) -> bool:
        """Whether any model lies in both the prior set and the noise set."""
        return self.fitting_model(data) is not None

    def __call__(self, data: Any) -> ConvexSet:
        """The feasible property set, as a support-function oracle."""
        return ConvexSet.from_support_function(
            self.target_space,
            lambda direction: self.support(direction, data),
            maximiser=lambda direction: self._target(
                self.extremal_model(direction, data)
            ),
        )

    # ----------------------------------------------------------------- #
    #                          Set inclusion                            #
    # ----------------------------------------------------------------- #

    @cached_property
    def _property_pseudo_inverse(self) -> LinearOperator:
        """``C T* (T C T*)^-1``: the model nearest the prior's centre with a given property.

        Factored rather than iterated. The property space is
        finite-dimensional and small -- that is what makes it a *property*
        space (§18.1) -- so ``T C T*`` is a handful of rows.
        """
        normal = (
            self._target @ self._prior.covariance @ self._target.adjoint
        ).with_traits(Traits.POSITIVE_DEFINITE)
        return self._prior.covariance @ self._target.adjoint @ CholeskySolver()(normal)

    @cached_property
    def _reduced(self) -> tuple[np.ndarray, np.ndarray]:
        """The spectrum of ``A P_Q C A*`` in the confidence set's inner product.

        Al-Attar (2021) §3.3: fixing a property value confines the model to
        ``m~ + ker T``, and asking whether any such model fits the data is
        *the same problem again* in that subspace, with the projector onto
        the kernel taken orthogonally in the prior's inner product, ``P_Q ==
        I - C T* (T C T*)^-1 T``, which is eq. (3.28) in that metric.
        """
        space = self._problem.model_space
        projector = (
            LinearOperator.identity(space)
            - self._property_pseudo_inverse @ self._target
        )
        forward = self._problem.forward_operator
        reduced = (
            forward @ projector @ self._prior.covariance @ forward.adjoint
        ).matrix(form="components")
        return self._reduce(reduced)

    def inclusion_norm(self, value: Any, data: Any, /) -> float:
        """The prior's level function's root of the smallest model with this value fitting the data.

        Reported in the prior's units: for a ball prior the model's norm,
        compared with the radius; for an ellipsoid the Mahalanobis form's
        root, compared with one. Infinite when no model at all can fit the
        data with this property -- a *proof* that the value is inadmissible,
        not a failure to find one.

        The reduction is Al-Attar (2021) §3.3. Writing ``m == m~ + u`` with
        ``m~`` the model nearest the prior's centre having the property and
        ``u`` in the kernel of ``T``, the level function separates and what
        is left is a discrepancy problem in the subspace, closed form in the
        reduced spectrum: the misfit is ``gamma ||z||`` and the model's level
        function ``(A P C A* z, z)``, both monotone in ``gamma`` and neither
        involving a cancellation.
        """
        space = self._problem.model_space
        forward = self._problem.forward_operator
        values, vectors = self._reduced
        offset = self.target_space.subtract(value, self._target(self._prior.centre))
        relative = self._property_pseudo_inverse(offset)
        anchor_squared = space.inner_product(self._prior.precision(relative), relative)
        residual = self.data_space.subtract(self._shifted(data), forward(relative))
        whitened = self._whiten(residual)
        if float(np.linalg.norm(whitened)) <= 1.0:
            return self._prior.scale * float(np.sqrt(anchor_squared))
        projected = vectors.T @ whitened
        live = self._live(values)
        unreachable = float(np.linalg.norm(projected[~live]))
        if unreachable > 1.0:
            return float("inf")

        def misfit(damping: float) -> float:
            return float(damping * np.linalg.norm(projected / (damping + values)))

        damping = self._bisect(misfit, 1.0, decreasing=False)
        correction = float(np.sum(values * projected**2 / (damping + values) ** 2))
        return self._prior.scale * float(np.sqrt(anchor_squared + correction))

    def admits(self, value: Any, data: Any, /, *, rtol: float = 1e-8) -> bool:
        """Whether a property value is consistent with the data and the prior.

        Args:
            value: the property value to test.
            data: the observations.
            rtol: tolerance on the bound, as for :meth:`_ClosedFormRoute.admits`.

        Returns:
            Whether the value is admissible.
        """
        return self.inclusion_norm(value, data) <= self._prior.scale * (1.0 + rtol)

    def push_forward(self, operator: LinearOperator, /) -> "_BisectionRoute":
        """The same inference about a further property."""
        return _BisectionRoute(
            self._problem,
            operator @ self._target,
            self._prior.subset,
            noise=self._noise.subset,
            solver=self._solver,
            iterations=self._iterations,
        )


class _DualRoute(SetEstimator):
    """The feasible property set for *any* convex prior and noise sets.

    Route (d) of §18.3, and the general one. Duality turns the supremum over an
    infinite-dimensional model set into an infimum over the finite-dimensional
    data space,

    .. code-block:: text

        h(q) == inf over lambda of
                (lambda, d) + h_prior(T* q - A* lambda) + h_noise(-lambda)

    which is BGP eq. (28), and which is exactly what v1's
    ``DualMasterCostFunction`` docstring writes down without naming it as a
    support function of an image.

    It uses only the two sets' *support functions*, so it accepts anything
    convex — an ellipsoid, a box, an intersection — where routes (a) and (c)
    accept norm balls. What it costs is a nonsmooth convex minimisation per
    direction, which is why the cheaper routes exist at all.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        target: LinearOperator,
        prior: ConvexSet,
        /,
        *,
        noise: ConvexSet | None = None,
        method: Any = None,
    ) -> None:
        """
        Args:
            problem: the forward problem.
            target: the property operator ``T``.
            prior: any convex set on the model space with a support function
                and a support maximiser.
            noise: likewise on the data space; taken from the problem if
                omitted.
            method: the minimiser. A proximal bundle method by default.
        """
        from ..numerics.convex import ProximalBundleMethod

        if target.domain != problem.model_space:
            raise ValueError("The property operator must act on the model space.")
        self._problem = problem
        self._target = target
        self._prior = prior
        if noise is None:
            if not problem.has_error or not isinstance(problem.error, ConvexSet):
                raise ValueError(
                    "This route needs a convex noise set; pass noise= or give "
                    "the problem a set-valued error."
                )
            noise = problem.error
        self._noise = noise
        self._method = method or ProximalBundleMethod(tolerance=1e-10, iterations=300)

    @property
    def data_space(self) -> HilbertSpace:
        """The problem's data space."""
        return self._problem.data_space

    @property
    def target_space(self) -> HilbertSpace:
        """The property space."""
        return self._target.codomain

    def dual_cost(self, direction: Any, data: Any, /) -> Any:
        """The functional whose infimum is the support value.

        A convex function of the certificate ``lambda``, built from the two
        support functions and nothing else. Its subgradient is
        ``d - A x_prior - x_noise`` with each ``x`` the point of its set
        attaining the corresponding support — so a set that can exhibit its own
        maximiser is all this route ever asks for.
        """
        from ..algebra.operators import Functional

        space = self.data_space
        model_space = self._problem.model_space
        forward = self._problem.forward_operator
        pulled = self._target.adjoint(direction)
        prior_support = self._prior.support_function()
        noise_support = self._noise.support_function()

        # A bundle method asks for the value and the subgradient at the *same*
        # point, one after the other, and both need the same two quantities:
        # the residual ``T* q - A* lambda`` and the negated certificate. v1
        # fused them into one oracle for exactly this reason; here a one-entry
        # memo does it without changing the Functional protocol, and the
        # saving is one adjoint application and one negation per oracle call.
        cache: dict[str, Any] = {}

        def prepare(certificate: Any) -> tuple[Any, Any]:
            # The key is the certificate *object*, not its ``id``: holding a
            # reference to it is what makes the identity test meaningful.
            # Keying on ``id()`` alone lets a freed array's address be reused
            # by the next one, and the memo then answers with the previous
            # certificate's residual -- a wrong subgradient, silently.
            if cache.get("certificate") is not certificate:
                cache["certificate"] = certificate
                cache["parts"] = (
                    model_space.subtract(pulled, forward.adjoint(certificate)),
                    space.scale(-1.0, certificate),
                )
            return cache["parts"]

        def value(certificate: Any) -> float:
            residual, negated = prepare(certificate)
            return (
                space.inner_product(certificate, data)
                + prior_support(residual)
                + noise_support(negated)
            )

        def gradient(certificate: Any) -> Any:
            residual, negated = prepare(certificate)
            from_prior = forward(self._prior.support_maximiser(residual))
            from_noise = self._noise.support_maximiser(negated)
            return space.subtract(space.subtract(data, from_prior), from_noise)

        return Functional.from_callables(space, value, gradient=gradient)

    @staticmethod
    def _smoothed_support(
        given: Any, epsilon: float, /
    ) -> tuple[Callable[[Any], float], Callable[[Any], Any]]:
        """A support function with its corner rounded off, and its gradient.

        Moreau-Yosida smoothing. A ball's support ``(z, c) + r ||z||`` is not
        differentiable at the origin, which is exactly where a bundle method
        spends its time; replacing the norm by ``sqrt(||z||^2 + eps^2)`` makes
        it smooth everywhere, at the cost of an ``O(eps)`` error in the value
        and a Lipschitz constant that grows like ``1 / eps``. A smaller
        epsilon is a better approximation and a harder problem, which is the
        whole trade.

        Args:
            given: a ball or an ellipsoid.
            epsilon: the smoothing.

        Returns:
            The smoothed support and its gradient.

        Raises:
            TypeError: for any other set. There is no general formula --
                smoothing a support function needs to know its shape.
        """
        from ..geometry.convex import Ball, Ellipsoid

        space = given.domain
        squared = epsilon * epsilon

        if isinstance(given, Ball):
            radius, centre = given.radius, given.centre

            def value(z: Any) -> float:
                return space.inner_product(z, centre) + radius * float(
                    np.sqrt(space.squared_norm(z) + squared)
                )

            def gradient(z: Any) -> Any:
                scale = radius / float(np.sqrt(space.squared_norm(z) + squared))
                return space.add(centre, space.scale(scale, z))

            return value, gradient

        if isinstance(given, Ellipsoid):
            if given._covariance is None:
                raise ValueError("A smoothed ellipsoid support needs the covariance.")
            covariance, centre = given._covariance, given.centre

            def value(z: Any) -> float:
                weighted = covariance(z)
                return space.inner_product(z, centre) + float(
                    np.sqrt(space.inner_product(z, weighted) + squared)
                )

            def gradient(z: Any) -> Any:
                weighted = covariance(z)
                scale = 1.0 / float(np.sqrt(space.inner_product(z, weighted) + squared))
                return space.add(centre, space.scale(scale, weighted))

            return value, gradient

        raise TypeError(
            f"Smoothing needs a ball or an ellipsoid, whose support has a "
            f"closed form to smooth; got {type(given).__name__}. Use the "
            "unsmoothed dual route, which handles any convex set."
        )

    def smoothed_dual_cost(
        self, direction: Any, data: Any, /, *, epsilon: float = 1e-3
    ) -> Any:
        """The dual cost with its corners rounded off, so it is differentiable.

        :meth:`dual_cost` is convex and non-smooth -- it is built from support
        functions, which have corners -- so it needs a subgradient method.
        Rounding the corners by Moreau-Yosida smoothing makes it a smooth
        problem, which L-BFGS solves with superlinear convergence instead.

        The approximation costs ``O(epsilon)`` in the value, and a smaller
        epsilon makes the problem stiffer -- the Lipschitz constant grows like
        ``1 / epsilon``. That is the trade this route offers, and there is no
        setting of it that is free.

        Args:
            direction: the direction to evaluate.
            data: the observations.
            epsilon: the smoothing.

        Returns:
            A differentiable ``Functional`` on the data space.

        Raises:
            ValueError: if the smoothing is not positive.
            TypeError: if either set is neither a ball nor an ellipsoid.
        """
        from ..algebra.operators import Functional

        if epsilon <= 0.0:
            raise ValueError(f"The smoothing must be positive, got {epsilon}.")

        space = self.data_space
        model_space = self._problem.model_space
        forward = self._problem.forward_operator
        pulled = self._target.adjoint(direction)
        prior_value, prior_gradient = self._smoothed_support(self._prior, epsilon)
        noise_value, noise_gradient = self._smoothed_support(self._noise, epsilon)

        def value(certificate: Any) -> float:
            residual = model_space.subtract(pulled, forward.adjoint(certificate))
            return (
                space.inner_product(certificate, data)
                + prior_value(residual)
                + noise_value(space.scale(-1.0, certificate))
            )

        def gradient(certificate: Any) -> Any:
            residual = model_space.subtract(pulled, forward.adjoint(certificate))
            negated = space.scale(-1.0, certificate)
            return space.subtract(
                space.subtract(data, forward(prior_gradient(residual))),
                noise_gradient(negated),
            )

        return Functional.from_callables(space, value, gradient=gradient)

    def primal_solver(self, data: Any, /, **kwargs: Any) -> Any:
        """The same problem set up for the *primal* route.

        :class:`~pygeoinf2.numerics.convex.ChambollePockSolver` maximises
        ``(c, m)`` over the feasible set directly, where everything else here
        minimises the dual. The two answer the same question and meet at the
        same number -- measured, they agree to 1e-10, which is strong duality
        checked rather than assumed -- but they cost differently: the dual
        route takes few expensive bundle iterations, the primal many cheap
        splitting steps, one application of ``A`` and one projection each.

        Which is better depends on the sets. A ball projects in closed form
        and the primal route is then very cheap per step; an intersection
        projects by Dykstra, iteratively, and it is not.

        Args:
            data: the observations.
            **kwargs: passed to the solver -- step sizes, tolerance, the
                iteration cap.

        Returns:
            A solver ready to take a direction's ``T* q``.
        """
        from ..numerics.convex import ChambollePockSolver

        return ChambollePockSolver(
            self._prior,
            self._noise,
            self._problem.forward_operator,
            data,
            **kwargs,
        )

    def support_values(
        self,
        directions: Sequence[Any],
        data: Any,
        /,
        *,
        route: str = "dual",
        warm_start: bool = True,
        n_jobs: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """The support values in many directions, sweeping with a warm start.

        Restores v1's ``solve_support_values``. Neighbouring directions have
        neighbouring certificates, so each minimisation started from the last
        one's answer is a correction rather than a fresh problem -- which is
        the whole reason to sweep rather than to call :meth:`support` in a
        loop, and the part :class:`ProximalBundleMethod` does not supply on
        its own.

        Warm starting is inherently *sequential*: each direction needs the one
        before it. Passing ``n_jobs`` runs the directions in parallel instead
        and gives the warm start up, which is the right trade only when the
        directions are few and each minimisation is dear.

        Args:
            directions: the directions to evaluate.
            data: the observations.
            route: which solver answers the question. All three agree, and
                differ only in cost:

                * ``"dual"`` minimises the dual cost with a bundle method.
                  Works for any convex sets, and is the most expensive.
                * ``"primal"`` maximises over the feasible set directly with
                  :class:`~pygeoinf2.numerics.convex.ChambollePockSolver`.
                  Also works for any convex sets, and is cheap per step when
                  they project cheaply. This is v1's
                  ``solve_primal_feasibility``.
                * ``"kkt"`` writes the answer down from the KKT conditions
                  with :class:`~pygeoinf2.numerics.convex.PrimalKKTSolver`.
                  Needs both sets to be balls or ellipsoids, and is by far the
                  cheapest where it applies -- tens of function evaluations
                  against hundreds of splitting steps or a bundle
                  minimisation. It also never discretises the model space.

                Measured agreement between the three: 1.7e-8 relative between
                dual and primal, and 2.7e-11 between primal and KKT.
            warm_start: carry each answer into the next. Ignored when running
                in parallel, where there is no previous answer to carry.
            n_jobs: workers. ``None`` or one keeps the sweep sequential.
            **kwargs: passed to the primal solver, and ignored on the dual
                route, which is configured through ``method=`` at construction.

        Returns:
            One support value per direction.

        Raises:
            ValueError: if the route is not one of the two, or -- on the dual
                route -- if the feasible set is empty, as :meth:`support` does.
        """
        if route not in ("dual", "primal", "kkt", "smoothed"):
            raise ValueError(
                f"The route is 'dual', 'primal', 'kkt' or 'smoothed', got "
                f"{route!r}."
            )
        directions = tuple(directions)
        if not directions:
            return np.empty(0)

        if route == "smoothed":
            from ..numerics.optimisation import LBFGS

            optimiser = kwargs.pop("optimiser", None) or LBFGS(max_iterations=500)
            values, start = [], None
            for direction in directions:
                cost = self.smoothed_dual_cost(direction, data, **kwargs)
                origin = self.data_space.zero() if start is None else start
                result = optimiser.minimise(cost, origin)
                values.append(result.value)
                if warm_start:
                    start = result.minimiser
            return np.array(values)

        if route == "kkt":
            from ..numerics.convex import PrimalKKTSolver

            solver = PrimalKKTSolver(
                self._prior,
                self._noise,
                self._problem.forward_operator,
                data,
                **kwargs,
            )
            return np.array(
                [
                    solver.solve(self._target.adjoint(direction)).value
                    for direction in directions
                ]
            )

        if route == "primal":
            solver = self.primal_solver(data, **kwargs)
            values, start = [], None
            for direction in directions:
                result = solver.solve(
                    self._target.adjoint(direction),
                    start=start if warm_start else None,
                )
                values.append(result.value)
                if warm_start:
                    start = result.model
            return np.array(values)

        from ..parallel import parallel_map, resolve_jobs

        if resolve_jobs(n_jobs) != 1:
            return np.array(
                parallel_map(
                    lambda direction: self.support(direction, data),
                    directions,
                    n_jobs=n_jobs,
                )
            )

        values, start = [], None
        for direction in directions:
            cost = self.dual_cost(direction, data)
            origin = self.data_space.zero() if start is None else start
            result = self._method.minimise(cost, origin)
            self._check_bounded(result)
            values.append(result.value)
            if warm_start:
                start = result.minimiser
        return np.array(values)

    def support(self, direction: Any, data: Any, /, *, start: Any = None) -> float:
        """The support value in one direction, by minimising the dual cost.

        **An unbounded infimum means the feasible set is empty**, not that the
        minimisation failed: with no model both inside the prior and fitting
        the data, the primal supremum is over nothing and the dual falls away
        without limit. That is reported rather than returned, because a large
        negative number is a perfectly plausible-looking support value.

        Args:
            direction: the direction to evaluate in.
            data: the observations.
            start: where to begin the minimisation. Used by
                :meth:`support_values` to carry each direction's certificate
                into the next.

        Returns:
            The support value.

        Raises:
            ValueError: if the dual is unbounded, meaning the feasible set is
                empty. :meth:`is_feasible` tests that without an exception.
        """
        cost = self.dual_cost(direction, data)
        origin = self.data_space.zero() if start is None else start
        result = self._method.minimise(cost, origin)
        self._check_bounded(result)
        return result.value

    @staticmethod
    def _check_bounded(result: Any) -> None:
        """Refuse a dual that ran away rather than converged."""
        if not result.converged and result.value < -_UNBOUNDED:
            raise ValueError(
                f"The dual fell to {result.value:.3g} without converging, "
                "which means no model lies both inside the prior set and "
                "within the noise set of the data. Check the two against each "
                "other before checking this."
            )

    def certificate(self, direction: Any, data: Any, /) -> Any:
        """The optimal ``lambda``: the linear combination of data that bounds.

        Interpretable in its own right — it is the weighting of the
        observations that certifies the bound, and any ``lambda`` at all gives
        a valid one (§18.3(b)). This is the best of them.
        """
        cost = self.dual_cost(direction, data)
        return self._method.minimise(cost, self.data_space.zero()).minimiser

    def is_feasible(self, data: Any, /) -> bool:
        """Whether any model lies in both the prior set and the noise set.

        When both sets are norm balls this is v1's test, matrix-free: the
        damped minimum-norm search of :func:`_minimum_norm_fits`, a few
        warm-started Krylov solves in the data space. For general convex
        sets it is the dual's own diagnosis: an unbounded dual *is* an empty
        primal, which is why :meth:`support` refuses rather than returning
        the large negative number the minimisation was heading towards, and
        this asks the same question without the exception, at the cost of
        one minimisation and with the caveat that a dual which has not yet
        fallen far enough is read as feasible.

        Args:
            data: the observations.

        Returns:
            Whether the feasible set is non-empty.
        """
        if isinstance(self._prior, Ball) and isinstance(self._noise, Ball):
            return _minimum_norm_fits(
                self._problem,
                data,
                noise_radius=self._noise.radius,
                prior_radius=self._prior.radius,
            )
        try:
            self.support(self.target_space.basis_vector(0), data)
        except ValueError:
            return False
        return True

    def __call__(self, data: Any) -> ConvexSet:
        """The feasible property set, as a support-function oracle.

        The oracle raises on data no model can match; :meth:`is_feasible`
        tests that in advance.
        """
        return ConvexSet.from_support_function(
            self.target_space, lambda direction: self.support(direction, data)
        )

    def push_forward(self, operator: LinearOperator, /) -> "_DualRoute":
        """The same inference about a further property."""
        return _DualRoute(
            self._problem,
            operator @ self._target,
            self._prior,
            noise=self._noise,
            method=self._method,
        )


# --------------------------------------------------------------------- #
#                 Membership by the likelihood, §3.3                    #
# --------------------------------------------------------------------- #


def _likelihood_of(noise: ConvexSet) -> tuple[Functional, float]:
    """A confidence set read as ``{ v : l(v) <= level }``.

    The form Al-Attar (2021) §3.3 works in, with ``l`` convex and
    differentiable. A ball is the squared distance to its centre at the
    squared radius; an ellipsoid its Mahalanobis form at one; a sublevel set
    is already in the form. Any other set has no ``l`` to write down.
    """
    space = noise.domain
    if isinstance(noise, Ball):
        centre = noise.centre

        def value(v: Any) -> float:
            return space.squared_norm(space.subtract(v, centre))

        def gradient(v: Any) -> Any:
            return space.scale(2.0, space.subtract(v, centre))

        def hessian(v: Any) -> LinearOperator:
            return (LinearOperator.identity(space) * 2.0).with_traits(
                Traits.POSITIVE_DEFINITE
            )

        return (
            Functional.from_callables(space, value, gradient=gradient, hessian=hessian),
            noise.radius**2,
        )
    if isinstance(noise, Ellipsoid):
        precision, centre = noise.precision, noise.centre

        def value(v: Any) -> float:
            return noise.mahalanobis_squared(v)

        def gradient(v: Any) -> Any:
            return space.scale(2.0, precision(space.subtract(v, centre)))

        def hessian(v: Any) -> LinearOperator:
            return (precision * 2.0).with_traits(Traits.POSITIVE_DEFINITE)

        return (
            Functional.from_callables(space, value, gradient=gradient, hessian=hessian),
            1.0,
        )
    if isinstance(noise, SublevelSet):
        return noise.functional, noise.level
    raise TypeError(
        f"The likelihood route needs the confidence set as a sublevel set of a "
        f"differentiable convex functional -- a Ball, an Ellipsoid or a "
        f"SublevelSet -- not a {type(noise).__name__}."
    )


class _LikelihoodRoute:
    """Membership of a property value, by Al-Attar (2021) §3.3.

    The confidence set is ``{ v : l(v) <= s^2 }`` for a convex,
    differentiable ``l``, the negative log-likelihood in the Gaussian case
    and anything of that shape otherwise. The smallest model that has a
    given property and fits the data within the set is found by a Lagrange
    multiplier on the likelihood constraint: for each ``eta > 0`` the
    convex functional ``||u||^2 / 2 + eta l(v - A u)`` has one minimiser,
    and by Lemma 3.1 its misfit ``l(v - A u_eta)`` is non-increasing in
    ``eta``, so the ``eta`` at which it meets ``s^2`` is a monotone scalar
    root find, the same kernel as the discrepancy principle's. A misfit
    that never reaches ``s^2`` however large ``eta`` grows is the
    constructive proof that no model fits at all.

    A fixed property confines the model to ``u~ + ker T`` with ``u~`` the
    minimum-norm model having that property, and the same problem is solved
    in the kernel by replacing ``A*`` with ``P A*``, eq. (3.28). The norms
    separate, and the answer is ``sqrt(||u~||^2 + ||u_eta||^2)``.

    Each probe is one convex minimisation, warm-started from the last. For
    a quadratic ``l``, a ball or an ellipsoid, that is a linear solve and
    Newton takes it in a step or two; for a general ``l`` it is Newton or
    L-BFGS proper, whichever the functional's derivatives allow. Nothing is
    assembled; the operators are only applied.
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        target: LinearOperator,
        prior: Ball,
        noise: ConvexSet,
        /,
        *,
        solver: LinearSolver | None = None,
        optimiser: Any = None,
        iterations: int = 60,
        rtol: float = 1e-6,
    ) -> None:
        from ..numerics.optimisation import LBFGS, NewtonCG

        self._problem = problem
        self._target = target
        self._radius = _ball_radius(prior, "The prior")
        self._likelihood, self._level = _likelihood_of(noise)
        self._solver = solver or CGSolver(rtol=1e-12)
        if optimiser is None:
            optimiser = (
                NewtonCG(forcing=1e-3, rtol=1e-10, gtol=0.0)
                if self._likelihood.has_hessian
                else LBFGS(rtol=1e-10, gtol=0.0)
            )
        self._optimiser = optimiser
        self._iterations = iterations
        self._rtol = rtol

    @cached_property
    def _kernel(self) -> LinearOperator:
        return OrthogonalProjector.onto_kernel(self._target, solver=self._solver)

    @cached_property
    def _property_pseudo_inverse(self) -> LinearOperator:
        """``T* (T T*)^-1``: the smallest model with a given property."""
        normal = (self._target @ self._target.adjoint).with_traits(
            Traits.POSITIVE_DEFINITE
        )
        return self._target.adjoint @ CholeskySolver()(normal)

    def _objective(
        self, base: Any, projector: LinearOperator, multiplier: float | None
    ) -> Functional:
        """``||u||^2 / 2 + eta l(base - A P u)`` on the model space.

        With ``multiplier=None`` the norm term is dropped and this is the
        misfit alone, whose minimum is the limit the multiplier search
        approaches: eq. (3.15)'s infimum, and the test of whether any model
        reaches the confidence set at all.
        """
        space = self._problem.model_space
        data_space = self._problem.data_space
        forward = self._problem.forward_operator
        likelihood = self._likelihood

        def misfit_point(u: Any) -> Any:
            return data_space.subtract(base, forward(projector(u)))

        def value(u: Any) -> float:
            fit = likelihood(misfit_point(u))
            if multiplier is None:
                return fit
            return multiplier * fit + 0.5 * space.squared_norm(u)

        def gradient(u: Any) -> Any:
            pulled = projector(forward.adjoint(likelihood.gradient(misfit_point(u))))
            if multiplier is None:
                return space.scale(-1.0, pulled)
            return space.axpy(-multiplier, pulled, space.copy(u))

        hessian = None
        if likelihood.has_hessian:

            def hessian(u: Any) -> LinearOperator:
                curvature = likelihood.hessian(misfit_point(u))
                pulled = projector @ forward.adjoint @ curvature @ forward @ projector
                if multiplier is None:
                    return pulled.with_traits(Traits.POSITIVE_SEMIDEFINITE)
                return (
                    LinearOperator.identity(space) + pulled * multiplier
                ).with_traits(Traits.POSITIVE_DEFINITE)

        return Functional.from_callables(
            space, value, gradient=gradient, hessian=hessian
        )

    def _fit(self, base: Any, projector: LinearOperator) -> Any | None:
        """The smallest ``u`` in the projector's range with ``l(base - A P u) <= s^2``.

        ``None`` when there is none. Decided in three steps, as the paper's
        remark after Lemma 3.1 suggests: the misfit's limit as the multiplier
        grows is the unconstrained minimum of ``l(base - A P u)``, so that is
        minimised first, from zero, which for a quadratic ``l`` gives the
        minimum-norm minimiser. A limit above the level proves nothing
        reaches the set; a limit at the level, to tolerance, makes that
        minimiser the answer, the root lying at infinity; and a limit below
        it guarantees a root at a finite multiplier, which the monotone
        search then brackets without ever needing a multiplier large enough
        to overflow the Newton system. Searching first and reading
        exhaustion afterwards, the other way round, widened the multiplier by
        two hundred decades on an unreachable value and met NaN on the way.
        """
        space = self._problem.model_space
        data_space = self._problem.data_space
        forward = self._problem.forward_operator
        level = self._level

        limit = self._optimiser.minimise(
            self._objective(base, projector, None), space.zero()
        )
        floor = float(limit.value)
        if floor > level * (1.0 + self._rtol):
            return None
        if floor >= level * (1.0 - self._rtol):
            return limit.minimiser

        def probe(multiplier: float, previous: Any) -> Evaluation:
            start = space.zero() if previous is None else previous
            result = self._optimiser.minimise(
                self._objective(base, projector, multiplier), start
            )
            model = result.minimiser
            misfit = float(
                self._likelihood(data_space.subtract(base, forward(projector(model))))
            )
            return Evaluation(misfit, model, result.iterations)

        found = monotone_root(
            probe,
            level,
            decreasing=True,
            iterations=self._iterations,
            rtol=self._rtol,
            expansions=40,
        )
        if found.breakdown is not None:
            raise found.breakdown
        if not found.converged:
            raise ValueError(
                "The likelihood route could not bracket its multiplier although "
                f"the misfit's limit {floor:.3g} lies below the level {level:.3g}."
            )
        return found.solution

    def fitting_model(self, data: Any, /) -> Any | None:
        """The smallest model fitting the data within the confidence set.

        ``None`` when no model does, eq. (3.15) with an infinite infimum.
        The prior plays no part: this is the data against the confidence
        set alone.
        """
        space = self._problem.model_space
        if self._likelihood(data) <= self._level:
            return space.zero()
        return self._fit(data, LinearOperator.identity(space))

    def is_feasible(self, data: Any, /) -> bool:
        """Whether a model within the prior ball fits the data within the set."""
        model = self.fitting_model(data)
        return (
            model is not None and self._problem.model_space.norm(model) <= self._radius
        )

    def inclusion_norm(self, value: Any, data: Any, /) -> float:
        """``min { ||m|| : T m == value, l(d - A m) <= s^2 }``.

        Infinite when no model reproduces the value and fits the data, which
        is a proof rather than a failure.
        """
        space = self._problem.model_space
        data_space = self._problem.data_space
        forward = self._problem.forward_operator

        anchor = self._property_pseudo_inverse(value)
        anchor_norm = space.norm(anchor)
        residual = data_space.subtract(data, forward(anchor))
        if self._likelihood(residual) <= self._level:
            return float(anchor_norm)
        correction = self._fit(residual, self._kernel)
        if correction is None:
            return float("inf")
        return float(np.sqrt(anchor_norm**2 + space.squared_norm(correction)))

    def admits(self, value: Any, data: Any, /, *, rtol: float = 1e-8) -> bool:
        """Whether a property value is consistent with the data and the prior.

        Args:
            value: the property value to test.
            data: the observations.
            rtol: how far outside the prior radius still counts as admissible.

        Returns:
            Whether the value is admissible.
        """
        return self.inclusion_norm(value, data) <= self._radius * (1.0 + rtol)


# --------------------------------------------------------------------- #
#                      The one estimator, routes inside                 #
# --------------------------------------------------------------------- #


class _FeasibleSupport(SupportFunction):
    """The support function of a feasible property set, answered by it."""

    def __init__(self, feasible: "FeasiblePropertySet", /) -> None:
        super().__init__(feasible.domain)
        self._set = feasible

    @property
    def has_subgradient(self) -> bool:
        """Only where the route exhibits the maximiser."""
        return self._set.has_maximiser

    def _value(self, y: Any) -> float:
        return self._set.support(y)

    def _maximiser(self, y: Any) -> Any:
        return self._set.support_maximiser(y)


class FeasiblePropertySet(ConvexSet):
    """The set of property values consistent with the data and the prior.

    What :class:`BackusGilbertParker` returns, and the object every question
    is asked of, as a posterior measure is on the Bayesian side. It is a
    :class:`~pygeoinf2.geometry.convex.ConvexSet` and declares what it can
    do. It always has a **support function**, computed by the estimator's
    route: :meth:`support` in a direction, :meth:`support_values` in many,
    :meth:`polytope` for the outer bound they give, :meth:`outside` for a
    certificate of exclusion. It has a **level function** when membership
    can be decided (a ball prior; a confidence set that is a ball, an
    ellipsoid or a sublevel set): :meth:`contains` and :meth:`admits`,
    :meth:`inclusion_norm` and :meth:`level_function`, :meth:`extent` for
    inner bounds along a line, :meth:`inner_hull` for an inner polytope. It
    has a **maximiser** on the closed form and the bisection, with
    :meth:`extremal_model` the model attaining it, and a **projection** on
    the closed form alone, where it is an ellipsoid and :attr:`ellipsoid`
    hands that over with everything an ellipsoid has.

    Nothing is computed at construction; each answer is computed when asked
    and the expensive ones are kept. The set may be empty, when no model
    within the prior fits the data: :meth:`is_empty` says so, and the
    probes that need a point of the set raise ``ValueError`` in that case.
    """

    def __init__(self, estimator: "BackusGilbertParker", data: Any, /) -> None:
        super().__init__(estimator.target_space)
        self._estimator = estimator
        self._data = data

    # ----------------------------------------------------------------- #
    #                            What it holds                          #
    # ----------------------------------------------------------------- #

    @property
    def estimator(self) -> "BackusGilbertParker":
        """The estimator that produced it."""
        return self._estimator

    @property
    def data(self) -> Any:
        """The observations it was produced from."""
        return self._data

    @property
    def route(self) -> str:
        """The route computing the support function."""
        return self._estimator.route

    @property
    def membership(self) -> str | None:
        """How membership is decided, or ``None`` when it cannot be."""
        return self._estimator.membership

    @property
    def _algorithm(self) -> Any:
        return self._estimator.algorithm

    @cached_property
    def ellipsoid(self) -> Ellipsoid:
        """The set as an ellipsoid, on the closed form.

        Raises:
            NotImplementedError: on any other route, where the set is not an
                ellipsoid.
            ValueError: if the set is empty.
        """
        if self.route != "closed_form":
            raise NotImplementedError(
                f"On the {self.route!r} route the feasible property set is not "
                "an ellipsoid; it is known through its support function."
            )
        return self._algorithm(self._data)

    # ----------------------------------------------------------------- #
    #                          What it can do                           #
    # ----------------------------------------------------------------- #

    @property
    def has_membership(self) -> bool:
        """Whether :meth:`contains` decides membership."""
        return self._estimator._inclusion is not None

    @property
    def has_projection(self) -> bool:
        """Whether :meth:`project` is available."""
        return self.route == "closed_form"

    @property
    def has_support_function(self) -> bool:
        """Whether :meth:`support_function` is available."""
        return True

    @property
    def has_maximiser(self) -> bool:
        """Whether :meth:`support_maximiser` can exhibit a point."""
        return self.route in ("closed_form", "bisection")

    @property
    def has_level_function(self) -> bool:
        """Whether a level function describes the set."""
        return self.has_membership

    @cached_property
    def _nonempty(self) -> bool:
        return bool(self._algorithm.is_feasible(self._data))

    def is_empty(self) -> bool:
        """Whether no model within the prior fits the data.

        Decided by the route: the closed form by its budget, the bisection
        by a matrix-free minimum-norm search, the dual by its own diagnosis
        of an unbounded dual. Computed once and kept.
        """
        return not self._nonempty

    # ----------------------------------------------------------------- #
    #                          The support side                         #
    # ----------------------------------------------------------------- #

    def support(self, direction: Any, /) -> float:
        """The support value in one direction.

        Raises:
            ValueError: if the set is empty.
        """
        route = self.route
        if route == "closed_form":
            return float(self.ellipsoid.support_function()(direction))
        if route in ("bisection", "dual"):
            return float(self._algorithm.support(direction, self._data))
        return float(
            self._algorithm.support_values([direction], self._data, route=route)[0]
        )

    def support_values(
        self, directions: Sequence[Any], /, **options: Any
    ) -> np.ndarray:
        """The support values in many directions.

        On a general route this is the dual engine's sweep, with its warm
        start across neighbouring directions and its ``route=``,
        ``warm_start=`` and ``n_jobs=`` options, the route defaulting to
        this set's. The closed form and the bisection have no state to
        carry between directions and evaluate each in turn; they take no
        options.

        Args:
            directions: the directions to evaluate.
            **options: the sweep's options, on a general route.

        Returns:
            One support value per direction.

        Raises:
            TypeError: if options are given on a route that has none.
        """
        if self.route in _GENERAL:
            options.setdefault("route", self.route)
            return self._algorithm.support_values(directions, self._data, **options)
        if options:
            raise TypeError(
                f"The {self.route!r} route sweeps directions one at a time and "
                f"takes no options; got {sorted(options)}."
            )
        return np.array([self.support(direction) for direction in directions])

    def support_function(self) -> SupportFunction:
        """The support function, answered by :meth:`support`."""
        return _FeasibleSupport(self)

    def support_maximiser(self, direction: Any, /) -> Any:
        """The property value attaining the support: the extremal model's.

        Raises:
            NotImplementedError: on a general route, which finds the value
                by duality and exhibits no point.
        """
        if self.route == "closed_form":
            return self.ellipsoid.support_maximiser(direction)
        return self._estimator.target(self.extremal_model(direction))

    def extremal_model(self, direction: Any, /) -> Any:
        """The model of the feasible set furthest along a direction.

        On the closed form, the minimum-norm model plus what the prior's
        budget allows along the kernel component of ``T* q``; on the
        bisection, the model its two multipliers fix. Not generally unique,
        but the bound it attains is.

        Raises:
            NotImplementedError: on a general route.
            ValueError: if the set is empty.
        """
        route = self.route
        if route == "bisection":
            return self._algorithm.extremal_model(direction, self._data)
        if route != "closed_form":
            raise NotImplementedError(
                f"The {route!r} route finds support values by duality and does "
                "not exhibit the model attaining them; the closed form and the "
                "bisection do."
            )
        space = self._estimator.problem.model_space
        budget = self._algorithm.budget(self._data)
        if budget < 0.0:
            raise ValueError("The feasible set is empty; there is no extremal model.")
        anchor = self._algorithm.minimum_norm_model(self._data)
        pulled = self._algorithm._kernel(self._estimator.target.adjoint(direction))
        length = space.norm(pulled)
        if length == 0.0:
            return anchor
        return space.axpy(np.sqrt(budget) / length, pulled, space.copy(anchor))

    def certificate(self, direction: Any, /) -> Any:
        """The optimal dual certificate, on a general route.

        Raises:
            NotImplementedError: on the closed form and the bisection.
        """
        if self.route not in _GENERAL:
            raise NotImplementedError(
                "A certificate belongs to the dual route; the closed form and "
                "the bisection work in the primal."
            )
        return self._algorithm.certificate(direction, self._data)

    def polytope(self, directions: Sequence[Any], /) -> Polytope:
        """A certified outer bound, one half-space per direction (§18.4)."""
        directions = tuple(directions)
        values = self.support_values(directions)
        return Polytope(
            self.domain,
            [
                HalfSpace(self.domain, direction, offset=float(value))
                for direction, value in zip(directions, values)
            ],
            outer=True,
        )

    def outside(
        self, x: Any, directions: Sequence[Any], /, *, rtol: float = 1e-9
    ) -> bool:
        """True when some direction proves the point is not in the set.

        One-sided: a ``True`` is a proof, a ``False`` only the absence of one
        among the directions tried. :meth:`contains` decides, when it can.

        Args:
            x: the point to test.
            directions: the directions to try.
            rtol: how far past a supporting hyperplane the point must lie.
        """
        for direction in directions:
            pairing = self.domain.inner_product(direction, x)
            if pairing > self.support(direction) * (1.0 + rtol) + rtol:
                return True
        return False

    # ----------------------------------------------------------------- #
    #                        The level-function side                    #
    # ----------------------------------------------------------------- #

    def _inclusion(self, what: str) -> Any:
        return self._estimator._need_inclusion(what)

    @property
    def level(self) -> float:
        """The level the inclusion norm is bounded by: a ball prior's radius, an ellipsoid's one."""
        self._inclusion("The level")
        return self._estimator._prior_level

    def inclusion_norm(self, value: Any, /) -> float:
        """``min { ||m|| : T m == value, A m fits the data }``, the cost of a value.

        §18.5: a value is admissible exactly when this is within the prior
        radius. Infinite when no model reproduces the value and fits the
        data, which is a proof of inadmissibility rather than a failure.

        Raises:
            NotImplementedError: unless membership can be decided.
        """
        return self._inclusion("The inclusion norm").inclusion_norm(value, self._data)

    def admits(self, value: Any, /, *, rtol: float = 1e-8) -> bool:
        """Whether a property value is consistent with the data and the prior.

        Args:
            value: the property value to test.
            rtol: how far outside the prior radius still counts.

        Raises:
            NotImplementedError: unless membership can be decided.
        """
        return self._inclusion("Membership").admits(value, self._data, rtol=rtol)

    def contains(self, x: Any, /, *, rtol: float = 1e-8) -> bool:
        """Membership, by the inclusion norm.

        Args:
            x: a property value.
            rtol: how far outside the prior radius still counts.

        Raises:
            NotImplementedError: unless membership can be decided; see
                :attr:`has_membership`. :meth:`outside` gives a certificate
                of exclusion from the support function alone.
        """
        return self.admits(x, rtol=rtol)

    def level_function(self) -> Functional:
        """``p -> inclusion_norm(p)``, whose sublevel set at :attr:`level` is this set.

        Convex on the property space (Al-Attar 2021 §2.3): the sublevel-set
        characterisation, as :meth:`support_function` is the other.
        """
        self._inclusion("The level function")
        return Functional.from_callables(self.domain, self.inclusion_norm)

    def fitting_model(self) -> Any:
        """The smallest model fitting the data within the confidence set.

        Its property is a point of this set whenever the set is non-empty,
        and the point :meth:`extent` draws its lines through.

        Raises:
            NotImplementedError: unless membership can be decided.
            ValueError: if no model within the prior fits the data.
        """
        engine = self._inclusion("The fitting model")
        if self.is_empty():
            raise ValueError(
                "No model within the prior fits the data; the set is empty."
            )
        if self.membership == "closed_form":
            return engine.minimum_norm_model(self._data)
        if self.membership == "reduced":
            model = engine.fitting_model(self._data)
        else:
            model = self._estimator._likelihood_engine.fitting_model(self._data)
        if model is None:
            raise ValueError(
                "No model within the prior fits the data; the set is empty."
            )
        return model

    def extent(self, direction: Any, /, *, iterations: int = 40) -> tuple[float, float]:
        """How far the set reaches along a line through an interior point.

        Al-Attar (2021) Fig. 8: along the line through the property of the
        smallest fitting model in the given direction, the values are
        admissible on an interval, since the set is convex, and its two ends
        are found by bracketing and bisecting :meth:`admits`. Returned as the
        pairing ``(direction, p)`` at the two ends, so that they compare
        directly with ``-support(-direction)`` and ``support(direction)``:
        these are **inner** bounds, points of the boundary, where the
        support values are outer ones, and the two coincide when the
        property space is one-dimensional. Each end costs one inclusion norm
        per bisection step.

        Args:
            direction: the direction of the line.
            iterations: bisection steps for each end.

        Returns:
            ``(lower, upper)``.

        Raises:
            NotImplementedError: unless membership can be decided.
            ValueError: if the set is empty.
        """
        self._inclusion("An extent")
        space = self.domain
        base = self._estimator.target(self.fitting_model())
        length = space.squared_norm(direction)
        if length == 0.0:
            value = space.inner_product(direction, base)
            return value, value
        if not self.admits(base):
            raise ValueError(
                "The interior point is not admitted; the set may be empty."
            )

        def crossing(sign: float) -> float:
            step = max(1.0, abs(space.inner_product(direction, base))) / length
            inside, outside = 0.0, sign * step
            for _ in range(60):
                if not self.admits(space.axpy(outside, direction, space.copy(base))):
                    break
                inside, outside = outside, 2.0 * outside
            else:
                raise ValueError("The set appears unbounded along this direction.")
            for _ in range(iterations):
                middle = 0.5 * (inside + outside)
                if self.admits(space.axpy(middle, direction, space.copy(base))):
                    inside = middle
                else:
                    outside = middle
            return inside

        centre = space.inner_product(direction, base)
        return centre + crossing(-1.0) * length, centre + crossing(1.0) * length

    def inner_hull(self, values: Any, /) -> Polytope:
        """The convex hull of whichever candidate values are admissible.

        The *inner* bound of §18.4, and the only thing that produces one: a
        support function can never exhibit a point of the set. Returned as an
        inner :class:`~pygeoinf2.geometry.convex.Polytope`, so it cannot be
        mistaken for the outer one.

        Args:
            values: candidate property values, of which the admissible ones
                are kept.

        Raises:
            ValueError: if fewer candidates are admissible than the property
                space has dimensions, there being no hull to take.
            NotImplementedError: unless membership can be decided.
        """
        from scipy.spatial import ConvexHull

        self._inclusion("An inner hull")
        space = self.domain
        inside = [space.to_components(value) for value in values if self.admits(value)]
        if len(inside) <= space.dim:
            raise ValueError(
                f"Only {len(inside)} of the candidates are admissible, which "
                f"is not enough to bound a hull in {space.dim} dimensions. "
                "Sample nearer the minimum-norm property."
            )
        hull = ConvexHull(np.stack(inside))
        planes = [
            HalfSpace(
                space, space.representer(equation[:-1]), offset=-float(equation[-1])
            )
            for equation in hull.equations
        ]
        return Polytope(space, planes, outer=False)

    # ----------------------------------------------------------------- #

    def project(self, x: Any, /) -> Any:
        """The nearest point, on the closed form; refused elsewhere.

        Raises:
            NotImplementedError: unless the set is an ellipsoid. Bound it
                with :meth:`polytope` and project onto that.
        """
        if self.route != "closed_form":
            raise NotImplementedError(
                "A feasible property set known through its support function "
                "has no projection; bound it with polytope() and project onto "
                "that."
            )
        return self.ellipsoid.project(x)

    def push_forward(self, operator: LinearOperator, /) -> "FeasiblePropertySet":
        """The same set for a further property of the model, ``T' == operator T``."""
        return self._estimator.push_forward(operator)(self._data)

    def __repr__(self) -> str:
        return (
            f"FeasiblePropertySet(route={self.route!r}, membership={self.membership!r})"
        )


_ROUTES = ("auto", "closed_form", "bisection", "dual", "primal", "kkt", "smoothed")
_GENERAL = ("dual", "primal", "kkt", "smoothed")
_MEMBERSHIPS = ("auto", "closed_form", "reduced", "likelihood")


class BackusGilbertParker(SetEstimator):
    """The feasible property set: what the data and a constraint let a property be.

    The core of Backus-Gilbert-Parker inference. In go the forward problem, a
    convex **constraint set** on the model, optionally a convex **confidence
    set** on the data, and the property operator ``T``; out comes a convex set
    on the property space, the image under ``T`` of every model that lies in
    the constraint set and fits the data to within the confidence set (§18.3,
    BGP eqs. 4-5). The confidence set is taken from the problem when not
    given: its own set if it has one, the credible ball at ``level`` if its
    error is a measure, and the single point ``{0}`` if it has no error at
    all. Error-free data are not a separate method; they are the confidence
    set shrunk to a point, and the estimator accounts for that itself.

    **The sets decide the algorithm**, and the choice is made here rather
    than by the caller naming a class:

    * a ball prior and exact data: the **closed form**, Al-Attar (2021) eq.
      (2.84), an ellipsoid costing ``dim(P) + 1`` minimum-norm solves;
    * a ball prior and a ball confidence set: **bisection**, BGP's primal
      route, a damped least-squares solve inside two nested monotone root
      finds per direction, which also produces the extremal model;
    * anything convex: the **dual**, BGP eq. (28), a nonsmooth minimisation
      over the data space per direction, with the primal splitting, the KKT
      and the smoothed solvers as alternatives where they apply.

    ``route=`` names one of these to force it, for testing them against each
    other -- they compute the same set -- or for taking the general route
    where a cheaper one applies; the request is refused when the sets do not
    allow it, with a message saying which route does.

    **Two characterisations of the answer.** A closed convex set is
    determined by its support function (Rockafellar 13.1), and every route
    gives that: :meth:`support` in one direction, :meth:`support_values` in
    many, and :meth:`__call__` returns the set as an object carrying it, an
    :class:`~pygeoinf2.geometry.convex.Ellipsoid` from the closed form and a
    support-function oracle otherwise. A set can also be characterised as
    a **sublevel set**, by a function saying how far a proposed value is
    from acceptable: the minimum norm of a model reproducing the value and
    fitting the data, which is acceptable when within the prior radius
    (§18.5, Al-Attar 2021 §2.3 and §3.3). That is a different computation,
    and ``membership=`` chooses it: the closed form's joint map for exact
    data, the data-space reduction for two balls, or the likelihood route
    for a confidence set given as a sublevel set of any differentiable
    convex functional, a ball, an ellipsoid or a ``SublevelSet``. It needs a
    ball prior. Both characterisations live on the returned
    :class:`FeasiblePropertySet`, which declares which it has: the support
    function bounds the set from outside, the level function decides points
    and gives inner bounds, and not both are required of every algorithm.

    The chosen route's own object is :attr:`algorithm`, for diagnostics that
    belong to one route and not the others: the closed form's budget and
    prior-only ellipsoid, the dual's cost and solvers.

    .. code-block:: python

        feasible = BackusGilbertParker(problem, target, prior)(data)
        feasible.is_empty()
        feasible.support(direction)          # the support-function side
        feasible.contains(value)             # the level-function side
        feasible.extent(direction)           # inner bounds along a line
        feasible.polytope(directions)        # an outer bound
    """

    def __init__(
        self,
        problem: LinearForwardProblem,
        target: LinearOperator,
        prior: ConvexSet,
        /,
        *,
        noise: ConvexSet | None = None,
        level: float = 0.95,
        route: str = "auto",
        membership: str = "auto",
        solver: LinearSolver | None = None,
        iterations: int = 60,
        method: Any = None,
        optimiser: Any = None,
    ) -> None:
        """
        Args:
            problem: the forward problem.
            target: the property operator ``T``, acting on the model space.
            prior: the constraint set on the model space. A
                :class:`~pygeoinf2.geometry.convex.Ball` admits the cheap
                routes; any convex set with a support function and a
                maximiser admits the dual.
            noise: the confidence set on the data space: a convex set, or a
                ``SublevelSet`` of a convex functional, which is taken as
                convex on the caller's word since a sublevel set does not
                know. Taken from the problem when omitted, as described
                above.
            level: the probability the confidence set carries when it is
                hardened from a measure.
            route: ``"auto"`` chooses by the sets. ``"closed_form"``,
                ``"bisection"``, ``"dual"``, ``"primal"``, ``"kkt"`` and
                ``"smoothed"`` force one; the last four are the general
                route's solvers, see the dual sweep's own documentation.
            membership: how membership is decided. ``"auto"`` chooses by
                the sets: ``"closed_form"`` for exact data, ``"reduced"``
                for two balls, ``"likelihood"`` for an ellipsoid or a
                sublevel set; each can be forced where the sets allow it,
                and the likelihood route applies to a ball as well.
            solver: how the closed form, the bisection and the likelihood
                route invert their operators.
            iterations: bisection steps, on each of the two multipliers,
                and the likelihood route's root find.
            method: the general route's minimiser. A proximal bundle method
                by default.
            optimiser: the likelihood route's minimiser for each probe.
                Newton-CG when the confidence set's functional has a
                Hessian, L-BFGS otherwise.

        Raises:
            ValueError: if the route or membership is unknown, or not
                available for these sets; if an operator or set is on the
                wrong space.
            TypeError: if the problem's error is neither a measure nor a
                convex set.
        """
        if route not in _ROUTES:
            raise ValueError(f"route must be one of {_ROUTES}, got {route!r}.")
        if membership not in _MEMBERSHIPS:
            raise ValueError(
                f"membership must be one of {_MEMBERSHIPS}, got {membership!r}."
            )
        if target.domain != problem.model_space:
            raise ValueError("The property operator must act on the model space.")
        if prior.domain != problem.model_space:
            raise ValueError("The constraint set must lie in the model space.")
        noise = self._resolve_noise(problem, noise, level)
        if noise.domain != problem.data_space:
            raise ValueError("The confidence set must lie in the data space.")

        self._problem = problem
        self._target = target
        self._prior = prior
        self._noise = noise
        self._level = level
        self._requested = route
        self._requested_membership = membership
        self._solver = solver
        self._iterations = iterations
        self._method = method
        self._optimiser = optimiser
        self._route = self._choose(route)
        self._algorithm = self._build(self._route)
        self._membership = self._choose_membership(membership)

    # ----------------------------------------------------------------- #
    #                              Dispatch                             #
    # ----------------------------------------------------------------- #

    @staticmethod
    def _resolve_noise(
        problem: LinearForwardProblem, noise: ConvexSet | None, level: float
    ) -> ConvexSet:
        if noise is not None:
            if not isinstance(noise, (ConvexSet, SublevelSet)):
                raise TypeError(
                    "The confidence set must be a ConvexSet, or a SublevelSet of "
                    "a convex functional."
                )
            return noise
        if not problem.has_error:
            return Ball(problem.data_space, radius=0.0)
        error = problem.error
        if isinstance(error, (ConvexSet, SublevelSet)):
            return error
        if isinstance(error, ProbabilityMeasure):
            return problem.error_measure.ambient_ball(level=level)
        raise TypeError(
            "The problem's error is neither a measure nor a convex set; pass "
            "noise= to say what confidence set to use."
        )

    @property
    def _exact(self) -> bool:
        """Whether the confidence set is the single point ``{0}``."""
        return isinstance(self._noise, Ball) and self._noise.radius == 0.0

    @property
    def _balls(self) -> bool:
        return isinstance(self._prior, Ball) and isinstance(self._noise, Ball)

    @property
    def _quadratic(self) -> bool:
        """Both sets have quadratic level functions: a ball or an ellipsoid each."""
        return isinstance(self._prior, (Ball, Ellipsoid)) and isinstance(
            self._noise, (Ball, Ellipsoid)
        )

    @property
    def _prior_level(self) -> float:
        """The prior's level in the units its inclusion norm is reported in."""
        if isinstance(self._prior, Ball):
            return float(self._prior.radius)
        return 1.0

    def _choose(self, route: str) -> str:
        if route == "auto":
            if self._exact and isinstance(self._prior, Ball):
                return "closed_form"
            if self._quadratic:
                return "bisection"
            return "dual"
        if route == "closed_form" and not (
            self._exact and isinstance(self._prior, Ball)
        ):
            raise ValueError(
                "The closed form needs a ball prior and exact data (a confidence "
                "set of radius zero); with these sets the route is "
                f"{self._choose('auto')!r}."
            )
        if route == "bisection":
            if not self._quadratic:
                raise ValueError(
                    "Bisection needs a ball or an ellipsoid on each side, a prior "
                    "and a confidence set with quadratic level functions; a "
                    "general convex set needs the dual route."
                )
            if self._exact:
                raise ValueError(
                    "Bisection needs a confidence set of positive radius: with "
                    "exact data its misfit search has nothing to bracket. The "
                    "closed form is the route for exact data."
                )
        return route

    def _build(self, route: str) -> Any:
        if route == "closed_form":
            return _ClosedFormRoute(
                self._problem, self._target, self._prior, solver=self._solver
            )
        if route == "bisection":
            return _BisectionRoute(
                self._problem,
                self._target,
                self._prior,
                noise=self._noise,
                solver=self._solver,
                iterations=self._iterations,
            )
        return _DualRoute(
            self._problem,
            self._target,
            self._prior,
            noise=self._noise,
            method=self._method,
        )

    @property
    def _likelihood_applies(self) -> bool:
        return (
            isinstance(self._prior, Ball)
            and not self._exact
            and isinstance(self._noise, (Ball, Ellipsoid, SublevelSet))
        )

    def _choose_membership(self, membership: str) -> str | None:
        """Which engine decides membership, or ``None`` when none can."""
        if membership == "auto":
            if self._exact and isinstance(self._prior, Ball):
                return "closed_form"
            if self._quadratic:
                return "reduced"
            if self._likelihood_applies:
                return "likelihood"
            return None
        if membership == "closed_form" and not (
            self._exact and isinstance(self._prior, Ball)
        ):
            raise ValueError(
                "Closed-form membership needs a ball prior and exact data; "
                f"with these sets membership is {self._choose_membership('auto')!r}."
            )
        if membership == "reduced" and not (self._quadratic and not self._exact):
            raise ValueError(
                "Reduced membership needs a ball or an ellipsoid on each side, "
                "with a confidence set of positive size; with these sets "
                f"membership is {self._choose_membership('auto')!r}."
            )
        if membership == "likelihood" and not self._likelihood_applies:
            raise ValueError(
                "Likelihood membership needs a ball prior and a confidence set "
                "that is a sublevel set of a differentiable convex functional "
                "-- a Ball of positive radius, an Ellipsoid or a SublevelSet."
            )
        return membership

    @cached_property
    def _inclusion(self) -> Any:
        """The engine that decides membership, or ``None`` without one.

        Independent of the route computing the support: membership is the
        minimum-norm computation of §18.5, and the support route's own
        object is reused when it happens to be the same engine.
        """
        if self._membership is None:
            return None
        if self._membership == "closed_form":
            return (
                self._algorithm
                if self._route == "closed_form"
                else self._build("closed_form")
            )
        if self._membership == "reduced":
            return (
                self._algorithm
                if self._route == "bisection"
                else self._build("bisection")
            )
        return self._likelihood_engine

    @cached_property
    def _likelihood_engine(self) -> Any:
        return _LikelihoodRoute(
            self._problem,
            self._target,
            self._prior,
            self._noise,
            solver=self._solver,
            optimiser=self._optimiser,
            iterations=self._iterations,
        )

    def _need_inclusion(self, what: str) -> Any:
        engine = self._inclusion
        if engine is None:
            raise NotImplementedError(
                f"{what} needs a ball prior and a confidence set that is a ball, "
                "an ellipsoid or a sublevel set of a differentiable convex "
                "functional; with other sets the feasible property set is "
                "known through its support function only. Use the returned "
                "set's outside() for a certificate of exclusion, or polytope() "
                "for an outer bound."
            )
        return engine

    # ----------------------------------------------------------------- #
    #                            What it holds                          #
    # ----------------------------------------------------------------- #

    @property
    def problem(self) -> LinearForwardProblem:
        """The forward problem."""
        return self._problem

    @property
    def target(self) -> LinearOperator:
        """The property operator ``T``."""
        return self._target

    @property
    def prior(self) -> ConvexSet:
        """The constraint set on the model space."""
        return self._prior

    @property
    def noise(self) -> ConvexSet:
        """The confidence set on the data space, as resolved."""
        return self._noise

    @property
    def route(self) -> str:
        """The route in use: ``"closed_form"``, ``"bisection"`` or a general one."""
        return self._route

    @property
    def membership(self) -> str | None:
        """How membership is decided, or ``None`` when it cannot be."""
        return self._membership

    @property
    def algorithm(self) -> Any:
        """The route's own object, for diagnostics particular to it."""
        return self._algorithm

    @property
    def data_space(self) -> HilbertSpace:
        """The problem's data space."""
        return self._problem.data_space

    @property
    def target_space(self) -> HilbertSpace:
        """The property space."""
        return self._target.codomain

    # ----------------------------------------------------------------- #

    def __call__(self, data: Any) -> "FeasiblePropertySet":
        """The feasible property set for these data, as an object to probe.

        Nothing is computed here. The set holds the engines and the data,
        and each question -- a support value, membership, the extent, the
        inner hull, emptiness -- is answered when asked and remembered.
        This is the shape of the Bayesian side, where the estimator returns
        a measure and the measure is what gets probed.
        """
        return FeasiblePropertySet(self, data)

    # ----------------------------------------------------------------- #

    def push_forward(self, operator: LinearOperator, /) -> "BackusGilbertParker":
        """The same inference about a further property of the model."""
        return BackusGilbertParker(
            self._problem,
            operator @ self._target,
            self._prior,
            noise=self._noise,
            level=self._level,
            route=self._requested,
            membership=self._requested_membership,
            solver=self._solver,
            iterations=self._iterations,
            method=self._method,
            optimiser=self._optimiser,
        )

    def __repr__(self) -> str:
        return (
            f"BackusGilbertParker(route={self._route!r}, prior={self._prior!r}, "
            f"noise={self._noise!r})"
        )
