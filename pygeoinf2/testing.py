"""
Numerical checks on the axioms the library assumes but cannot enforce.

These are free functions for test suites, kept out of the classes they check.

Every check raises ``AssertionError`` with a message naming the axiom that
failed, and on success returns a :class:`CheckReport` carrying the worst
relative defect it measured for each identity it tested. The report is what a
backend over an inexact adjoint wants: pass an ``rtol`` the implementation can
meet, read the defect back, and record it falling under refinement. The
default tolerances suit an exact implementation in double precision.

Probe vectors come from the space's ``random`` method unless a ``measure=`` on
the space is passed, in which case they are draws from it. ``random`` is the
neutral default on a space with no further structure, and an unrealistic one
on a function space, where it is rough in every mode at once and not the kind
of vector an operator will meet; a check can then pass or fail for the wrong
reason. A prior on the space is usually the better source of probes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.random import Generator

from .algebra.operators import Functional, LinearOperator, Operator
from .algebra.spaces import CoordinateSpace, HilbertSpace, resolve_rng
from .geometry.convex import ConvexSet
from .probability.base import ProbabilityMeasure
from .traits import Traits

__all__ = [
    "CheckReport",
    "check_space",
    "check_coordinates",
    "check_white_noise",
    "check_representer",
    "check_operator",
    "check_traits",
    "check_derivative",
    "check_gradient",
    "check_second_derivative",
    "check_measure",
    "check_projection",
    "check_convexity",
    "check_affine",
]


@dataclass(frozen=True)
class CheckReport:
    """What a check measured on the way to passing.

    ``defects`` maps each identity tested to the worst *relative* defect
    observed for it -- the residual over the scale it was judged against, the
    same quantity the check compared with its tolerance. A failure raises
    before any report is made, so a report in hand means the check passed;
    the defects say by how much. For a backend whose adjoint is consistent
    only to discretisation level, the adjoint defect under a loose ``rtol``
    is the number to record as the mesh refines, and the floor below which a
    solver tolerance is not meaningful.
    """

    check: str
    defects: dict[str, float] = field(default_factory=dict)

    @property
    def worst(self) -> float:
        """The largest recorded defect, zero when nothing was recorded."""
        return max(self.defects.values(), default=0.0)

    def __repr__(self) -> str:
        return f"CheckReport({self.check}, worst={self.worst:.3g})"


def _record(record: dict[str, float] | None, axiom: str, defect: float) -> None:
    if record is not None:
        record[axiom] = max(record.get(axiom, 0.0), float(defect))


def _draw(
    space: HilbertSpace, rng: Generator | None, measure: ProbabilityMeasure | None
) -> object:
    """A probe vector: the space's ``random`` vector, or a draw from *measure*.

    See the module docstring for why a measure is often the better source.

    Raises:
        ValueError: if the measure lives on another space.
    """
    if measure is None:
        return space.random(rng=rng)
    if measure.domain != space:
        raise ValueError(
            f"The measure lives on {measure.domain!r}, and the probes are "
            f"drawn on {space!r}."
        )
    return measure.sample(rng=rng)


def _fail(axiom: str, detail: str) -> None:
    raise AssertionError(f"Axiom failed: {axiom}. {detail}")


def _assert_close(
    space: HilbertSpace,
    a: object,
    b: object,
    axiom: str,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    record: dict[str, float] | None = None,
) -> None:
    """Assert two vectors agree, measured in the space's own norm."""
    residual = space.norm(space.subtract(a, b))
    scale = max(space.norm(a), space.norm(b), 1.0)
    _record(record, axiom, residual / scale)
    if residual > atol + rtol * scale:
        _fail(axiom, f"residual {residual:g} against scale {scale:g}")


def _assert_scalar(
    value: float,
    expected: float,
    axiom: str,
    *,
    rtol: float = 1e-10,
    record: dict[str, float] | None = None,
) -> None:
    scale = max(abs(value), abs(expected), 1.0)
    _record(record, axiom, abs(value - expected) / scale)
    if abs(value - expected) > rtol * scale:
        _fail(axiom, f"{value!r} != {expected!r}")


def check_space(
    space: HilbertSpace,
    /,
    *,
    rng: Generator | None = None,
    measure: ProbabilityMeasure | None = None,
    trials: int = 5,
    rebuild: Callable[[], HilbertSpace] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> CheckReport:
    """Check the vector space and inner product axioms.

    Args:
        space: the space to check.
        rng: generator used to draw test vectors.
        measure: a measure on the space to draw the probe vectors from, in
            place of the space's ``random`` method; see the module docstring.
        trials: how many random triples to test.
        rebuild: optional zero-argument callable returning an independently
            constructed space that should compare equal to ``space``. Supply it
            to catch identity-based equality, which silently produces spurious
            "domain mismatch" errors later.
        rtol: the agreement required of each identity.
        atol: an absolute allowance on vector comparisons.

    Returns:
        The report, carrying the worst defect of each axiom.
    """
    rng = resolve_rng(rng)
    defects: dict[str, float] = {}

    def close(a: object, b: object, axiom: str) -> None:
        _assert_close(space, a, b, axiom, rtol=rtol, atol=atol, record=defects)

    def scalar(value: float, expected: float, axiom: str) -> None:
        _assert_scalar(value, expected, axiom, rtol=rtol, record=defects)

    if space.dim < 0:
        _fail("dimension is non-negative", f"dim == {space.dim}")

    # --- identity ---------------------------------------------------------
    if space != space:
        _fail("a space equals itself", repr(space))
    try:
        hash(space)
    except TypeError as error:
        _fail("a space is hashable", str(error))
    if rebuild is not None:
        other = rebuild()
        if space != other:
            _fail(
                "structurally identical spaces are equal",
                f"{space!r} != {other!r}; the key is probably built from an "
                f"object compared by identity",
            )
        if hash(space) != hash(other):
            _fail("equal spaces hash equally", f"{space!r} vs {other!r}")

    if space.dim == 0:
        return CheckReport("check_space", defects)

    for _ in range(trials):
        x, y, z = (
            _draw(space, rng, measure),
            _draw(space, rng, measure),
            _draw(space, rng, measure),
        )
        a, b = float(rng.normal()), float(rng.normal())

        # --- vector space axioms -----------------------------------------
        close(space.add(x, y), space.add(y, x), "addition commutes")
        close(
            space.add(space.add(x, y), z),
            space.add(x, space.add(y, z)),
            "addition associates",
        )
        close(space.add(x, space.zero()), x, "zero is the additive identity")
        close(
            space.add(x, space.negative(x)),
            space.zero(),
            "negation inverts addition",
        )
        close(space.scale(1.0, x), x, "scaling by one is the identity")
        close(
            space.scale(a, space.scale(b, x)),
            space.scale(a * b, x),
            "scalar multiplication associates",
        )
        close(
            space.scale(a, space.add(x, y)),
            space.add(space.scale(a, x), space.scale(a, y)),
            "scaling distributes over addition",
        )
        close(
            space.scale(a + b, x),
            space.add(space.scale(a, x), space.scale(b, x)),
            "scaling distributes over scalar addition",
        )

        # --- inner product axioms ----------------------------------------
        scalar(
            space.inner_product(x, y),
            space.inner_product(y, x),
            "the inner product is symmetric",
        )
        scalar(
            space.inner_product(space.add(x, y), z),
            space.inner_product(x, z) + space.inner_product(y, z),
            "the inner product is additive",
        )
        scalar(
            space.inner_product(space.scale(a, x), y),
            a * space.inner_product(x, y),
            "the inner product is homogeneous",
        )
        if space.squared_norm(x) <= 0.0:
            _fail(
                "the inner product is positive definite",
                f"||x||^2 == {space.squared_norm(x)}",
            )
        scalar(
            space.norm(x) ** 2,
            space.squared_norm(x),
            "the norm is the root of the squared norm",
        )

        # --- in-place operations agree with their out-of-place forms ------
        expected = space.add(x, space.scale(a, y))
        target = space.copy(x)
        result = space.axpy(a, y, target)
        close(result, expected, "axpy agrees with add and scale")

        expected = space.scale(a, x)
        target = space.copy(x)
        result = space.scale_inplace(a, target)
        close(result, expected, "scale_inplace agrees with scale")

        # --- copy is independent ------------------------------------------
        original = space.copy(x)
        duplicate = space.copy(x)
        space.scale_inplace(2.0, duplicate)
        close(x, original, "copy does not alias its source")

    # --- derived helpers ---------------------------------------------------
    vectors = [_draw(space, rng, measure) for _ in range(min(3, space.dim))]
    orthonormal = space.gram_schmidt(vectors)
    for i, u in enumerate(orthonormal):
        for j, v in enumerate(orthonormal):
            scalar(
                space.inner_product(u, v),
                1.0 if i == j else 0.0,
                "gram_schmidt returns an orthonormal set",
            )

    samples = [_draw(space, rng, measure) for _ in range(4)]
    total = space.zero()
    for s in samples:
        total = space.axpy(0.25, s, total)
    close(space.mean(samples), total, "mean is the average of its arguments")
    return CheckReport("check_space", defects)


def check_coordinates(
    space: CoordinateSpace,
    /,
    *,
    rng: Generator | None = None,
    measure: ProbabilityMeasure | None = None,
    trials: int = 5,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> CheckReport:
    """Check the coordinate map, the Gram matrix, and the pairing axiom.

    Args:
        space: the space to check.
        rng: the generator for the probe vectors.
        measure: a measure on the space to draw the probe vectors from, in
            place of the space's ``random`` method; see the module docstring.
        trials: how many random vectors to check each axiom on. More than one
            because a single draw can satisfy an identity by accident.
        rtol: the agreement required of each identity. The ``solve_gram``
            round trip is allowed ``1e-8`` even under a tighter ``rtol``,
            because a Gram solve may be iterative.
        atol: an absolute allowance on vector comparisons.

    Returns:
        The report, carrying the worst defect of each axiom.

    Raises:
        AssertionError: naming the axiom that failed and the numbers that
            failed it.
    """
    rng = resolve_rng(rng)
    defects: dict[str, float] = {}

    if not isinstance(space, CoordinateSpace):
        _fail(
            "the space provides coordinates",
            f"{type(space).__name__} is not a CoordinateSpace",
        )
    if space.dim == 0:
        return CheckReport("check_coordinates", defects)

    gram = space.gram_matrix()
    if not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-12):
        _fail(
            "the Gram matrix is symmetric", f"asymmetry {np.abs(gram - gram.T).max():g}"
        )
    eigenvalues = np.linalg.eigvalsh(gram)
    if eigenvalues.min() <= 0.0:
        _fail(
            "the Gram matrix is positive definite",
            f"least eigenvalue {eigenvalues.min():g}",
        )
    if space.is_orthonormal and not np.allclose(gram, np.identity(space.dim)):
        _fail("an orthonormal basis has an identity Gram matrix", "it does not")

    for _ in range(trials):
        x, y = _draw(space, rng, measure), _draw(space, rng, measure)
        cx = space.to_components(x)
        cy = space.to_components(y)
        a = float(rng.normal())

        if cx.shape != (space.dim,):
            _fail("components have shape (dim,)", f"got {cx.shape} for dim {space.dim}")

        _assert_close(
            space,
            space.from_components(cx),
            x,
            "the coordinate map round-trips",
            rtol=rtol,
            atol=atol,
            record=defects,
        )
        if not np.allclose(space.to_components(space.from_components(cx)), cx):
            _fail("components round-trip", "to_components(from_components(c)) != c")

        if not np.allclose(space.to_components(space.add(x, y)), cx + cy):
            _fail("the coordinate map is additive", "to_components(x + y) != c_x + c_y")
        if not np.allclose(space.to_components(space.scale(a, x)), a * cx):
            _fail("the coordinate map is homogeneous", "to_components(a x) != a c_x")

        _assert_scalar(
            space.inner_product(x, y),
            float(cx @ gram @ cy),
            "the inner product agrees with the Gram matrix",
            rtol=rtol,
            record=defects,
        )
        if not np.allclose(space.apply_gram(cx), gram @ cx):
            _fail("apply_gram agrees with the Gram matrix", "they differ")
        if not np.allclose(
            space.solve_gram(space.apply_gram(cx)),
            cx,
            rtol=max(rtol, 1e-8),
            atol=max(atol, 1e-10),
        ):
            _fail("solve_gram inverts apply_gram", "the round trip does not return c")

        nested = check_representer(space, cx, rng=rng, measure=measure, rtol=rtol)
        for axiom, defect in nested.defects.items():
            _record(defects, axiom, defect)

    for i in range(min(space.dim, 4)):
        e = space.basis_vector(i)
        expected = np.zeros(space.dim)
        expected[i] = 1.0
        if not np.allclose(space.to_components(e), expected):
            _fail("basis_vector has the expected components", f"index {i}")
    return CheckReport("check_coordinates", defects)


def check_representer(
    space: CoordinateSpace,
    derivative_components: np.ndarray,
    /,
    *,
    rng: Generator | None = None,
    measure: ProbabilityMeasure | None = None,
    trials: int = 3,
    rtol: float = 1e-10,
) -> CheckReport:
    """Check the pairing axiom that separates a derivative from a gradient.

    For a functional given by ``g`` in the derivative convention — the array a
    numerical adjoint method returns, acting as ``x -> g . c_x`` — its Riesz
    representer ``v`` must satisfy ``(v, x) == g . c_x`` for every ``x``.

    Handing ``g`` straight to an optimizer as if it were the gradient fails this
    identity by a factor of the Gram matrix, which is exactly the classic
    adjoint-method error. On an orthonormal basis the two coincide and there is
    nothing to catch.

    Args:
        space: the space to check.
        rng: the generator for the probe vectors.
        measure: a measure on the space to draw the probe vectors from, in
            place of the space's ``random`` method; see the module docstring.
        trials: how many random functionals to check.
        rtol: the agreement required.

    Returns:
        The report, carrying the worst pairing defect.

    Raises:
        AssertionError: if a representer does not reproduce its functional.
    """
    rng = resolve_rng(rng)
    defects: dict[str, float] = {}
    g = np.asarray(derivative_components, dtype=float)
    v = space.representer(g)

    for _ in range(trials):
        x = _draw(space, rng, measure)
        _assert_scalar(
            space.inner_product(v, x),
            float(g @ space.to_components(x)),
            "the representer pairs with vectors as the derivative does",
            rtol=rtol,
            record=defects,
        )
    return CheckReport("check_representer", defects)


def check_white_noise(
    space: HilbertSpace,
    /,
    *,
    rng: Generator | None = None,
    measure: ProbabilityMeasure | None = None,
    samples: int = 20000,
    rtol: float = 0.06,
) -> CheckReport:
    """Check that ``white_noise`` really has identity covariance on the space.

    Tests ``E[(x, u) (x, v)] == (u, v)`` over pairs of basis-like directions.
    This is a statistical check, so it needs a seeded generator and a tolerance
    that scales like ``1 / sqrt(samples)``.

    An entry is judged against ``sqrt((u, u) (v, v))``, as :func:`check_measure`
    judges a covariance, and not against one. The products ``(x, u)`` and
    ``(x, v)`` are jointly normal with those variances, so the sampling error
    of their mean product is at most ``sqrt(2 (u, u) (v, v) / samples)``. It
    grows with the metric, and judging each entry against that scale is what
    lets one ``rtol`` serve a metric of any size.

    A space that draws standard normal *components* fails this check whenever
    its basis is not orthonormal: their covariance is ``G``, not the identity.

    Args:
        space: the space to check.
        rng: the generator. Seed it: this is a statistical check and an
            unseeded failure cannot be reproduced.
        measure: on a space without coordinates, draw the test directions
            ``u`` from this measure rather than from ``space.random``. On a
            coordinate space the directions are basis vectors and this is
            unused. The noise itself always comes from ``white_noise``.
        samples: how many draws to average over.
        rtol: the agreement required, as a fraction of
            ``sqrt((u, u) (v, v))``. It must scale like ``1 / sqrt(samples)``
            -- tightening one without the other only makes the check flaky --
            and the default is some six standard errors at the default
            *samples*.

    Returns:
        The report, carrying the worst covariance-entry deviation as a
        fraction of its scale -- the same quantity ``rtol`` bounds.

    Raises:
        AssertionError: if the sample covariance is not the identity on the
            space to within *rtol*.
    """
    rng = resolve_rng(rng)
    defects: dict[str, float] = {}
    if space.dim == 0:
        return CheckReport("check_white_noise", defects)

    if isinstance(space, CoordinateSpace):
        directions = [space.basis_vector(i) for i in range(min(space.dim, 3))]
    else:
        directions = [_draw(space, rng, measure) for _ in range(min(space.dim, 3))]

    projections = np.empty((samples, len(directions)))
    for n in range(samples):
        x = space.white_noise(rng=rng)
        for k, u in enumerate(directions):
            projections[n, k] = space.inner_product(x, u)

    empirical = projections.T @ projections / samples
    sizes = [space.inner_product(u, u) for u in directions]
    for i, u in enumerate(directions):
        for j, v in enumerate(directions):
            expected = space.inner_product(u, v)
            scale = np.sqrt(max(sizes[i], 0.0) * max(sizes[j], 0.0))
            _record(
                defects,
                "white noise has identity covariance",
                abs(empirical[i, j] - expected) / max(scale, 1e-12),
            )
            allowance = rtol * max(scale, 1e-12)
            if abs(empirical[i, j] - expected) > allowance:
                _fail(
                    "white noise has identity covariance",
                    f"E[(x,u_{i})(x,u_{j})] == {empirical[i, j]:g} but "
                    f"(u_{i}, u_{j}) == {expected:g}, beyond {allowance:g}; a "
                    f"factor of the Gram matrix here means components were "
                    f"drawn as N(0, I) instead of N(0, G^-1)",
                )
    return CheckReport("check_white_noise", defects)


# --------------------------------------------------------------------- #
#                               Operators                               #
# --------------------------------------------------------------------- #


def check_operator(
    operator: LinearOperator,
    /,
    *,
    rng: Generator | None = None,
    measure: ProbabilityMeasure | None = None,
    trials: int = 5,
    rtol: float = 1e-9,
    atol: float = 0.0,
) -> CheckReport:
    """Check linearity and the defining property of the adjoint.

    The adjoint identity ``(A x, y)_Y == (x, A* y)_X`` is the one that catches
    a mass matrix applied in the wrong place, which is the most common way for
    a hand-written adjoint to be subtly wrong on a non-orthonormal space.

    One ``rtol`` covers both identities, deliberately: an operator that is a
    solve is linear, and adjoint, only to its inner tolerance, so the two
    defects degrade together. A discrete adjoint should pass at ``1e-9``; a
    continuous one is checked with the ``rtol`` it can meet, and the report's
    defect is the number to watch fall under refinement.

    Args:
        operator: the operator to check.
        rng: the generator for the probe vectors.
        measure: a measure on the domain to draw the probe vectors from, in
            place of its ``random`` method; see the module docstring.
        trials: how many pairs to check the adjoint identity on.
        rtol: the agreement required, for linearity and the adjoint alike.
        atol: an absolute allowance on the linearity residual, for operators
            whose output is zero to rounding.

    Returns:
        The report, carrying the worst linearity and adjoint defects.

    Raises:
        AssertionError: naming the axiom that failed.
    """
    rng = resolve_rng(rng)
    domain, codomain = operator.domain, operator.codomain
    defects: dict[str, float] = {}

    for _ in range(trials):
        x, y = _draw(domain, rng, measure), _draw(domain, rng, measure)
        a, b = float(rng.normal()), float(rng.normal())

        combination = domain.add(domain.scale(a, x), domain.scale(b, y))
        expected = codomain.add(
            codomain.scale(a, operator(x)), codomain.scale(b, operator(y))
        )
        _assert_close(
            codomain,
            operator(combination),
            expected,
            "the operator is linear",
            rtol=rtol,
            atol=atol,
            record=defects,
        )

    adjoint = operator.adjoint
    if adjoint.domain != codomain or adjoint.codomain != domain:
        _fail(
            "the adjoint maps between the right spaces",
            f"{adjoint.domain!r} -> {adjoint.codomain!r}",
        )
    if operator.adjoint is not adjoint:
        _fail("the adjoint is memoized", "A.adjoint is not A.adjoint")
    if adjoint.adjoint is not operator:
        _fail("the adjoint is an involution", "A.adjoint.adjoint is not A")

    for _ in range(trials):
        x, y = _draw(domain, rng, measure), codomain.random(rng=rng)
        _assert_scalar(
            codomain.inner_product(operator(x), y),
            domain.inner_product(x, adjoint(y)),
            "the adjoint satisfies (A x, y) == (x, A* y)",
            rtol=rtol,
            record=defects,
        )
    return CheckReport("check_operator", defects)


def check_traits(
    operator: LinearOperator,
    /,
    *,
    rng: Generator | None = None,
    measure: ProbabilityMeasure | None = None,
    trials: int = 5,
    rtol: float = 1e-9,
) -> CheckReport:
    """Verify every trait the operator *claims*.

    Traits are assertions by whoever built the operator; nothing enforces them
    at construction. This is the safety net. Only claimed traits are checked —
    an absent trait is not a claim that the property fails.

    Args:
        operator: the operator whose claims are to be checked.
        rng: the generator for the probe vectors.
        measure: a measure on the domain to draw the probe vectors from, in
            place of its ``random`` method; see the module docstring.
        trials: how many vectors to test each claim on.
        rtol: the agreement required of each claimed identity.

    Returns:
        The report, carrying the worst defect of each claimed trait.

    Raises:
        AssertionError: naming the trait that is claimed but does not hold.
    """
    rng = resolve_rng(rng)
    defects: dict[str, float] = {}
    traits = operator.traits
    domain, codomain = operator.domain, operator.codomain

    def claims(member: Traits) -> bool:
        return traits & member == member

    if claims(Traits.SELF_ADJOINT) and domain != codomain:
        _fail(
            "a self-adjoint operator is an endomorphism", f"{domain!r} -> {codomain!r}"
        )

    for _ in range(trials):
        x, y = _draw(domain, rng, measure), _draw(domain, rng, measure)

        if claims(Traits.SELF_ADJOINT):
            _assert_scalar(
                codomain.inner_product(operator(x), y),
                domain.inner_product(x, operator(y)),
                "the claimed SELF_ADJOINT operator is symmetric",
                rtol=rtol,
                record=defects,
            )

        if claims(Traits.POSITIVE_SEMIDEFINITE):
            quadratic = domain.inner_product(operator(x), x)
            if quadratic < -rtol * max(domain.squared_norm(x), 1.0):
                _fail(
                    "the claimed POSITIVE_SEMIDEFINITE operator is non-negative",
                    f"(A x, x) == {quadratic:g}",
                )

        if claims(Traits.POSITIVE_DEFINITE):
            quadratic = domain.inner_product(operator(x), x)
            if quadratic <= 0.0:
                _fail(
                    "the claimed POSITIVE_DEFINITE operator is strictly positive",
                    f"(A x, x) == {quadratic:g} for a nonzero x",
                )

        if claims(Traits.ISOMETRY):
            _assert_scalar(
                codomain.squared_norm(operator(x)),
                domain.squared_norm(x),
                "the claimed ISOMETRY preserves the norm",
                rtol=rtol,
                record=defects,
            )

        if claims(Traits.IDEMPOTENT):
            image = operator(x)
            _assert_close(
                codomain,
                operator(image),
                image,
                "the claimed IDEMPOTENT operator satisfies A A == A",
                rtol=max(rtol, 1e-10),
                record=defects,
            )

    if claims(Traits.UNITARY) and domain.dim != codomain.dim:
        _fail("a claimed UNITARY operator is square", f"{domain.dim} != {codomain.dim}")

    if claims(Traits.INVERTIBLE):
        if domain.dim != codomain.dim:
            _fail(
                "a claimed INVERTIBLE operator is square",
                f"{domain.dim} != {codomain.dim}",
            )
        if isinstance(domain, CoordinateSpace) and isinstance(
            codomain, CoordinateSpace
        ):
            matrix = operator.matrix(form="components")
            if np.linalg.matrix_rank(matrix) < domain.dim:
                _fail(
                    "the claimed INVERTIBLE operator has full rank",
                    f"rank {np.linalg.matrix_rank(matrix)} of {domain.dim}",
                )
    return CheckReport("check_traits", defects)


def check_derivative(
    operator: Operator,
    point: object,
    /,
    *,
    rng: Generator | None = None,
    measure: ProbabilityMeasure | None = None,
    step: float = 1e-6,
    trials: int = 3,
    rtol: float = 1e-5,
) -> CheckReport:
    """Check the derivative against central finite differences.

    Args:
        operator: the operator to check.
        point: where to check it.
        rng: the generator for the probe directions.
        measure: a measure on the domain to draw the probe vectors from, in
            place of its ``random`` method; see the module docstring.
        step: the finite-difference step. Central differences, so the error
            is ``O(step^2)`` in truncation and ``O(1/step)`` in rounding --
            which is why the default is near the square root of machine
            epsilon rather than as small as possible.
        trials: how many directions to check.
        rtol: the agreement required.

    Returns:
        The report, carrying the worst finite-difference defect.

    Raises:
        AssertionError: if the operator carries no derivative, or the
            difference disagrees with it.
    """
    rng = resolve_rng(rng)
    defects: dict[str, float] = {}
    if not operator.has_derivative:
        _fail(
            "the operator carries a derivative", f"{type(operator).__name__} does not"
        )

    domain, codomain = operator.domain, operator.codomain
    linearization = operator.at(point)
    derivative = linearization.derivative

    for _ in range(trials):
        direction = _draw(domain, rng, measure)
        direction = domain.scale(1.0 / max(domain.norm(direction), 1e-30), direction)

        forward = operator(domain.axpy(step, direction, domain.copy(point)))
        backward = operator(domain.axpy(-step, direction, domain.copy(point)))
        numerical = codomain.scale(
            1.0 / (2.0 * step), codomain.subtract(forward, backward)
        )
        analytic = derivative(direction)

        residual = codomain.norm(codomain.subtract(numerical, analytic))
        scale = max(codomain.norm(numerical), codomain.norm(analytic), 1.0)
        _record(
            defects, "the derivative agrees with finite differences", residual / scale
        )
        if residual > rtol * scale:
            _fail(
                "the derivative agrees with finite differences",
                f"residual {residual:g} against scale {scale:g}",
            )
    return CheckReport("check_derivative", defects)


def check_gradient(
    functional: Functional,
    point: object,
    /,
    *,
    rng: Generator | None = None,
    measure: ProbabilityMeasure | None = None,
    step: float = 1e-6,
    trials: int = 3,
    rtol: float = 1e-5,
) -> CheckReport:
    """Check that the gradient really is the gradient, not the derivative.

    Compares ``(grad, d)`` against the central difference of the functional
    along ``d``. On a non-orthonormal space, supplying a derivative array where
    a gradient was wanted fails this by exactly a factor of the Gram matrix —
    which is the classic adjoint-method error. On an orthonormal space the two
    coincide and there is nothing to catch, which is why the error survives in
    practice.

    Args:
        functional: the functional to check.
        point: where to check it.
        rng: the generator for the probe directions.
        measure: a measure on the domain to draw the probe vectors from, in
            place of its ``random`` method; see the module docstring.
        step: the finite-difference step; see :func:`check_derivative`.
        trials: how many directions to check.
        rtol: the agreement required.

    Returns:
        The report, carrying the worst pairing defect.

    Raises:
        AssertionError: if the gradient is not the derivative's representer.
    """
    rng = resolve_rng(rng)
    defects: dict[str, float] = {}
    domain = functional.domain
    gradient = functional.at(point).gradient

    for _ in range(trials):
        direction = _draw(domain, rng, measure)
        direction = domain.scale(1.0 / max(domain.norm(direction), 1e-30), direction)

        forward = functional(domain.axpy(step, direction, domain.copy(point)))
        backward = functional(domain.axpy(-step, direction, domain.copy(point)))
        numerical = (forward - backward) / (2.0 * step)
        analytic = domain.inner_product(gradient, direction)

        scale = max(abs(numerical), abs(analytic), 1.0)
        _record(
            defects,
            "the gradient pairs with a direction as the derivative does",
            abs(numerical - analytic) / scale,
        )
        if abs(numerical - analytic) > rtol * scale:
            _fail(
                "the gradient pairs with a direction as the derivative does",
                f"(grad, d) == {analytic:g} but the finite difference is "
                f"{numerical:g}; a mismatch by a factor of the Gram matrix "
                f"means a derivative was supplied where a gradient was wanted",
            )
    return CheckReport("check_gradient", defects)


def check_second_derivative(
    operator: Operator,
    point: object,
    /,
    *,
    rng: Generator | None = None,
    measure: ProbabilityMeasure | None = None,
    step: float = 1e-5,
    trials: int = 3,
    rtol: float = 1e-4,
) -> CheckReport:
    """Check ``F''(x)[d, .]`` against finite differences of the derivative.

    Args:
        operator: the operator to check.
        point: where to check it.
        rng: the generator for the probe directions.
        measure: a measure on the domain to draw the probe vectors from, in
            place of its ``random`` method; see the module docstring.
        step: the finite-difference step; see :func:`check_derivative`.
        trials: how many directions to check.
        rtol: the agreement required. Looser than for a first derivative, the
            differencing being of an already-differenced quantity.

    Returns:
        The report, carrying the worst finite-difference defect.

    Raises:
        AssertionError: if the operator carries no second derivative, or it
            disagrees with the differences.
    """
    rng = resolve_rng(rng)
    defects: dict[str, float] = {}
    if not operator.has_second_derivative:
        _fail(
            "the operator carries a second derivative",
            f"{type(operator).__name__} does not",
        )

    domain, codomain = operator.domain, operator.codomain
    for _ in range(trials):
        d = _draw(domain, rng, measure)
        d = domain.scale(1.0 / max(domain.norm(d), 1e-30), d)
        e = _draw(domain, rng, measure)
        e = domain.scale(1.0 / max(domain.norm(e), 1e-30), e)

        forward = operator.derivative(domain.axpy(step, d, domain.copy(point)))(e)
        backward = operator.derivative(domain.axpy(-step, d, domain.copy(point)))(e)
        numerical = codomain.scale(
            1.0 / (2.0 * step), codomain.subtract(forward, backward)
        )
        analytic = operator.second_derivative(point, d)(e)

        residual = codomain.norm(codomain.subtract(numerical, analytic))
        scale = max(codomain.norm(numerical), codomain.norm(analytic), 1.0)
        _record(
            defects,
            "the second derivative agrees with finite differences",
            residual / scale,
        )
        if residual > rtol * scale:
            _fail(
                "the second derivative agrees with finite differences",
                f"residual {residual:g} against scale {scale:g}",
            )
    return CheckReport("check_second_derivative", defects)


# --------------------------------------------------------------------- #
#                               Measures                                #
# --------------------------------------------------------------------- #


def check_measure(
    measure: ProbabilityMeasure,
    /,
    *,
    rng: Generator | None = None,
    samples: int = 20000,
    rtol: float = 0.08,
    directions: int = 3,
) -> CheckReport:
    """Check that a measure's samples match the moments it declares.

    Compares the sample mean against ``expectation`` and the sample second
    moments against ``covariance``, along a few probe directions. A moment the
    measure does not claim is not checked: ``None`` is not a failure.

    The tolerances are statistical, not fixed fractions. The sample covariance
    estimator has standard error of order ``sqrt(C_ii C_jj / n)``, so an
    off-diagonal entry is judged against ``sqrt(C_ii C_jj)`` rather than
    against 1 — otherwise an entry that is exactly zero in expectation gets
    compared against an absolute floor that has nothing to do with the scale of
    the problem.

    Args:
        measure: the measure to check.
        rng: the generator. Seed it, as for :func:`check_white_noise`.
        samples: how many draws to take.
        rtol: the agreement required, scaling with the sample count.
        directions: how many probe directions to compare moments along.

    Returns:
        The report: the mean's worst deviation in standard errors, and the
        covariance's worst deviation as the fraction ``rtol`` bounds.

    Raises:
        AssertionError: if a declared moment disagrees with the samples.
    """
    rng = resolve_rng(rng)
    defects: dict[str, float] = {}
    space = measure.domain
    if space.dim == 0:
        return CheckReport("check_measure", defects)

    if isinstance(space, CoordinateSpace):
        probes = [space.basis_vector(i) for i in range(min(space.dim, directions))]
    else:
        probes = [space.random(rng=rng) for _ in range(min(space.dim, directions))]

    draws = measure.samples(samples, rng=rng)
    projections = np.array([[space.inner_product(x, u) for u in probes] for x in draws])
    spread = np.std(projections, axis=0)

    if measure.has_expectation:
        for i, u in enumerate(probes):
            observed = float(projections[:, i].mean())
            expected = space.inner_product(measure.expectation, u)
            # Six standard errors of the mean: a false alarm is far worse than
            # a missed one in a check that runs on every test invocation.
            standard_error = spread[i] / np.sqrt(samples) + 1e-300
            _record(
                defects,
                "the sample mean matches the declared expectation",
                abs(observed - expected) / standard_error,
            )
            allowance = 6.0 * spread[i] / np.sqrt(samples) + 1e-12
            if abs(observed - expected) > allowance:
                _fail(
                    "the sample mean matches the declared expectation",
                    f"E[(X, u_{i})] == {observed:g} but "
                    f"(m, u_{i}) == {expected:g}, beyond {allowance:g}",
                )

    if not measure.has_covariance:
        return CheckReport("check_measure", defects)

    covariance = measure.covariance
    centered = projections - projections.mean(axis=0, keepdims=True)
    empirical = centered.T @ centered / (samples - 1)
    diagonal = [space.inner_product(covariance(u), u) for u in probes]

    for i, u in enumerate(probes):
        for j, v in enumerate(probes):
            expected = space.inner_product(covariance(u), v)
            scale = np.sqrt(max(diagonal[i], 0.0) * max(diagonal[j], 0.0))
            _record(
                defects,
                "the sample covariance matches the declared covariance",
                abs(empirical[i, j] - expected) / max(scale, 1e-12),
            )
            allowance = rtol * max(scale, 1e-12)
            if abs(empirical[i, j] - expected) > allowance:
                _fail(
                    "the sample covariance matches the declared covariance",
                    f"Cov[(X,u_{i}), (X,u_{j})] == {empirical[i, j]:g} but "
                    f"(C u_{i}, u_{j}) == {expected:g}, beyond {allowance:g}",
                )
    return CheckReport("check_measure", defects)


def check_projection(
    subset: ConvexSet,
    /,
    *,
    rng: Generator | None = None,
    measure: ProbabilityMeasure | None = None,
    trials: int = 5,
    probes: int = 30,
    rtol: float = 1e-8,
) -> CheckReport:
    """Check that a convex set's projection really is the metric projection.

    Three properties, for each of a few probe points:

    - **lands in the set**: the projected point is a member;
    - **idempotent**: projecting the projected point leaves it where it is,
      which is what a proximal method relies on;
    - **nearest**: no nearby point of the set is closer to the probe.

    The third is checked by sampling rather than proved, which is the honest
    thing for a numerical check. It is what tells the metric projection apart
    from other maps onto the set, such as one onto its boundary.

    Args:
        subset: the convex set whose projection is to be checked.
        rng: the generator for the probe points.
        measure: a measure on the space to draw the probe vectors from, in
            place of the space's ``random`` method; see the module docstring.
        trials: how many points to project.
        probes: how many nearby points of the set to compare distances with,
            for each projection.
        rtol: the agreement required of the idempotency identity. The two
            membership tests have fixed tolerances of their own: a projected
            point counts as a member at ``contains(..., rtol=1e-6)``, which
            allows an iterative projection its own stopping tolerance, and a
            nearby point is compared only when it is a member at ``1e-9``,
            so that a point just outside the set cannot be mistaken for a
            closer one inside it.

    Returns:
        The report, carrying the worst idempotency defect.

    Raises:
        AssertionError: if the projection does not land in the set, is not
            idempotent, or a nearby point of the set is closer than it.
    """
    rng = resolve_rng(rng)
    defects: dict[str, float] = {}
    space = subset.domain

    for _ in range(trials):
        x = _draw(space, rng, measure)
        projected = subset.project(x)

        if not subset.contains(projected, rtol=1e-6):
            _fail("a projection lands in the set", f"{space.norm(projected):g}")

        _assert_close(
            space,
            subset.project(projected),
            projected,
            "the projection is idempotent",
            rtol=rtol,
            record=defects,
        )

        distance = space.norm(space.subtract(x, projected))
        for _ in range(probes):
            nearby = space.axpy(
                0.1 * distance / max(space.norm(x), 1.0),
                _draw(space, rng, measure),
                space.copy(projected),
            )
            if not subset.contains(nearby, rtol=1e-9):
                continue
            if space.norm(space.subtract(x, nearby)) < distance * (1.0 - 1e-9):
                _fail(
                    "the projection is the nearest point",
                    f"found a point of the set closer than {distance:g}",
                )
    return CheckReport("check_projection", defects)


def check_convexity(
    functional: Functional,
    /,
    *,
    rng: Generator | None = None,
    measure: ProbabilityMeasure | None = None,
    trials: int = 20,
    rtol: float = 1e-8,
) -> CheckReport:
    """Check that a functional is convex, by sampling.

    The inequality ``f(t x + (1 - t) y) <= t f(x) + (1 - t) f(y)`` on random
    pairs and random ``t``: a check, not a proof, as every sampled axiom
    here is. Its use is on a functional about to be handed to a method that
    assumes convexity, such as a bundle method, or made into a sublevel set
    that is meant to be convex. A value of ``+inf`` on the right leaves
    nothing to test and the pair is skipped.

    Args:
        functional: the functional to test.
        rng: the generator for the probe points.
        measure: a measure on the space to draw the probe vectors from, in
            place of the space's ``random`` method; see the module docstring.
        trials: how many pairs to test.
        rtol: the allowance, relative to the size of the right-hand side.

    Returns:
        The report, carrying the worst convexity violation, zero when no
        pair tried violates the inequality.

    Raises:
        AssertionError: if a pair violates the inequality.
    """
    rng = resolve_rng(rng)
    defects: dict[str, float] = {}
    space = functional.domain
    for _ in range(trials):
        x, y = _draw(space, rng, measure), _draw(space, rng, measure)
        t = float(rng.uniform())
        between = space.axpy(1.0 - t, y, space.scale(t, x))
        fx, fy = float(functional(x)), float(functional(y))
        right = t * fx + (1.0 - t) * fy
        if not np.isfinite(right):
            continue
        left = float(functional(between))
        scale = max(abs(t * fx) + abs((1.0 - t) * fy), 1e-300)
        _record(defects, "the functional is convex", max(left - right, 0.0) / scale)
        allowance = rtol * scale
        if left > right + allowance:
            _fail(
                "the functional is convex",
                f"f(t x + (1 - t) y) == {left:g} exceeds "
                f"t f(x) + (1 - t) f(y) == {right:g} at t == {t:.3f}",
            )
    return CheckReport("check_convexity", defects)


def check_affine(
    operator: Operator,
    /,
    *,
    rng: Generator | None = None,
    measure: ProbabilityMeasure | None = None,
    trials: int = 5,
    rtol: float = 1e-8,
) -> CheckReport:
    """Check that an affine operator is what it claims, ``x -> A x + b``.

    Three identities, then the linear part's own axioms. The translation is
    recovered by applying the operator to zero; affine combinations are
    preserved, ``F(a x + (1 - a) y) == a F(x) + (1 - a) F(y)`` for any
    ``a``, which is linearity with the translation canceling; and the
    derivative at a random point is the linear part, applied to a random
    direction. Then :func:`check_operator` on the linear part, which is
    where a mass matrix in the wrong place shows up as a failed adjoint
    identity.

    Args:
        operator: an :class:`~pygeoinf2.algebra.operators.AffineOperator`.
        rng: the generator for the probe vectors.
        measure: a measure on the domain to draw the probe vectors from, in
            place of its ``random`` method; see the module docstring.
        trials: how many random points and combinations to test.
        rtol: the allowance, relative to the size of what is compared.

    Returns:
        The report, carrying the affine identities' worst defects and the
        linear part's, from :func:`check_operator`, together.

    Raises:
        AssertionError: naming the identity that failed.
        TypeError: if the operator is not affine.
    """
    from .algebra.operators import AffineOperator

    if not isinstance(operator, AffineOperator):
        raise TypeError(
            f"check_affine needs an AffineOperator, got {type(operator).__name__}."
        )
    rng = resolve_rng(rng)
    defects: dict[str, float] = {}
    domain, codomain = operator.domain, operator.codomain
    linear = operator.linear_part

    _assert_close(
        codomain,
        operator(domain.zero()),
        operator.translation,
        "applying the operator to zero recovers the translation",
        rtol=rtol,
        record=defects,
    )
    for _ in range(trials):
        x, y = _draw(domain, rng, measure), _draw(domain, rng, measure)
        a = float(rng.normal())
        combination = domain.axpy(1.0 - a, y, domain.scale(a, x))
        expected = codomain.axpy(1.0 - a, operator(y), codomain.scale(a, operator(x)))
        _assert_close(
            codomain,
            operator(combination),
            expected,
            "the operator preserves affine combinations",
            rtol=rtol,
            record=defects,
        )
        direction = _draw(domain, rng, measure)
        _assert_close(
            codomain,
            operator.derivative(x)(direction),
            linear(direction),
            "the derivative is the linear part",
            rtol=rtol,
            record=defects,
        )
    nested = check_operator(linear, rng=rng, measure=measure, trials=trials, rtol=rtol)
    for axiom, defect in nested.defects.items():
        _record(defects, axiom, defect)
    return CheckReport("check_affine", defects)
