"""
Numerical checks on the axioms the library assumes but cannot enforce.

These live here rather than in the production classes. In v1 the equivalents
are mixins in the MRO of every space and operator
(``class HilbertSpace(ABC, HilbertSpaceAxiomChecks)``), which puts a test suite
inside the objects under test.

Every check raises ``AssertionError`` with a message naming the axiom that
failed, and returns ``None`` on success.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.random import Generator, default_rng

from .algebra.spaces import CoordinateSpace, HilbertSpace

__all__ = [
    "check_space",
    "check_coordinates",
    "check_white_noise",
    "check_representer",
]


def _fail(axiom: str, detail: str) -> None:
    raise AssertionError(f"Axiom failed: {axiom}. {detail}")


def _assert_close(
    space: HilbertSpace,
    a,
    b,
    axiom: str,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    """Assert two vectors agree, measured in the space's own norm."""
    residual = space.norm(space.subtract(a, b))
    scale = max(space.norm(a), space.norm(b), 1.0)
    if residual > atol + rtol * scale:
        _fail(axiom, f"residual {residual:g} against scale {scale:g}")


def _assert_scalar(
    value: float, expected: float, axiom: str, *, rtol: float = 1e-10
) -> None:
    scale = max(abs(value), abs(expected), 1.0)
    if abs(value - expected) > rtol * scale:
        _fail(axiom, f"{value!r} != {expected!r}")


def check_space(
    space: HilbertSpace,
    /,
    *,
    rng: Generator | None = None,
    trials: int = 5,
    rebuild: Callable[[], HilbertSpace] | None = None,
) -> None:
    """Check the vector space and inner product axioms.

    Args:
        space: the space to check.
        rng: generator used to draw test vectors.
        trials: how many random triples to test.
        rebuild: optional zero-argument callable returning an independently
            constructed space that should compare equal to ``space``. Supply it
            to catch identity-based equality, which silently produces spurious
            "domain mismatch" errors later. See DESIGN.md section 9.
    """
    rng = default_rng() if rng is None else rng

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
        return

    for _ in range(trials):
        x, y, z = space.random(rng), space.random(rng), space.random(rng)
        a, b = float(rng.normal()), float(rng.normal())

        # --- vector space axioms -----------------------------------------
        _assert_close(space, space.add(x, y), space.add(y, x), "addition commutes")
        _assert_close(
            space,
            space.add(space.add(x, y), z),
            space.add(x, space.add(y, z)),
            "addition associates",
        )
        _assert_close(
            space, space.add(x, space.zero()), x, "zero is the additive identity"
        )
        _assert_close(
            space,
            space.add(x, space.negative(x)),
            space.zero(),
            "negation inverts addition",
        )
        _assert_close(space, space.scale(1.0, x), x, "scaling by one is the identity")
        _assert_close(
            space,
            space.scale(a, space.scale(b, x)),
            space.scale(a * b, x),
            "scalar multiplication associates",
        )
        _assert_close(
            space,
            space.scale(a, space.add(x, y)),
            space.add(space.scale(a, x), space.scale(a, y)),
            "scaling distributes over addition",
        )
        _assert_close(
            space,
            space.scale(a + b, x),
            space.add(space.scale(a, x), space.scale(b, x)),
            "scaling distributes over scalar addition",
        )

        # --- inner product axioms ----------------------------------------
        _assert_scalar(
            space.inner_product(x, y),
            space.inner_product(y, x),
            "the inner product is symmetric",
        )
        _assert_scalar(
            space.inner_product(space.add(x, y), z),
            space.inner_product(x, z) + space.inner_product(y, z),
            "the inner product is additive",
        )
        _assert_scalar(
            space.inner_product(space.scale(a, x), y),
            a * space.inner_product(x, y),
            "the inner product is homogeneous",
        )
        if space.squared_norm(x) <= 0.0:
            _fail(
                "the inner product is positive definite",
                f"||x||^2 == {space.squared_norm(x)}",
            )
        _assert_scalar(
            space.norm(x) ** 2,
            space.squared_norm(x),
            "the norm is the root of the squared norm",
        )

        # --- in-place operations agree with their out-of-place forms ------
        expected = space.add(x, space.scale(a, y))
        target = space.copy(x)
        result = space.axpy(a, y, target)
        _assert_close(space, result, expected, "axpy agrees with add and scale")

        expected = space.scale(a, x)
        target = space.copy(x)
        result = space.scale_inplace(a, target)
        _assert_close(space, result, expected, "scale_inplace agrees with scale")

        # --- copy is independent ------------------------------------------
        original = space.copy(x)
        duplicate = space.copy(x)
        space.scale_inplace(2.0, duplicate)
        _assert_close(space, x, original, "copy does not alias its source")

    # --- derived helpers ---------------------------------------------------
    vectors = [space.random(rng) for _ in range(min(3, space.dim))]
    orthonormal = space.gram_schmidt(vectors)
    for i, u in enumerate(orthonormal):
        for j, v in enumerate(orthonormal):
            _assert_scalar(
                space.inner_product(u, v),
                1.0 if i == j else 0.0,
                "gram_schmidt returns an orthonormal set",
            )

    samples = [space.random(rng) for _ in range(4)]
    total = space.zero()
    for s in samples:
        total = space.axpy(0.25, s, total)
    _assert_close(
        space, space.mean(samples), total, "mean is the average of its arguments"
    )


def check_coordinates(
    space: CoordinateSpace,
    /,
    *,
    rng: Generator | None = None,
    trials: int = 5,
) -> None:
    """Check the coordinate map, the Gram matrix, and the pairing axiom."""
    rng = default_rng() if rng is None else rng

    if not isinstance(space, CoordinateSpace):
        _fail(
            "the space provides coordinates",
            f"{type(space).__name__} is not a CoordinateSpace",
        )
    if space.dim == 0:
        return

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
        x, y = space.random(rng), space.random(rng)
        cx = space.to_components(x)
        cy = space.to_components(y)
        a = float(rng.normal())

        if cx.shape != (space.dim,):
            _fail("components have shape (dim,)", f"got {cx.shape} for dim {space.dim}")

        _assert_close(
            space, space.from_components(cx), x, "the coordinate map round-trips"
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
        )
        if not np.allclose(space.apply_gram(cx), gram @ cx):
            _fail("apply_gram agrees with the Gram matrix", "they differ")
        if not np.allclose(
            space.solve_gram(space.apply_gram(cx)), cx, rtol=1e-8, atol=1e-10
        ):
            _fail("solve_gram inverts apply_gram", "the round trip does not return c")

        check_representer(space, cx, rng=rng)

    for i in range(min(space.dim, 4)):
        e = space.basis_vector(i)
        expected = np.zeros(space.dim)
        expected[i] = 1.0
        if not np.allclose(space.to_components(e), expected):
            _fail("basis_vector has the expected components", f"index {i}")


def check_representer(
    space: CoordinateSpace,
    derivative_components: np.ndarray,
    /,
    *,
    rng: Generator | None = None,
    trials: int = 3,
) -> None:
    """Check the pairing axiom that separates a derivative from a gradient.

    For a functional given by ``g`` in the derivative convention — the array a
    numerical adjoint method returns, acting as ``x -> g . c_x`` — its Riesz
    representer ``v`` must satisfy ``(v, x) == g . c_x`` for every ``x``.

    Handing ``g`` straight to an optimiser as if it were the gradient fails this
    identity by a factor of the Gram matrix, which is exactly the classic
    adjoint-method error. On an orthonormal basis the two coincide and there is
    nothing to catch. See DESIGN.md section 5.6.
    """
    rng = default_rng() if rng is None else rng
    g = np.asarray(derivative_components, dtype=float)
    v = space.representer(g)

    for _ in range(trials):
        x = space.random(rng)
        _assert_scalar(
            space.inner_product(v, x),
            float(g @ space.to_components(x)),
            "the representer pairs with vectors as the derivative does",
        )


def check_white_noise(
    space: HilbertSpace,
    /,
    *,
    rng: Generator | None = None,
    samples: int = 20000,
    rtol: float = 0.06,
) -> None:
    """Check that ``white_noise`` really has identity covariance on the space.

    Tests ``E[(x, u) (x, v)] == (u, v)`` over pairs of basis-like directions.
    This is a statistical check, so it needs a seeded generator and a tolerance
    that scales like ``1 / sqrt(samples)``.

    v1 fails this check on every mass-weighted space: drawing standard normal
    *components* gives covariance ``G`` rather than the identity. See DESIGN.md
    section 9.
    """
    rng = default_rng() if rng is None else rng
    if space.dim == 0:
        return

    if isinstance(space, CoordinateSpace):
        directions = [space.basis_vector(i) for i in range(min(space.dim, 3))]
    else:
        directions = [space.random(rng) for _ in range(min(space.dim, 3))]

    projections = np.empty((samples, len(directions)))
    for n in range(samples):
        x = space.white_noise(rng)
        for k, u in enumerate(directions):
            projections[n, k] = space.inner_product(x, u)

    empirical = projections.T @ projections / samples
    for i, u in enumerate(directions):
        for j, v in enumerate(directions):
            expected = space.inner_product(u, v)
            scale = max(abs(expected), 1.0)
            if abs(empirical[i, j] - expected) > rtol * scale:
                _fail(
                    "white noise has identity covariance",
                    f"E[(x,u_{i})(x,u_{j})] == {empirical[i, j]:g} but "
                    f"(u_{i}, u_{j}) == {expected:g}; a factor of the Gram "
                    f"matrix here means components were drawn as N(0, I) "
                    f"instead of N(0, G^-1)",
                )
