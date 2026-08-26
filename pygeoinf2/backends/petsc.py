"""
PETSc vectors and matrices.

The point of this backend is the one thing the test doubles cannot settle:
whether the coordinate-free core really works when vectors are opaque,
distributed objects owned by somebody else. A ``PETSc.Vec`` is not a NumPy
array, cannot be indexed freely under MPI, and has its own memory model — and
none of that matters here, because arithmetic goes through the space.

Two spaces are offered, and the difference between them is the whole subject of
DESIGN.md 5.6:

- :class:`PetscSpace` is ``R^n`` with the Euclidean inner product. An
  operator's adjoint is then its transpose, and a derivative and a gradient
  coincide.
- :class:`PetscWeightedSpace` carries a mass matrix. The adjoint is
  ``M^-1 A^T M``, and the two do **not** coincide. This is the finite element
  situation, and using ``multTranspose`` as though it were the adjoint is the
  mistake the library exists to make hard.

Requires petsc4py, an optional dependency; install it with the ``petsc`` extra.
Note that PETSc has no binary wheel and builds from source, so installing it is
a substantial operation rather than a quick one.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Hashable

import numpy as np
from numpy.random import Generator

from ..algebra.operators import LinearOperator
from ..algebra.spaces import CoordinateSpace, _resolve_rng
from ..traits import Traits

__all__ = ["PetscSpace", "PetscWeightedSpace", "operator_from_matrix"]


def _require_petsc() -> Any:
    """Import petsc4py, with a message that says what to install."""
    try:
        from petsc4py import PETSc
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "PETSc spaces need petsc4py, which is an optional dependency. "
            "Install it with the 'petsc' extra. PETSc has no wheel, so this "
            "builds from source."
        ) from error
    return PETSc


class PetscSpace(CoordinateSpace):
    """``R^n`` over PETSc vectors, with the Euclidean inner product.

    Vectors are ``PETSc.Vec`` objects. Every operation the core needs — a dot
    product, an ``axpy``, a scaling, a copy — is one PETSc call, which is the
    whole adapter.
    """

    def __init__(self, dim: int, /, *, comm: Any = None) -> None:
        """
        Args:
            dim: the global vector length.
            comm: an MPI communicator. Defaults to ``COMM_SELF``, which is the
                right choice for a single-process example.
        """
        petsc = _require_petsc()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self._dim = int(dim)
        self._comm = comm if comm is not None else petsc.COMM_SELF

    @property
    def dim(self) -> int:
        """The global vector length."""
        return self._dim

    @property
    def comm(self) -> Any:
        """The MPI communicator these vectors live on."""
        return self._comm

    def _key(self) -> Hashable:
        return (self._dim, id(self._comm))

    def __repr__(self) -> str:
        return f"PetscSpace(dim={self._dim})"

    # ----------------------------------------------------------------- #
    #                          Vector operations                        #
    # ----------------------------------------------------------------- #

    def zero(self) -> Any:
        """A new zeroed ``PETSc.Vec``."""
        petsc = _require_petsc()
        vector = petsc.Vec().createSeq(self._dim, comm=self._comm)
        vector.set(0.0)
        return vector

    def copy(self, x: Any) -> Any:
        """An independent copy, using PETSc's own duplication."""
        return x.copy()

    def inner_product(self, x: Any, y: Any) -> float:
        """``x . y``, which under MPI is a collective reduction."""
        return float(x.dot(y))

    def axpy(self, a: float, x: Any, y: Any) -> Any:
        """``y += a x``, in place."""
        y.axpy(float(a), x)
        return y

    def scale_inplace(self, a: float, x: Any) -> Any:
        """``x *= a``, in place."""
        x.scale(float(a))
        return x

    def random(self, *, rng: Generator | None = None) -> Any:
        """A vector with independent standard normal entries."""
        return self.from_components(_resolve_rng(rng).standard_normal(self._dim))

    # ----------------------------------------------------------------- #
    #                             Coordinates                           #
    # ----------------------------------------------------------------- #

    def to_components(self, x: Any) -> np.ndarray:
        """The entries, as an array.

        A copy, for the same reason as the MFEM backend: an array that views
        memory the backend owns can outlive its owner, and the result is wrong
        numbers rather than an error.
        """
        return np.array(x.getArray(readonly=True), copy=True)

    def from_components(self, c: np.ndarray) -> Any:
        """A ``PETSc.Vec`` with the given entries."""
        values = np.asarray(c, dtype=float)
        if values.shape != (self._dim,):
            raise ValueError(f"Expected {self._dim} components, got {values.shape}.")
        vector = self.zero()
        vector.getArray()[:] = values
        return vector


class PetscWeightedSpace(PetscSpace):
    """``R^n`` with the inner product a mass matrix defines.

    ``(x, y) == x^T M y``, so the Gram matrix is ``M`` and an operator's
    adjoint is ``M^-1 A^T M`` rather than ``A^T``. That is the finite element
    situation expressed in PETSc terms, and the reason
    :func:`operator_from_matrix` asks which convention a matrix is in.
    """

    def __init__(self, mass: Any, /, *, comm: Any = None) -> None:
        """
        Args:
            mass: a square, symmetric, positive-definite ``PETSc.Mat``.
            comm: an MPI communicator.
        """
        rows, columns = mass.getSize()
        if rows != columns:
            raise ValueError(f"The mass matrix must be square, got {rows}x{columns}.")
        super().__init__(rows, comm=comm)
        self._mass = mass

    @property
    def mass(self) -> Any:
        """The mass matrix, which is this space's Gram matrix."""
        return self._mass

    def _key(self) -> Hashable:
        return (self._dim, id(self._mass))

    def __repr__(self) -> str:
        return f"PetscWeightedSpace(dim={self._dim})"

    def inner_product(self, x: Any, y: Any) -> float:
        """``x^T M y``."""
        scratch = self.zero()
        self._mass.mult(y, scratch)
        return float(x.dot(scratch))

    def apply_gram(self, c: np.ndarray) -> np.ndarray:
        """``M c``."""
        return self.to_components(self._apply_mass(self.from_components(c)))

    def _apply_mass(self, x: Any) -> Any:
        result = self.zero()
        self._mass.mult(x, result)
        return result

    @cached_property
    def _mass_solver(self) -> Any:
        """A factorisation of the mass matrix, built once.

        A direct solve, because a mass matrix is well conditioned and its
        factors stay sparse. This is the only place the inverse metric appears.
        """
        petsc = _require_petsc()
        solver = petsc.KSP().create(comm=self._comm)
        solver.setOperators(self._mass)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setFromOptions()
        return solver

    def solve_gram(self, c: np.ndarray) -> np.ndarray:
        """``M^-1 c``."""
        right = self.from_components(c)
        result = self.zero()
        self._mass_solver.solve(right, result)
        return self.to_components(result)


def operator_from_matrix(
    space: PetscSpace,
    matrix: Any,
    /,
    *,
    form: str = "components",
    traits: Traits = Traits.NONE,
) -> LinearOperator:
    """Wrap a ``PETSc.Mat`` as an operator on a space.

    ``form`` says which representation the matrix is in, because no trait
    implies it:

    - ``"components"``: ``A x`` is ``matrix.mult(x)``. On a weighted space the
      adjoint is then ``M^-1 A^T M``, **not** ``multTranspose``.
    - ``"galerkin"``: the matrix is ``M A_c``, which is what an assembled
      bilinear form gives. Its transpose is then the adjoint's Galerkin matrix,
      and the mass solve happens inside the operator.

    On an unweighted :class:`PetscSpace` the two coincide, which is exactly why
    the distinction is easy to miss until the mass matrix is not the identity.
    """
    if form not in ("components", "galerkin"):
        raise ValueError(f"Unknown form {form!r}.")

    dense = _matrix_to_array(matrix)
    if form == "components":
        return LinearOperator.from_component_matrix(space, space, dense, traits=traits)
    return LinearOperator.from_derivative_matrix(space, space, dense, traits=traits)


def _matrix_to_array(matrix: Any) -> np.ndarray:
    """A dense array of a PETSc matrix, for the small problems examples use."""
    rows, columns = matrix.getSize()
    return np.array(
        [
            [matrix.getValue(row, column) for column in range(columns)]
            for row in range(rows)
        ]
    )
