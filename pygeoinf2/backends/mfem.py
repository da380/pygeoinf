"""
MFEM finite element spaces.

This is the case the whole design was built for. In a finite element space the
inner product is not the dot product of the degree-of-freedom vector: it is

    ``(u, v) == u^T M v``

with ``M`` the mass matrix. So the mass matrix *is* the Gram matrix of
DESIGN.md 3.2, and three things that FEM practitioners write out by hand fall
out of the general machinery instead:

- **An assembled bilinear form is a Galerkin matrix.** ``a(u, v) == u^T K v``
  means ``K == M A_c``, which is precisely what
  :meth:`LinearOperator.from_derivative_matrix` expects. The adjoint then
  applies ``M^-1`` on its own.
- **An assembled linear form is a derivative, not a gradient.** The load vector
  has entries ``b_i == l(phi_i)``, so it is the functional's derivative
  components. Its Riesz representer is ``M^-1 b`` — the mass solve that turns a
  load vector back into a function.
- **A mass solve is a metric solve.** ``solve_gram`` is that solve, and it is
  the only place the inverse metric appears.

Vectors are ``mfem.Vector`` objects, not NumPy arrays. That is deliberate: it
exercises the claim that a vector can be any backend object, since arithmetic
goes through the space rather than through the vector.

Requires PyMFEM, which is an optional dependency; install it with the ``mfem``
extra.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Hashable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.random import Generator

from ..algebra.operators import LinearFunctional, LinearOperator
from ..algebra.spaces import CoordinateSpace, _resolve_rng
from ..traits import Traits

__all__ = [
    "MfemSpace",
    "operator_from_bilinear_form",
    "functional_from_linear_form",
]


def _require_mfem() -> Any:
    """Import PyMFEM's serial interface, with a message that says what to install."""
    try:
        import mfem.ser as mfem
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "MFEM spaces need PyMFEM, which is an optional dependency. "
            "Install it with the 'mfem' extra."
        ) from error
    return mfem


def _to_scipy(matrix: Any) -> sp.csr_matrix:
    """An MFEM sparse matrix as a SciPy one, sharing nothing."""
    size = matrix.Height()
    return sp.csr_matrix(
        (
            np.array(matrix.GetDataArray(), copy=True),
            np.array(matrix.GetJArray(), copy=True),
            np.array(matrix.GetIArray(), copy=True),
        ),
        shape=(size, matrix.Width()),
    )


class MfemSpace(CoordinateSpace):
    """A finite element space, with its mass matrix as the metric.

    Vectors are ``mfem.Vector`` objects holding degree-of-freedom values;
    components are those values as a NumPy array. The Gram matrix is the mass
    matrix, so the inner product is the ``L2`` inner product of the underlying
    functions rather than a dot product of coefficients — which is what makes a
    norm computed here mean the same thing under mesh refinement.
    """

    def __init__(self, finite_element_space: Any, /) -> None:
        """
        Args:
            finite_element_space: an ``mfem.FiniteElementSpace``.
        """
        mfem = _require_mfem()
        self._fes = finite_element_space
        self._dim = int(finite_element_space.GetTrueVSize())

        form = mfem.BilinearForm(finite_element_space)
        form.AddDomainIntegrator(mfem.MassIntegrator())
        form.Assemble()
        form.Finalize()
        self._mass_form = form
        self._mass = form.SpMat()

    # ----------------------------------------------------------------- #
    #                              Structure                            #
    # ----------------------------------------------------------------- #

    @property
    def dim(self) -> int:
        """The number of degrees of freedom."""
        return self._dim

    @property
    def finite_element_space(self) -> Any:
        """The wrapped ``mfem.FiniteElementSpace``."""
        return self._fes

    @property
    def mass_matrix(self) -> Any:
        """The assembled mass matrix, which is this space's Gram matrix."""
        return self._mass

    def _key(self) -> Hashable:
        # Two spaces built on the same FiniteElementSpace object are the same
        # space; two built on separately meshed but identical geometries are
        # not, because MFEM gives no cheap way to tell that they agree.
        return id(self._fes)

    def __repr__(self) -> str:
        return f"MfemSpace(dofs={self._dim})"

    # ----------------------------------------------------------------- #
    #                          Vector operations                        #
    # ----------------------------------------------------------------- #

    def zero(self) -> Any:
        """A new zero ``mfem.Vector``."""
        mfem = _require_mfem()
        vector = mfem.Vector(self._dim)
        vector.Assign(0.0)
        return vector

    def copy(self, x: Any) -> Any:
        """An independent copy of an ``mfem.Vector``."""
        mfem = _require_mfem()
        return mfem.Vector(x)

    def inner_product(self, x: Any, y: Any) -> float:
        """``x^T M y``, the ``L2`` inner product of the underlying functions."""
        return float(self._mass.InnerProduct(x, y))

    def axpy(self, a: float, x: Any, y: Any) -> Any:
        """``y += a x``, in place, using MFEM's own operation."""
        y.Add(float(a), x)
        return y

    def scale_inplace(self, a: float, x: Any) -> Any:
        """``x *= a``, in place."""
        x *= float(a)
        return x

    def random(self, *, rng: Generator | None = None) -> Any:
        """A vector with independent standard normal degree-of-freedom values."""
        return self.from_components(_resolve_rng(rng).standard_normal(self._dim))

    # ----------------------------------------------------------------- #
    #                             Coordinates                           #
    # ----------------------------------------------------------------- #

    def to_components(self, x: Any) -> np.ndarray:
        """The degree-of-freedom values, as an array.

        A **copy**, deliberately. ``GetDataArray`` hands back a view into
        memory MFEM owns, and that view does not keep its owner alive: in an
        expression like ``to_components(from_components(c))`` the temporary
        vector is collected and the view is left pointing at freed memory. The
        symptom is silent -- plausible numbers that are simply wrong -- so the
        copy is not negotiable, and the cost of one array copy per call is the
        price of a foreign backend owning its own memory.
        """
        return np.array(x.GetDataArray(), copy=True)

    def from_components(self, c: np.ndarray) -> Any:
        """An ``mfem.Vector`` with the given degree-of-freedom values."""
        mfem = _require_mfem()
        values = np.asarray(c, dtype=float)
        if values.shape != (self._dim,):
            raise ValueError(f"Expected {self._dim} components, got {values.shape}.")
        vector = mfem.Vector(self._dim)
        vector.GetDataArray()[:] = values
        return vector

    @cached_property
    def _scipy_mass(self) -> sp.csr_matrix:
        """The mass matrix in SciPy form, for factorisation."""
        return _to_scipy(self._mass)

    @cached_property
    def _mass_factorisation(self) -> Any:
        """A sparse factorisation of the mass matrix, computed once.

        The mass matrix is well conditioned and its factors stay sparse, which
        is why a direct solve is the right choice here and an iterative one
        would be a false economy.
        """
        return spla.factorized(self._scipy_mass.tocsc())

    def apply_gram(self, c: np.ndarray) -> np.ndarray:
        """``M c``."""
        return self._scipy_mass @ c

    def solve_gram(self, c: np.ndarray) -> np.ndarray:
        """``M^-1 c``: the mass solve that turns a load vector into a function."""
        return self._mass_factorisation(np.asarray(c, dtype=float))

    def white_noise_components(self, *, rng: Generator | None = None) -> np.ndarray:
        """Components drawn from ``N(0, M^-1)``.

        Which is what makes the covariance the identity on the space rather
        than the mass matrix — the correction that is easy to omit and hard to
        notice, since it is invisible whenever the metric is trivial.
        """
        factor = np.linalg.cholesky(self._scipy_mass.toarray())
        noise = _resolve_rng(rng).standard_normal(self._dim)
        return spla.spsolve_triangular(sp.csr_matrix(factor.T), noise, lower=False)


def operator_from_bilinear_form(
    space: MfemSpace,
    form: Any,
    /,
    *,
    traits: Traits = Traits.NONE,
) -> LinearOperator:
    """The operator a bilinear form defines, as an operator on the space.

    An assembled bilinear form is the **Galerkin matrix** of the operator it
    represents: ``a(u, v) == u^T K v`` and ``(A u, v) == (A u)^T M v`` together
    give ``K == M A_c``. So the assembled matrix goes straight into
    :meth:`LinearOperator.from_derivative_matrix`, and the mass solve that
    turns it into an action on functions happens inside the operator rather
    than in the caller's code.

    Args:
        space: the finite element space.
        form: an assembled and finalised ``mfem.BilinearForm``.
        traits: claims about the operator. A symmetric integrator gives a
            self-adjoint operator, but nothing here can check the integrator,
            so the claim is the caller's and ``testing.check_traits`` verifies
            it.
    """
    return LinearOperator.from_derivative_matrix(
        space, space, _to_scipy(form.SpMat()).toarray(), traits=traits
    )


def functional_from_linear_form(space: MfemSpace, form: Any, /) -> LinearFunctional:
    """The functional a linear form defines.

    An assembled linear form is a **load vector**: its entries are ``l(phi_i)``,
    the functional's derivative components. So this is
    ``from_derivative_components``, and ``.representer`` is ``M^-1 b`` — the
    mass solve that recovers the function representing the functional.

    Handing the load vector to an optimiser as if it were a gradient is the
    error of DESIGN.md 5.6, in the setting where it is most often made.
    """
    return LinearFunctional.from_derivative_components(
        space, np.asarray(form.GetDataArray(), dtype=float)
    )
