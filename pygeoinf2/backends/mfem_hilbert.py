"""
MFEM as a plain Hilbert space: this library conducts, MFEM computes.

:mod:`pygeoinf2.backends.mfem` presents a finite element space as a
:class:`~pygeoinf2.algebra.spaces.CoordinateSpace`. It reads the mass matrix
out of MFEM into SciPy, factorises it there, and hands assembled forms over as
Galerkin matrices — so once a form is assembled, this library does the linear
algebra. That is the right arrangement for a modest serial problem, where a
sparse factorisation of the mass matrix is cheap and a dense Gram matrix is
available for checking things.

This module is the other arrangement. The space is a
:class:`~pygeoinf2.algebra.spaces.HilbertSpace` and nothing more: it has no
component map, no Gram matrix, and never reads a CSR array. Every operation
goes through MFEM's own ``Operator`` interface —

- the inner product is ``(M x, y)`` with ``M`` applied by MFEM's ``Mult``;
- the mass solve is MFEM's conjugate gradients on MFEM's operator;
- a bilinear form becomes an operator through ``FormSystemMatrix``, which is
  how MFEM itself imposes essential conditions, and which may just as well
  hand back a matrix-free (partially assembled) operator as a sparse matrix;
- a linear form is a true-dof vector and its representer is a mass solve;
- white noise is MFEM's own integrator, followed by that mass solve.

Nothing in it asks whether a vector is local or distributed, and nothing in
it forms a matrix. That is the point: the same code over a
``ParFiniteElementSpace`` with ``HypreParMatrix`` operators is the MPI version,
with this library orchestrating solves it never sees. The two things a
parallel space must supply are named in :class:`MfemDofSpace` — a reduced
dot product and a global dimension — and nothing else changes. That path is
*not exercised here*: the test environment has no ``mpi4py``, so the claim is
that the design admits it, not that it has been run.

What this arrangement costs is honesty about adjoints. An operator's adjoint
on a mass-weighted space is ``M^-1 A^T M``; without a factorisation every
``M^-1`` is an iterative solve, well conditioned and short, but a solve. The
coordinate backend pays that once at construction; this one pays it per
adjoint. For a PDE-constrained problem the PDE solves dominate either way.

The construction is the core's own :class:`MassWeightedSpace` — the vectors
belong to :class:`MfemDofSpace`, a bare Euclidean space over true-dof vectors,
and the finite element inner product is that space reweighted by the mass
operator. DESIGN.md section 3.5 describes exactly this chain; a finite element
space is its most natural instance.

Requires PyMFEM, which is an optional dependency; install it with the ``mfem``
extra.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Hashable, Literal

import numpy as np
from numpy.random import Generator

from ..algebra.operators import LinearFunctional, LinearOperator
from ..algebra.spaces import (
    EuclideanSpace,
    HilbertSpace,
    MassWeightedSpace,
    _resolve_rng,
)
from ..traits import Traits
from .mfem import (
    _as_indices,
    _matern_parameters,
    _require_mfem,
    _white_noise_integrator,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..probability.gaussian import GaussianMeasure

__all__ = [
    "MfemDofSpace",
    "MfemHilbertSpace",
    "operator_from_bilinear_form",
    "solver_from_bilinear_form",
    "functional_from_linear_form",
    "operator_from_linear_forms",
    "white_noise_load",
    "matern_measure",
]

Assembly = Literal["full", "partial"]
_DEFINITE = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE


# --------------------------------------------------------------------- #
#                           The degree-of-freedom space                  #
# --------------------------------------------------------------------- #


class MfemDofSpace(HilbertSpace):
    """True-dof vectors under the plain dot product: the space the mass
    operator reweights.

    Vectors are ``mfem.Vector`` objects of the full true-dof length, and on a
    constrained space they are zero on the essential degrees of freedom — so
    a vector here can be handed to a ``GridFunction`` as it is. The dimension
    is the number of *free* degrees of freedom, which is the dimension of the
    subspace the vectors actually range over.

    This space has no finite element meaning on its own: its inner product is
    the dot product of coefficients, which is the thing DESIGN.md section 5.6
    warns against mistaking for an inner product of functions. It exists to
    be reweighted by :class:`MfemHilbertSpace`, and to be the base from which
    :meth:`LinearOperator.from_formal_adjoint` lifts an operator whose
    transpose is known.

    **For a parallel space.** Two things are rank-local here and must be
    supplied for a ``ParFiniteElementSpace``: *dot*, which must reduce over
    the communicator (``mfem.InnerProduct(comm, x, y)`` in ``mfem.par``), and
    *dimension*, the global count of free degrees of freedom. Everything else
    — allocation, ``Add``, scaling, filling the local entries — is local
    already. Untested: see the module docstring.
    """

    def __init__(
        self,
        finite_element_space: Any,
        /,
        *,
        essential_dofs: Any = None,
        dot: Callable[[Any, Any], float] | None = None,
        dimension: int | None = None,
    ) -> None:
        """
        Args:
            finite_element_space: an ``mfem.FiniteElementSpace``.
            essential_dofs: true degrees of freedom held at zero, as an
                ``mfem.intArray`` or any sequence of indices. See
                :func:`pygeoinf2.backends.mfem.essential_dofs_of`.
            dot: the dot product of two vectors. MFEM's serial
                ``InnerProduct`` if omitted; a parallel space passes the
                reducing one.
            dimension: the number of free degrees of freedom. Counted locally
                if omitted, which is right in serial; a parallel space passes
                the global count.

        Raises:
            ValueError: for an essential index out of range, or if every
                degree of freedom is constrained.
        """
        mfem = _require_mfem()
        self._fes = finite_element_space
        self._size = int(finite_element_space.GetTrueVSize())

        constrained = (
            np.empty(0, dtype=int)
            if essential_dofs is None
            else np.unique(np.asarray(_as_indices(essential_dofs), dtype=int))
        )
        if constrained.size and (
            constrained.min() < 0 or constrained.max() >= self._size
        ):
            raise ValueError(
                f"Essential degrees of freedom must lie in [0, {self._size}); "
                f"got indices from {constrained.min()} to {constrained.max()}."
            )
        self._essential = constrained
        self._essential_array = mfem.intArray(constrained.tolist())
        self._dim = (
            int(self._size - constrained.size) if dimension is None else int(dimension)
        )
        if self._dim <= 0:
            raise ValueError(
                "Every degree of freedom is constrained, leaving a space of "
                "dimension zero."
            )
        self._dot = mfem.InnerProduct if dot is None else dot

    # ----------------------------------------------------------------- #
    #                              Structure                            #
    # ----------------------------------------------------------------- #

    @property
    def dim(self) -> int:
        """The number of free degrees of freedom."""
        return self._dim

    @property
    def size(self) -> int:
        """The length of a vector: every true degree of freedom, free or not."""
        return self._size

    @property
    def finite_element_space(self) -> Any:
        """The wrapped ``mfem.FiniteElementSpace``."""
        return self._fes

    @property
    def essential_dofs(self) -> np.ndarray:
        """The true degrees of freedom held at zero. Empty when unconstrained."""
        return self._essential.copy()

    @property
    def essential_dof_array(self) -> Any:
        """The same list as the ``mfem.intArray`` MFEM's own routines take."""
        return self._essential_array

    @property
    def is_constrained(self) -> bool:
        """Whether an essential boundary condition has been imposed."""
        return self._essential.size > 0

    def _key(self) -> Hashable:
        # As in the coordinate backend: the same FiniteElementSpace object and
        # the same constraint make the same space.
        return (id(self._fes), self._essential.tobytes())

    def __repr__(self) -> str:
        return f"MfemDofSpace(dofs={self._dim})"

    # ----------------------------------------------------------------- #
    #                          Vector operations                        #
    # ----------------------------------------------------------------- #

    def zero(self) -> Any:
        """A new zero ``mfem.Vector`` of full true-dof length."""
        mfem = _require_mfem()
        vector = mfem.Vector(self._size)
        vector.Assign(0.0)
        return vector

    def copy(self, x: Any) -> Any:
        """An independent copy."""
        mfem = _require_mfem()
        return mfem.Vector(x)

    def inner_product(self, x: Any, y: Any) -> float:
        """The dot product of the degree-of-freedom values."""
        return float(self._dot(x, y))

    def axpy(self, a: float, x: Any, y: Any) -> Any:
        """``y += a x``, in place, by MFEM's own operation."""
        y.Add(float(a), x)
        return y

    def scale_inplace(self, a: float, x: Any) -> Any:
        """``x *= a``, in place."""
        x *= float(a)
        return x

    def constrain(self, x: Any, /) -> Any:
        """Zero the essential entries of ``x`` in place, and return it.

        The projection onto the subspace. Applied to anything that enters from
        outside — a random fill, an assembled load — so that every vector of
        the space vanishes where the boundary condition says it must.
        """
        if self.is_constrained:
            x.SetSubVector(self._essential_array, 0.0)
        return x

    def random(self, *, rng: Generator | None = None) -> Any:
        """Independent standard normal values on the free degrees of freedom."""
        vector = self.zero()
        vector.GetDataArray()[:] = _resolve_rng(rng).standard_normal(self._size)
        return self.constrain(vector)

    def white_noise(self, *, rng: Generator | None = None) -> Any:
        """White noise *on this space*, whose inner product is the dot product:
        the same draw as :meth:`random`. Not white on the finite element space;
        :meth:`MfemHilbertSpace.white_noise` is.
        """
        return self.random(rng=rng)


# --------------------------------------------------------------------- #
#                              MFEM systems                              #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class _System:
    """A bilinear form taken through ``FormSystemMatrix``, kept whole.

    Three MFEM objects that must live and die together. The handle refers to a
    matrix the *form* owns (full assembly eliminates rows and columns in
    place) or to a ``ConstrainedOperator`` that refers back to the form
    (partial assembly); drop either while the operator is in use and the next
    ``Mult`` reads freed memory, which segfaults rather than raising. Every
    closure below holds the whole record.
    """

    form: Any
    handle: Any
    operator: Any
    space: "MfemHilbertSpace"

    def apply(self, x: Any) -> Any:
        """``K x``, into a new vector."""
        out = self.space.dof_space.zero()
        self.operator.Mult(x, out)
        return out

    def apply_transpose(self, x: Any) -> Any:
        """``K^T x``, into a new vector.

        Available on the two operators ``FormSystemMatrix`` produces — a
        ``SparseMatrix`` and a ``ConstrainedOperator`` — which is why the
        adjoint of a non-symmetric form costs no more than the form.
        """
        out = self.space.dof_space.zero()
        self.operator.MultTranspose(x, out)
        return out

    @property
    def is_sparse(self) -> bool:
        """Whether the handle holds an assembled ``SparseMatrix``."""
        mfem = _require_mfem()
        return self.handle.Type() == mfem.Operator.MFEM_SPARSEMAT


def _system_of(space: "MfemHilbertSpace", form: Any) -> _System:
    """Impose the space's constraint on an assembled form, MFEM's way.

    ``FormSystemMatrix`` is what every MFEM example calls: with full assembly
    it eliminates the essential rows and columns of the form's matrix and
    puts ones on their diagonal; with partial assembly it wraps the form's
    action in a ``ConstrainedOperator`` that does the same thing without a
    matrix. Either way the result, restricted to vectors that vanish on the
    essential degrees of freedom, is the operator of the constrained problem.

    The form is *consumed*: its matrix is eliminated in place, and reading
    ``SpMat()`` from it afterwards is not meaningful. Hand each form to one
    space.
    """
    mfem = _require_mfem()
    handle = mfem.OperatorPtr()
    form.FormSystemMatrix(space.dof_space.essential_dof_array, handle)
    return _System(form, handle, handle.Ptr(), space)


def _mfem_solver(
    system: _System,
    /,
    *,
    make_solver: Callable[[Any], Any] | None,
    rtol: float,
    max_iterations: int,
) -> Any:
    """An MFEM solver for a system, with a preconditioner MFEM can build.

    Gauss-Seidel on a sparse matrix, and the operator's own Jacobi smoother
    when there is no matrix — the latter being the one that works without
    ever assembling, which is what a partially assembled form is for.
    """
    mfem = _require_mfem()
    if make_solver is not None:
        solver = make_solver(system.operator)
        solver._pygeoinf_keepalive = (system,)
        return solver
    if system.is_sparse:
        smoother = mfem.GSSmoother(mfem.OperatorHandle2SparseMatrix(system.handle))
    else:
        smoother = mfem.OperatorJacobiSmoother(
            system.form, system.space.dof_space.essential_dof_array
        )
    solver = mfem.CGSolver()
    solver.SetOperator(system.operator)
    solver.SetPreconditioner(smoother)
    solver.SetRelTol(float(rtol))
    solver.SetMaxIter(int(max_iterations))
    solver.SetPrintLevel(-1)
    # MFEM's solver holds raw pointers to its operator and preconditioner and
    # owns neither; see _System for what happens otherwise.
    solver._pygeoinf_keepalive = (system, smoother)
    return solver


def _solve_with(solver: Any, space: MfemDofSpace, b: Any, /, *, strict: bool) -> Any:
    """Run an MFEM solver from zero, and say so if it did not converge."""
    solution = space.zero()
    solver.Mult(b, solution)
    if strict and hasattr(solver, "GetConverged") and not solver.GetConverged():
        raise RuntimeError(
            f"The MFEM solver did not converge in {solver.GetNumIterations()} "
            f"iterations. Loosen rtol, raise max_iterations, or supply a "
            f"better preconditioner through make_solver."
        )
    return solution


# --------------------------------------------------------------------- #
#                        The finite element space                        #
# --------------------------------------------------------------------- #


class MfemHilbertSpace(MassWeightedSpace):
    """A finite element space with the mass operator as its metric, and no
    coordinates.

    The core's :class:`MassWeightedSpace` over :class:`MfemDofSpace`: the
    inner product is ``(M x, y)`` with ``M`` the mass operator MFEM assembled
    — fully, or partially so that no matrix exists at all — and ``M^-1`` is
    MFEM's conjugate gradients on it. The library sees ``Mult`` and a solve,
    and nothing else.

    What follows from having no coordinates: :meth:`LinearOperator.matrix`
    refuses, so nothing dense can be formed by accident; every solver that
    runs here is a Krylov one; and the randomised and functional-calculus
    routines take their coordinate-free paths. What does not follow is any
    loss of meaning — the norm is the ``L2`` norm of the function, the
    adjoint is the adjoint, and the Bayesian machinery runs unchanged.
    """

    def __init__(
        self,
        finite_element_space: Any,
        /,
        *,
        essential_dofs: Any = None,
        assembly: Assembly = "full",
        make_solver: Callable[[Any], Any] | None = None,
        rtol: float = 1e-14,
        max_iterations: int = 500,
        dot: Callable[[Any, Any], float] | None = None,
        dimension: int | None = None,
    ) -> None:
        """
        Args:
            finite_element_space: an ``mfem.FiniteElementSpace``.
            essential_dofs: true degrees of freedom held at zero, as for
                :class:`MfemDofSpace`.
            assembly: how MFEM assembles the mass form. ``"full"`` gives a
                sparse matrix; ``"partial"`` gives a matrix-free operator,
                and on a tensor-product mesh is the arrangement that scales.
            make_solver: ``operator -> solver`` for the mass solve, with
                ``SetOperator`` already called. MFEM's conjugate gradients
                with a preconditioner it can build if omitted. Keep a
                reference to anything the solver does not own; MFEM will not.
            rtol: the mass solve's relative residual. Tight by default: the
                mass operator is well conditioned and every adjoint goes
                through this solve, so it should not be where error enters.
            max_iterations: the cap on that solve.
            dot: see :class:`MfemDofSpace`; for a parallel space.
            dimension: likewise.

        Raises:
            ValueError: for an unknown assembly level, or the constraint
                errors :class:`MfemDofSpace` raises.
        """
        mfem = _require_mfem()
        if assembly not in ("full", "partial"):
            raise ValueError(f"assembly must be 'full' or 'partial', got {assembly!r}.")
        dof_space = MfemDofSpace(
            finite_element_space,
            essential_dofs=essential_dofs,
            dot=dot,
            dimension=dimension,
        )
        self._assembly: Assembly = assembly
        # MassWeightedSpace needs the mass operator on the base; _system_of
        # needs self.dof_space. Set the base first, then let the parent set it
        # again, identically.
        self._base = dof_space

        form = mfem.BilinearForm(finite_element_space)
        if assembly == "partial":
            form.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
        form.AddDomainIntegrator(mfem.MassIntegrator())
        form.Assemble()
        system = _system_of(self, form)
        solver = _mfem_solver(
            system, make_solver=make_solver, rtol=rtol, max_iterations=max_iterations
        )

        def mass_apply(x: Any, _system: _System = system) -> Any:
            return _system.apply(x)

        def mass_solve(b: Any, _solver: Any = solver) -> Any:
            return _solve_with(_solver, dof_space, b, strict=True)

        mass = LinearOperator.self_adjoint(
            dof_space, mass_apply, traits=Traits.POSITIVE_DEFINITE
        )
        inverse = LinearOperator.self_adjoint(
            dof_space, mass_solve, traits=Traits.POSITIVE_DEFINITE
        )
        super().__init__(dof_space, mass, mass_solver=inverse)
        self._mass_system = system

    # ----------------------------------------------------------------- #
    #                              Structure                            #
    # ----------------------------------------------------------------- #

    @property
    def dof_space(self) -> MfemDofSpace:
        """The unweighted space the vectors belong to."""
        return self._base

    @property
    def finite_element_space(self) -> Any:
        """The wrapped ``mfem.FiniteElementSpace``."""
        return self._base.finite_element_space

    @property
    def essential_dofs(self) -> np.ndarray:
        """The true degrees of freedom held at zero. Empty when unconstrained."""
        return self._base.essential_dofs

    @property
    def is_constrained(self) -> bool:
        """Whether an essential boundary condition has been imposed."""
        return self._base.is_constrained

    @property
    def assembly(self) -> Assembly:
        """How the mass form was assembled."""
        return self._assembly

    def _key(self) -> Hashable:
        # The parent keys on the mass operator, whose identity is not
        # structural. Here the mass is determined by the element space, the
        # constraint and the assembly, so two constructions over the same
        # FiniteElementSpace are the same space -- as in the coordinate backend.
        return (self._base._key(), self._assembly)

    def __repr__(self) -> str:
        return f"MfemHilbertSpace(dofs={self.dim}, assembly={self._assembly!r})"

    def new_system(self, form: Any, /) -> _System:
        """Take an assembled bilinear form through ``FormSystemMatrix`` here.

        What :func:`operator_from_bilinear_form` and
        :func:`solver_from_bilinear_form` do first; exposed so that both can
        be built from one system, as :func:`matern_measure` does.

        Args:
            form: an assembled ``mfem.BilinearForm``. It is consumed; see
                the note on :func:`operator_from_bilinear_form`.

        Returns:
            The system, an opaque record the two constructors accept in place
            of a form.
        """
        return _system_of(self, form)

    # ----------------------------------------------------------------- #
    #                             Randomness                            #
    # ----------------------------------------------------------------- #

    def white_noise(self, *, rng: Generator | None = None) -> Any:
        """A draw with identity covariance on the finite element space.

        MFEM's white-noise integrator gives a load with covariance ``M``; the
        mass solve turns that into a vector with covariance ``M^-1``, which is
        the identity in this inner product. No factorisation of ``M`` is
        involved anywhere, and no matrix is formed.
        """
        return self.mass_inverse(white_noise_load(self, rng=rng))


# --------------------------------------------------------------------- #
#                               Operators                                #
# --------------------------------------------------------------------- #


def _as_system(space: MfemHilbertSpace, form_or_system: Any) -> _System:
    if isinstance(form_or_system, _System):
        if form_or_system.space is not space:
            raise ValueError("This system was built on a different space.")
        return form_or_system
    return space.new_system(form_or_system)


def operator_from_bilinear_form(
    space: MfemHilbertSpace,
    form: Any,
    /,
    *,
    traits: Traits = Traits.NONE,
) -> LinearOperator:
    """The operator a bilinear form defines, applied by MFEM.

    ``a(u, v) == (K u, v)`` with ``K`` the system MFEM assembles, and
    ``(A u, v)_V == (M A u, v)``, so ``A == M^-1 K``: apply the form, then a
    mass solve. The adjoint is ``M^-1 K^T``, with the transpose MFEM's
    ``MultTranspose`` — so a non-symmetric form gets its true adjoint, which is
    *not* ``K^T`` and not ``K`` either.

    **The form is consumed.** ``FormSystemMatrix`` eliminates the essential
    rows and columns of the form's matrix in place, and MFEM's convention is
    that a form built for one system is that system's. Assemble a fresh form
    for each space, and do not read ``SpMat()`` from it afterwards. To make
    both the operator and its inverse from one form, take
    :meth:`MfemHilbertSpace.new_system` once and pass the result to both
    constructors.

    Args:
        space: the finite element space, carrying any essential conditions.
        form: an assembled ``mfem.BilinearForm``, or a system from
            :meth:`MfemHilbertSpace.new_system`.
        traits: claims about the operator. A symmetric integrator gives a
            self-adjoint operator; nothing here can check the integrator, so
            the claim is the caller's and ``testing.check_traits`` verifies it.

    Returns:
        The operator, as a ``LinearOperator`` on *space*.
    """
    system = _as_system(space, form)

    def value(x: Any, _system: _System = system) -> Any:
        return space.mass_inverse(_system.apply(x))

    def adjoint(y: Any, _system: _System = system) -> Any:
        return space.mass_inverse(_system.apply_transpose(y))

    return LinearOperator.from_callables(
        space,
        space,
        value,
        adjoint=value if Traits.SELF_ADJOINT & traits else adjoint,
        traits=traits,
    )


def solver_from_bilinear_form(
    space: MfemHilbertSpace,
    form: Any,
    /,
    *,
    make_solver: Callable[[Any], Any] | None = None,
    rtol: float = 1e-12,
    max_iterations: int = 1000,
    traits: Traits = _DEFINITE,
    strict: bool = True,
) -> LinearOperator:
    """The *inverse* of the operator a bilinear form defines, solved by MFEM.

    The operator is ``M^-1 K``, so its inverse is ``K^-1 M``: a mass
    application turns the function into a load vector, MFEM solves, and the
    solution is the answer. The mass application is the step that is easy to
    leave out — the answer then comes back smooth, plausible and wrong by a
    mass matrix, DESIGN.md section 5.6 in its most convincing disguise.

    Args:
        space: the finite element space, carrying any essential conditions.
        form: an assembled ``mfem.BilinearForm``, or a system from
            :meth:`MfemHilbertSpace.new_system`. Consumed, as for
            :func:`operator_from_bilinear_form`.
        make_solver: ``operator -> solver``, a configured MFEM solver with
            ``SetOperator`` already called. Conjugate gradients with Gauss-
            Seidel (sparse) or the operator's Jacobi smoother (matrix-free)
            if omitted. This is where a multigrid cycle or a parallel
            preconditioner goes, and nothing else changes.
        rtol: the relative residual for the default solver.
        max_iterations: its cap.
        traits: claims about the inverse. A symmetric form's inverse is
            self-adjoint, which is the usual case and the default; a
            non-symmetric one has no adjoint here, because ``M^-1 K^-T M``
            would need a transposed solve MFEM does not offer generically.
        strict: raise when the solver reports it did not converge.

    Returns:
        The inverse, as a ``LinearOperator`` on *space*.
    """
    system = _as_system(space, form)
    solver = _mfem_solver(
        system, make_solver=make_solver, rtol=rtol, max_iterations=max_iterations
    )

    def solve(x: Any, _solver: Any = solver) -> Any:
        return _solve_with(_solver, space.dof_space, space.mass(x), strict=strict)

    return LinearOperator.from_callables(
        space,
        space,
        solve,
        adjoint=solve if Traits.SELF_ADJOINT & traits else None,
        traits=traits,
    )


def _true_dof_vector(space: MfemHilbertSpace, form: Any) -> Any:
    """An assembled linear form as a true-dof vector of the space.

    A copy, with the essential entries zeroed: the load on a constrained
    degree of freedom is not part of the constrained functional. The one
    parallel-specific line in this module is the ``ParallelAssemble`` branch,
    which a ``ParLinearForm`` has and a ``LinearForm`` does not; untested.
    """
    mfem = _require_mfem()
    if hasattr(form, "ParallelAssemble"):  # pragma: no cover - needs mfem.par
        vector = mfem.Vector(form.ParallelAssemble())
    else:
        vector = mfem.Vector(form)
    return space.dof_space.constrain(vector)


def functional_from_linear_form(
    space: MfemHilbertSpace, form: Any, /
) -> LinearFunctional:
    """The functional a linear form defines.

    Its action is the dot product with the load vector, ``l(u) == b . u``; its
    representer — the gradient — is ``M^-1 b``, a mass solve that is not paid
    until asked for. The load vector is the *derivative* and the representer
    is the *gradient*, and they differ by the metric.

    Args:
        space: the finite element space.
        form: an assembled ``mfem.LinearForm``.

    Returns:
        The functional.
    """
    load = _true_dof_vector(space, form)
    dof_space = space.dof_space

    def value(x: Any, _load: Any = load) -> float:
        return dof_space.inner_product(_load, x)

    def representer(_load: Any = load) -> Any:
        return space.mass_inverse(_load)

    return LinearFunctional.from_callables(space, value, representer=representer)


def operator_from_linear_forms(
    space: MfemHilbertSpace,
    forms: Sequence[Any],
    /,
    *,
    codomain: HilbertSpace | None = None,
) -> LinearOperator:
    """Several linear forms stacked into one observation operator.

    Each row is a load vector, so the action is a list of dot products and
    costs nothing but those. The adjoint sends data ``d`` to
    ``M^-1 sum_i d_i b_i``: one combination of load vectors and one mass
    solve, however many sensors there are. Treating the load vectors as the
    sensors' kernels would drop that solve and return an adjoint off by a
    mass matrix.

    Args:
        space: the finite element space, carrying any essential conditions.
        forms: assembled ``mfem.LinearForm`` objects, one per observation.
        codomain: the data space. A Euclidean space of the right size if
            omitted, which is what a list of numbers is.

    Returns:
        The observation operator.

    Raises:
        ValueError: if no forms are given, or the codomain's dimension does
            not match how many there are.
    """
    loads = [_true_dof_vector(space, form) for form in forms]
    if not loads:
        raise ValueError("At least one linear form is needed.")
    if codomain is None:
        codomain = EuclideanSpace(len(loads))
    elif codomain.dim != len(loads):
        raise ValueError(
            f"{len(loads)} linear forms for a data space of dimension "
            f"{codomain.dim}."
        )
    dof_space = space.dof_space

    def value(x: Any, _loads: list[Any] = loads) -> np.ndarray:
        return np.array([dof_space.inner_product(load, x) for load in _loads])

    def adjoint(d: Any, _loads: list[Any] = loads) -> Any:
        combination = dof_space.zero()
        for weight, load in zip(np.asarray(d, dtype=float), _loads):
            combination = dof_space.axpy(float(weight), load, combination)
        return space.mass_inverse(combination)

    return LinearOperator.from_callables(space, codomain, value, adjoint=adjoint)


# --------------------------------------------------------------------- #
#                                Measures                                #
# --------------------------------------------------------------------- #


def white_noise_load(
    space: MfemHilbertSpace, /, *, rng: Generator | None = None
) -> Any:
    """A load vector whose covariance is the mass operator.

    The finite element discretisation of white noise, ``(W, phi_i)``, whose
    covariance is ``(phi_i, phi_j) == M``. MFEM assembles it element by
    element; nothing is factorised. The vector vanishes on the essential
    degrees of freedom, where its covariance is the constrained mass
    operator's — the free block of ``M``.

    Args:
        space: the finite element space.
        rng: the generator. Only a seed reaches MFEM, drawn from this, so a
            run is reproducible from a NumPy generator like everything else.

    Returns:
        An ``mfem.Vector``.
    """
    mfem = _require_mfem()
    seed = int(_resolve_rng(rng).integers(1, 2**31 - 1))
    # Bound to a name, not passed inline; see the coordinate backend for why.
    integrator = _white_noise_integrator(seed)
    form = mfem.LinearForm(space.finite_element_space)
    form.AddDomainIntegrator(integrator)
    form.Assemble()
    vector = _true_dof_vector(space, form)
    del integrator
    return vector


def matern_measure(
    space: MfemHilbertSpace,
    /,
    *,
    smoothness: float = 1.0,
    correlation_length: Any = 0.1,
    rotation: float = 0.0,
    amplitude: float = 1.0,
    solver: Callable[[Any], Any] | None = None,
    rtol: float = 1e-10,
    max_iterations: int = 1000,
) -> "GaussianMeasure":
    """A Matern random field by the SPDE method, with every solve MFEM's.

    The same construction as :func:`pygeoinf2.backends.mfem.matern_measure`
    — ``(I - div Theta grad)^a u == eta W`` — on a space with no coordinates:
    the operator is a form MFEM applies, its inverse is MFEM's solver on the
    same system, and the white noise is MFEM's integrator. The measure gets
    its covariance ``eta^2 A^-2a``, its factor ``eta S^a`` and its precision
    ``A^2a / eta^2`` as compositions of those, and nothing is assembled by
    this library at all.

    Args:
        space: the finite element space, carrying any essential conditions.
        smoothness: the Matern ``nu``; ``(nu + d/2) / 2`` must be a positive
            integer, as for the coordinate backend.
        correlation_length: one length, or one per dimension.
        rotation: the anisotropy's rotation angle, two dimensions only.
        amplitude: a scale on the field.
        solver: ``operator -> solver``, passed to
            :func:`solver_from_bilinear_form` as ``make_solver``.
        rtol: the relative residual of each elliptic solve.
        max_iterations: their cap.

    Returns:
        The measure.

    Raises:
        ValueError: for a non-integer or non-positive exponent, a
            non-positive correlation length, or an anisotropic field on a
            partially assembled space (see the note in the source).
    """
    mfem = _require_mfem()
    from ..probability.gaussian import GaussianMeasure

    dimension = space.finite_element_space.GetMesh().Dimension()
    theta, normalisation, order = _matern_parameters(
        dimension, smoothness, correlation_length, rotation
    )

    # An isotropic field has Theta == c I, and a scalar coefficient is both
    # cheaper and the only kind partial assembly handles: in PyMFEM 4.8 a
    # partially assembled DiffusionIntegrator with a MatrixConstantCoefficient
    # returns values of order 1e290, whatever keeps the coefficient alive
    # (measured; scalar coefficients agree with full assembly to 1e-16).
    isotropic = np.allclose(theta, theta[0, 0] * np.identity(dimension))
    if isotropic:
        coefficient = mfem.ConstantCoefficient(float(theta[0, 0]))
    elif space.assembly == "partial":
        raise ValueError(
            "An anisotropic field needs a matrix coefficient, and partial "
            "assembly of one produces garbage in this PyMFEM build. Build the "
            "space with assembly='full' for an anisotropic Matern field."
        )
    else:
        coefficient = mfem.MatrixConstantCoefficient(theta)

    form = mfem.BilinearForm(space.finite_element_space)
    if space.assembly == "partial":
        form.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
    form.AddDomainIntegrator(mfem.DiffusionIntegrator(coefficient))
    form.AddDomainIntegrator(mfem.MassIntegrator())
    form.Assemble()
    system = space.new_system(form)

    operator = operator_from_bilinear_form(space, system, traits=_DEFINITE)
    solve = solver_from_bilinear_form(
        space, system, make_solver=solver, rtol=rtol, max_iterations=max_iterations
    )

    scale = float(amplitude) * normalisation
    factor = solve
    powered = operator
    for _ in range(order - 1):
        factor = factor @ solve
        powered = powered @ operator
    covariance = (scale**2) * (factor @ factor)
    return GaussianMeasure(
        space,
        covariance=covariance.with_traits(_DEFINITE),
        covariance_factor=scale * factor,
        precision=((1.0 / scale**2) * (powered @ powered)).with_traits(_DEFINITE),
    )
