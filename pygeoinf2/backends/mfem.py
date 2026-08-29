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
  :meth:`LinearOperator.from_matrix` in Galerkin form expects. The adjoint then
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
from typing import TYPE_CHECKING, Any, Hashable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.random import Generator

from ..algebra.operators import LinearFunctional, LinearOperator
from ..algebra.spaces import CoordinateSpace, EuclideanSpace, _resolve_rng
from ..traits import Traits

if TYPE_CHECKING:  # pragma: no cover
    from ..probability.gaussian import GaussianMeasure

__all__ = [
    "MfemSpace",
    "essential_dofs_of",
    "operator_from_bilinear_form",
    "operator_from_linear_forms",
    "solver_from_bilinear_form",
    "functional_from_linear_form",
    "white_noise_load",
    "matern_measure",
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
    """An MFEM sparse matrix as a SciPy one, sharing nothing.

    Refuses an unfinalised matrix. Before ``Finalize`` an MFEM sparse matrix is
    held as linked lists and has no CSR arrays at all, so asking for them does
    not return something wrong — it returns pointers into nothing, and reading
    them segfaults the interpreter with no traceback. A check costs one call
    and turns the worst failure mode available into a sentence.
    """
    if hasattr(matrix, "Finalized") and not matrix.Finalized():
        raise ValueError(
            "This MFEM matrix has not been finalised, so it has no CSR arrays "
            "to read. Call Finalize() on the form after Assemble(). (Reading "
            "them anyway segfaults rather than raising, which is why this is "
            "checked.)"
        )
    size = matrix.Height()
    return sp.csr_matrix(
        (
            np.array(matrix.GetDataArray(), copy=True),
            np.array(matrix.GetJArray(), copy=True),
            np.array(matrix.GetIArray(), copy=True),
        ),
        shape=(size, matrix.Width()),
    )


def _as_indices(dofs: Any) -> Any:
    """Index values from an ``mfem.intArray`` or any ordinary sequence."""
    return dofs.ToList() if hasattr(dofs, "ToList") else dofs


def essential_dofs_of(
    finite_element_space: Any,
    /,
    *,
    attributes: Any = None,
) -> np.ndarray:
    """The true degrees of freedom on an essential boundary.

    A homogeneous Dirichlet condition says a function vanishes on part of the
    boundary, and in a finite element space that is a statement about which
    degrees of freedom are held at zero. This asks MFEM which they are, so that
    :class:`MfemSpace` can be built on the subspace where they already are.

    Args:
        finite_element_space: an ``mfem.FiniteElementSpace``.
        attributes: the boundary attributes to constrain. All of them if
            omitted, which is the usual case; give a sequence of attribute
            numbers for a mixed problem where only part of the boundary is
            essential.

    Returns:
        The true-dof indices, sorted.

    Raises:
        ValueError: if an attribute number is not among the mesh's boundary
            attributes -- almost always a mesh that does not have the
            boundary the caller thinks it has.
    """
    mfem = _require_mfem()
    mesh = finite_element_space.GetMesh()
    count = mesh.bdr_attributes.Max()
    marker = mfem.intArray(count)
    if attributes is None:
        marker.Assign(1)
    else:
        marker.Assign(0)
        for attribute in attributes:
            if not 1 <= int(attribute) <= count:
                raise ValueError(
                    f"Boundary attribute {attribute} is out of range; this "
                    f"mesh has attributes 1 to {count}."
                )
            marker[int(attribute) - 1] = 1
    dofs = mfem.intArray()
    finite_element_space.GetEssentialTrueDofs(marker, dofs)
    return np.unique(np.asarray(dofs.ToList(), dtype=int))


class MfemSpace(CoordinateSpace):
    """A finite element space, with its mass matrix as the metric.

    Vectors are ``mfem.Vector`` objects holding degree-of-freedom values;
    components are those values as a NumPy array. The Gram matrix is the mass
    matrix, so the inner product is the ``L2`` inner product of the underlying
    functions rather than a dot product of coefficients — which is what makes a
    norm computed here mean the same thing under mesh refinement.
    """

    def __init__(
        self,
        finite_element_space: Any,
        /,
        *,
        essential_dofs: Any = None,
    ) -> None:
        """
        Args:
            finite_element_space: an ``mfem.FiniteElementSpace``.
            essential_dofs: true degrees of freedom held at zero — the ones an
                essential (Dirichlet) boundary condition fixes. Given, the
                space becomes the *subspace* of functions vanishing there, of
                dimension ``GetTrueVSize() - len(essential_dofs)``, and every
                matrix taken from a form is restricted to the free block. See
                :func:`essential_dofs_of`, which produces the list from a set
                of boundary attributes.
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
        self._free = np.setdiff1d(np.arange(self._size), constrained)
        self._dim = int(self._free.size)
        if self._dim == 0:
            raise ValueError(
                "Every degree of freedom is constrained, leaving a space of "
                "dimension zero."
            )

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
        """The assembled mass matrix over *all* degrees of freedom.

        The space's Gram matrix is its free block; see :meth:`restrict`.
        """
        return self._mass

    @property
    def essential_dofs(self) -> np.ndarray:
        """The true degrees of freedom held at zero. Empty when unconstrained."""
        return self._essential.copy()

    @property
    def free_dofs(self) -> np.ndarray:
        """The true degrees of freedom this space actually varies."""
        return self._free.copy()

    @property
    def is_constrained(self) -> bool:
        """Whether an essential boundary condition has been imposed."""
        return self._essential.size > 0

    def restrict(self, matrix: Any, /) -> Any:
        """The free-free block of a matrix assembled over all degrees of freedom.

        A bilinear form assembled on the full space is the Galerkin matrix of
        an operator on the full space. Restricted to the functions vanishing on
        the essential boundary, it is the Galerkin matrix of the operator on
        *that* space — which is what the constrained problem is, and why
        eliminating rows and columns is the whole of what a homogeneous
        Dirichlet condition does.

        **Sparsity survives.** A finite element matrix is sparse because the
        basis functions have local support, and that is the only reason a large
        problem fits in memory at all: at 1e5 degrees of freedom a dense block
        is 80 GB. This used to call ``.toarray()`` unconditionally, so
        :func:`operator_from_bilinear_form` handed a dense array to
        ``from_matrix`` and :func:`solver_from_bilinear_form`
        densified it only to re-sparsify it immediately.

        Args:
            matrix: an ``mfem.SparseMatrix`` or a NumPy array over all dofs.

        Returns:
            The free block, sparse when the input was sparse.
        """
        if isinstance(matrix, np.ndarray):
            return (
                matrix[np.ix_(self._free, self._free)]
                if self.is_constrained
                else matrix
            )
        sparse = _to_scipy(matrix)
        if not self.is_constrained:
            return sparse
        # Row slice then column slice, as _scipy_mass does: CSR does both
        # without ever forming the dense block.
        return sp.csr_matrix(sparse[self._free][:, self._free])

    def restrict_vector(self, values: Any, /) -> np.ndarray:
        """The free entries of a vector assembled over all degrees of freedom.

        A copy, for the reason :meth:`to_components` gives at length: an array
        handed over by MFEM is a view into memory MFEM owns and does not keep
        its owner alive. Fancy indexing copies anyway, so only the
        unconstrained path needed saying — which is exactly the path where the
        bug is invisible until the owning form goes out of scope.
        """
        array = np.array(values, dtype=float, copy=True)
        return array if not self.is_constrained else array[self._free]

    def _key(self) -> Hashable:
        # Two spaces built on the same FiniteElementSpace object are the same
        # space; two built on separately meshed but identical geometries are
        # not, because MFEM gives no cheap way to tell that they agree.
        return (id(self._fes), self._essential.tobytes())

    def __repr__(self) -> str:
        return f"MfemSpace(dofs={self._dim})"

    # ----------------------------------------------------------------- #
    #                          Vector operations                        #
    # ----------------------------------------------------------------- #

    def zero(self) -> Any:
        """A new zero ``mfem.Vector``, of full degree-of-freedom length.

        Vectors stay the size MFEM expects even when the space is constrained,
        and simply hold zero on the essential degrees of freedom. So a vector
        of this space can be handed straight to a ``GridFunction`` and drawn,
        which a vector of free values alone could not.
        """
        mfem = _require_mfem()
        vector = mfem.Vector(self._size)
        vector.Assign(0.0)
        return vector

    def copy(self, x: Any) -> Any:
        """An independent copy of an ``mfem.Vector``."""
        mfem = _require_mfem()
        return mfem.Vector(x)

    def inner_product(self, x: Any, y: Any) -> float:
        """``x^T M y``, the ``L2`` inner product of the underlying functions.

        Correct on a constrained space without restricting anything: the
        vectors are zero on the essential degrees of freedom, so the full
        quadratic form already equals the free one.
        """
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
        """The *free* degree-of-freedom values, as an array.

        A **copy**, deliberately. ``GetDataArray`` hands back a view into
        memory MFEM owns, and that view does not keep its owner alive: in an
        expression like ``to_components(from_components(c))`` the temporary
        vector is collected and the view is left pointing at freed memory. The
        symptom is silent -- plausible numbers that are simply wrong -- so the
        copy is not negotiable, and the cost of one array copy per call is the
        price of a foreign backend owning its own memory.
        """
        values = np.array(x.GetDataArray(), copy=True)
        return values if not self.is_constrained else values[self._free]

    def from_components(self, c: np.ndarray) -> Any:
        """An ``mfem.Vector``, zero on the essential degrees of freedom.

        Args:
            c: the components, one per true degree of freedom.

        Returns:
            A fresh ``mfem.Vector`` owning its own buffer -- the library frees
            what it allocates, so this must not alias the caller's array.

        Raises:
            ValueError: if the component count is not the dimension.
        """
        mfem = _require_mfem()
        values = np.asarray(c, dtype=float)
        if values.shape != (self._dim,):
            raise ValueError(f"Expected {self._dim} components, got {values.shape}.")
        vector = mfem.Vector(self._size)
        vector.Assign(0.0)
        vector.GetDataArray()[self._free] = values
        return vector

    @cached_property
    def _scipy_mass(self) -> sp.csr_matrix:
        """The Gram matrix in SciPy form: the mass matrix's free block."""
        full = _to_scipy(self._mass)
        if not self.is_constrained:
            return full
        return sp.csr_matrix(full[self._free][:, self._free])

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

        Done by MFEM. Its ``WhiteGaussianNoiseDomainLFIntegrator`` assembles a
        *load vector* whose covariance is the mass matrix, element by element
        and with no factorisation at all; one mass solve then turns that into
        components with covariance ``M^-1``. The obvious alternative — factor
        ``M`` and solve against a standard normal — needs a Cholesky of the
        mass matrix, which for a real finite element space means densifying it,
        and that is not a thing to do in a sampler.
        """
        return self.solve_gram(white_noise_load(self, rng=rng))


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
    :meth:`LinearOperator.from_matrix` in Galerkin form, and the mass solve that
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
    return LinearOperator.from_matrix(
        space, space, space.restrict(form.SpMat()), traits=traits, form="galerkin"
    )


def _to_mfem(matrix: Any) -> Any:
    """A SciPy sparse matrix as an ``mfem.SparseMatrix``.

    Built entry by entry, which is not elegant and is not the bottleneck: it
    happens once per operator, against a solve that happens many times, and it
    is linear in the number of *stored* entries — 16 ms for 33k of them. What
    did matter was the dense intermediate this used to be handed; see
    :meth:`MfemSpace.restrict`.

    **The CSR constructor is bound, and must not be used here.**
    ``mfem.SparseMatrix([indptr, indices, data, rows, columns])`` exists in
    this binding and builds a matrix whose ``Mult`` is correct — the earlier
    claim that it "has nowhere to bind" was simply wrong, and would invite
    someone to swap it in. The reason it is unusable is that MFEM then *owns*
    the three NumPy buffers it was handed and frees them in its destructor,
    while NumPy still owns them too: the process ends in ``double free or
    corruption``, reproducibly, whether or not the Python arrays are still
    alive. Anyone tempted by the faster constructor is being offered a
    segfault, not a speed-up.

    The alternative was ``BilinearForm.FormSystemMatrix``, MFEM's own way of
    imposing essential conditions. It is not used here because it **takes
    ownership of the form's matrix**: after calling it, reading ``SpMat()``
    from the same form is a use-after-free, which does not raise — it
    segfaults, and did. Building the constrained operator from the free block
    instead leaves every MFEM object the caller passed in exactly as it was.
    """
    mfem = _require_mfem()
    coordinates = matrix.tocoo()
    built = mfem.SparseMatrix(int(matrix.shape[0]), int(matrix.shape[1]))
    for row, column, value in zip(coordinates.row, coordinates.col, coordinates.data):
        built.Add(int(row), int(column), float(value))
    built.Finalize()
    return built


def _default_mfem_solver(matrix: Any, rtol: float, max_iterations: int) -> Any:
    """Conjugate gradients with a Gauss-Seidel smoother, MFEM's own.

    A reasonable default for a symmetric positive definite finite element
    system, and the point is that it is MFEM's: the mesh, the assembly and the
    solve all stay on one side of the boundary, and this library only says what
    the result *is*.
    """
    mfem = _require_mfem()
    solver = mfem.CGSolver()
    smoother = mfem.GSSmoother(matrix)
    solver.SetOperator(matrix)
    solver.SetPreconditioner(smoother)
    solver.SetRelTol(float(rtol))
    solver.SetMaxIter(int(max_iterations))
    solver.SetPrintLevel(-1)
    # MFEM's solver holds raw pointers to its operator and preconditioner and
    # owns neither. Without these references Python is free to collect the
    # smoother the moment this function returns, and the next solve reads freed
    # memory — which does not raise, it segfaults.
    solver._pygeoinf_keepalive = (matrix, smoother)
    return solver


def solver_from_bilinear_form(
    space: MfemSpace,
    form: Any,
    /,
    *,
    make_solver: Any = None,
    rtol: float = 1e-12,
    max_iterations: int = 1000,
    traits: Traits = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
    strict: bool = True,
) -> LinearOperator:
    """The *inverse* of the operator a bilinear form defines, solved by MFEM.

    Where :func:`operator_from_bilinear_form` hands over the assembled matrix
    and lets this library invert it, this hands over nothing: MFEM's solver
    solves MFEM's system, and what comes back is a :class:`LinearOperator` that
    happens to be a PDE solve. Assembly, preconditioning and the solve stay
    where they belong; the result still composes, still has an adjoint, and
    still lives in the right metric.

    The metric is the part worth spelling out. The operator of the form has
    Galerkin matrix ``K``, so its component matrix is ``M^-1 K`` and the
    inverse's is ``K^-1 M``. Applying it therefore means: take the function's
    components, multiply by the mass matrix to get a **load vector**, solve,
    and read the solution's components back. The mass multiply is what turns a
    function into the right-hand side of a weak form, and omitting it is the
    error of DESIGN.md section 5.6 in its most convincing disguise — the answer
    comes back smooth, plausible, and wrong by a mass matrix.

    Essential boundary conditions come from *space*: its free block already is
    the constrained system, so nothing is eliminated and no object the caller
    passed in is modified.

    Args:
        space: the finite element space, carrying any essential conditions.
        form: an assembled ``mfem.BilinearForm``.
        make_solver: ``matrix -> solver``, returning a configured MFEM solver
            with ``SetOperator`` already called. Conjugate gradients with a
            Gauss-Seidel smoother if omitted. This is where an application
            supplies something better — a multigrid cycle, a direct
            factorisation — without anything else changing. Keep a reference to
            anything the solver does not own; MFEM will not.
        rtol: the relative residual for the default solver; ignored when
            *make_solver* is given. An operator defined by an inexact solve is
            only as good as that solve.
        max_iterations: the cap for the default solver, likewise ignored.
        traits: claims about the *inverse*. A symmetric form gives a
            self-adjoint operator whose inverse is self-adjoint too, which is
            the usual case and the default.
        strict: raise when the solver reports it did not converge. A silently
            unconverged solve inside an operator is an operator that is quietly
            not the one it claims to be.

    Returns:
        The inverse of the form's operator, as a ``LinearOperator``.

    Raises:
        ImportError: without MFEM installed.
    """
    mfem = _require_mfem()
    system = _to_mfem(sp.csr_matrix(space.restrict(form.SpMat())))
    solver = (
        _default_mfem_solver(system, rtol, max_iterations)
        if make_solver is None
        else make_solver(system)
    )
    retained = (system, solver)
    size = space.dim

    def solve(x: Any, _retained: Any = retained) -> Any:
        # Components -> load vector. The mass multiply is the whole difference
        # between solving with a function and solving with its coefficients.
        load = space.apply_gram(space.to_components(x))

        right_hand_side = mfem.Vector(size)
        right_hand_side.GetDataArray()[:] = load
        solution = mfem.Vector(size)
        solution.Assign(0.0)
        solver.Mult(right_hand_side, solution)
        if strict and hasattr(solver, "GetConverged") and not solver.GetConverged():
            raise RuntimeError(
                f"The MFEM solver did not converge in "
                f"{solver.GetNumIterations()} iterations. Loosen rtol, raise "
                f"max_iterations, or supply a better preconditioner through "
                f"make_solver."
            )
        return space.from_components(np.array(solution.GetDataArray(), copy=True))

    adjoint = solve if Traits.SELF_ADJOINT & traits else None
    return LinearOperator.from_callables(
        space, space, solve, adjoint=adjoint, traits=traits
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
        space, space.restrict_vector(form.GetDataArray())
    )


def operator_from_linear_forms(
    space: MfemSpace,
    forms: Any,
    /,
    *,
    codomain: Any = None,
) -> LinearOperator:
    """Several linear forms stacked into one observation operator.

    An observation is a linear form — a sensor integrates the field against its
    footprint, and MFEM assembles exactly that. So an observation operator is a
    stack of load vectors, and because each row is a set of *derivative*
    components rather than a function, the whole thing is
    ``from_matrix`` and the mass solve that its adjoint needs stays
    inside the operator.

    Getting this wrong is the standard way to break a finite element inverse
    problem: treat the load vectors as though they were the sensors' kernels,
    and the adjoint comes back off by a mass matrix — smooth, plausible and
    wrong, the same disguise as in :func:`solver_from_bilinear_form`.

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
    rows = [space.restrict_vector(form.GetDataArray()) for form in forms]
    if not rows:
        raise ValueError("At least one linear form is needed.")
    if codomain is None:
        codomain = EuclideanSpace(len(rows))
    elif codomain.dim != len(rows):
        raise ValueError(
            f"{len(rows)} linear forms for a data space of dimension "
            f"{codomain.dim}."
        )
    return LinearOperator.from_matrix(space, codomain, np.array(rows), form="galerkin")


def _white_noise_integrator(seed: int) -> Any:
    """MFEM's white-noise integrator, around a broken binding.

    ``WhiteGaussianNoiseDomainLFIntegrator.__init__`` in PyMFEM ends with
    ``self._coeff = QG``, and ``QG`` is not defined anywhere — so the
    constructor raises ``NameError`` every time, after the underlying C++
    object has already been made. This does the SWIG initialisation directly
    and skips the line that fails.

    A workaround for an upstream bug, and narrow on purpose: if PyMFEM fixes
    it, the plain constructor will work and this can go.
    """
    _require_mfem()
    from mfem._ser import _lininteg, lininteg

    integrator = lininteg.WhiteGaussianNoiseDomainLFIntegrator.__new__(
        lininteg.WhiteGaussianNoiseDomainLFIntegrator
    )
    _lininteg.WhiteGaussianNoiseDomainLFIntegrator_swiginit(
        integrator,
        _lininteg.new_WhiteGaussianNoiseDomainLFIntegrator(int(seed)),
    )
    return integrator


def white_noise_load(
    space: MfemSpace, /, *, rng: Generator | None = None
) -> np.ndarray:
    """A load vector whose covariance is the mass matrix.

    The finite element discretisation of white noise: the right-hand side of
    ``(W, phi_i)`` for white noise ``W``, whose covariance is
    ``(phi_i, phi_j) == M``. MFEM assembles it directly, which is the reason to
    ask MFEM rather than to build one here — no factorisation of ``M`` is
    involved anywhere.

    Args:
        space: the finite element space; a constrained one gets the free
            entries, whose covariance is its own Gram matrix.
        rng: the generator. Only a seed reaches MFEM, drawn from this, so a run
            is reproducible from a NumPy generator like everything else here.
    """
    mfem = _require_mfem()
    seed = int(_resolve_rng(rng).integers(1, 2**31 - 1))
    # Bound to a name, not passed inline. AddDomainIntegrator does not take
    # ownership of an object built the way _white_noise_integrator has to build
    # it, so a temporary is collected before Assemble runs and the form
    # integrates freed memory — which produces a load vector containing 1e130
    # rather than an error.
    integrator = _white_noise_integrator(seed)
    form = mfem.LinearForm(space.finite_element_space)
    form.AddDomainIntegrator(integrator)
    form.Assemble()
    values = space.restrict_vector(form.GetDataArray())
    del integrator
    return values


def matern_measure(
    space: MfemSpace,
    /,
    *,
    smoothness: float = 1.0,
    correlation_length: Any = 0.1,
    rotation: float = 0.0,
    amplitude: float = 1.0,
    solver: Any = None,
    rtol: float = 1e-10,
    max_iterations: int = 1000,
) -> "GaussianMeasure":
    """A Matern random field on a finite element space, by the SPDE method.

    Lindgren, Rue and Lindstrom (2011) observed that a Gaussian field with a
    Matern covariance is the solution of a stochastic PDE,

    .. code-block:: text

        (I - div Theta grad)^a u = eta W,    a = (nu + d/2) / 2

    with ``W`` white noise. That is worth far more than a change of formula: a
    Matern covariance *matrix* is dense and needs a factorisation to sample
    from, while the differential operator is sparse and local, so a field on a
    million-cell mesh costs a few elliptic solves and never an eigenvalue.

    Everything expensive here is MFEM's. The operator is a bilinear form it
    assembles, the solves are its own conjugate gradients
    (:func:`solver_from_bilinear_form`), and the white noise is its
    ``WhiteGaussianNoiseDomainLFIntegrator`` — which is the piece worth
    borrowing rather than rebuilding, since the right-hand side of a weak form
    driven by white noise has covariance ``M`` rather than the identity, and
    assembling it directly avoids factorising the mass matrix at all.

    This layer supplies the composition and says what the result *is*: with
    ``S`` the solve and ``A`` the operator,

    .. code-block:: text

        factor      eta S^a          covariance  eta^2 A^-2a
        precision   A^2a / eta^2

    all three of which the measure is given, so it can be sampled, conditioned
    and used in a model-space formalism without anything being formed.

    Args:
        space: the finite element space, carrying any essential conditions.
        smoothness: the Matern parameter ``nu``. ``(nu + d/2) / 2`` must be a
            positive integer, which in two dimensions means odd ``nu``; the
            fractional case needs a rational approximation that PyMFEM does not
            expose (see the note below).
        correlation_length: one length, or one per dimension for an
            anisotropic field.
        rotation: the anisotropy's rotation angle, in radians. Two dimensions
            only.
        amplitude: a scale on the field. At 1.0 the pointwise standard
            deviation is 1.0 away from the boundary.
        solver: a ``matrix -> solver`` factory, passed to
            :func:`solver_from_bilinear_form` as ``make_solver``.
        rtol, max_iterations: for the default solver.

    Note:
        **The field is not stationary near the boundary.** The SPDE is posed on
        a bounded domain, so the boundary condition — whichever *space* carries
        — distorts the covariance within roughly one correlation length of it.
        This is inherent to the method rather than to this implementation, and
        the usual remedy is a domain padded by a few correlation lengths.

    Note:
        MFEM handles a fractional exponent with an AAA rational approximation,
        which its Python bindings do not expose
        (``ComputePartialFractionApproximation`` is absent). Rather than
        reimplement it here — badly, and duplicating what MFEM already
        does well — this refuses a non-integer exponent and says so.
        rtol: the relative residual for each of the solves that build the
            covariance factor.

    Returns:
        The measure, with a covariance factor built from the solves.

    Raises:
        ImportError: without MFEM installed.
        ValueError: for a non-integer or non-positive exponent, or a
            non-positive correlation length.
    """
    mfem = _require_mfem()
    from ..probability.gaussian import GaussianMeasure

    dimension = space.finite_element_space.GetMesh().Dimension()
    theta, normalisation, order = _matern_parameters(
        dimension, smoothness, correlation_length, rotation
    )

    form = mfem.BilinearForm(space.finite_element_space)
    form.AddDomainIntegrator(
        mfem.DiffusionIntegrator(mfem.MatrixConstantCoefficient(theta))
    )
    form.AddDomainIntegrator(mfem.MassIntegrator())
    form.Assemble()
    form.Finalize()

    definite = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
    operator = operator_from_bilinear_form(space, form, traits=definite)
    solve = solver_from_bilinear_form(
        space, form, make_solver=solver, rtol=rtol, max_iterations=max_iterations
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
        covariance=covariance.with_traits(definite),
        covariance_factor=scale * factor,
        precision=((1.0 / scale**2) * (powered @ powered)).with_traits(definite),
    )


def _matern_parameters(
    dimension: int, smoothness: float, correlation_length: Any, rotation: float
) -> tuple[np.ndarray, float, int]:
    """Validate the Matern parameters and turn them into the SPDE's.

    Shared by :func:`matern_measure` and its counterpart in
    :mod:`pygeoinf2.backends.mfem_hilbert`, which pose the same SPDE on the
    two kinds of space.

    Returns:
        MFEM's anisotropy matrix ``Theta``, the normalisation ``eta``, and the
        integer exponent ``(nu + d/2) / 2``.
    """
    lengths = np.atleast_1d(np.asarray(correlation_length, dtype=float))
    if lengths.size == 1:
        lengths = np.full(dimension, float(lengths[0]))
    if lengths.size != dimension:
        raise ValueError(
            f"{lengths.size} correlation lengths for a {dimension}-dimensional "
            f"mesh."
        )
    if np.any(lengths <= 0.0):
        raise ValueError("Every correlation length must be positive.")
    if smoothness <= 0.0:
        raise ValueError(f"The smoothness must be positive, got {smoothness}.")

    exponent = (smoothness + dimension / 2.0) / 2.0
    order = int(round(exponent))
    if abs(exponent - order) > 1e-9 or order < 1:
        raise ValueError(
            f"(nu + d/2)/2 == {exponent:.4g} is not a positive integer. MFEM "
            f"reaches the fractional case with a rational approximation that "
            f"its Python bindings do not expose, so only the integer case is "
            f"available here. In {dimension} dimensions, try nu = "
            f"{', '.join(str(2 * k - dimension / 2) for k in (1, 2, 3))}."
        )
    theta = _anisotropy(lengths, rotation, smoothness, dimension)
    normalisation = _matern_normalisation(smoothness, lengths, dimension)
    return theta, normalisation, order


def _anisotropy(
    lengths: np.ndarray, rotation: float, smoothness: float, dimension: int
) -> np.ndarray:
    """``R^T diag(l^2) R / (2 nu)``, MFEM's Theta.

    The correlation lengths enter squared and divided by twice the smoothness,
    which is what makes ``correlation_length`` the distance at which the
    correlation has fallen to about 0.14 rather than an arbitrary scale.
    """
    scaled = lengths**2 / (2.0 * smoothness)
    if dimension == 2 and rotation != 0.0:
        cosine, sine = np.cos(rotation), np.sin(rotation)
        rotate = np.array([[cosine, sine], [-sine, cosine]])
        return rotate @ np.diag(scaled) @ rotate.T
    if rotation != 0.0:
        raise ValueError(
            f"A rotation is only defined for a two-dimensional mesh; this one "
            f"is {dimension}-dimensional."
        )
    return np.diag(scaled)


def _matern_normalisation(
    smoothness: float, lengths: np.ndarray, dimension: int
) -> float:
    """The ``eta`` that makes the pointwise variance one.

    MFEM's ``ConstructNormalizationCoefficient``, which is where the Matern
    constants live:

    .. code-block:: text

        eta = sqrt( (2 pi)^(d/2) prod(l) Gamma(nu + d/2)
                    / ( Gamma(nu) nu^(d/2) ) )
    """
    from math import gamma

    return float(
        np.sqrt(
            (2.0 * np.pi) ** (dimension / 2.0)
            * float(np.prod(lengths))
            * gamma(smoothness + dimension / 2.0)
            / (gamma(smoothness) * smoothness ** (dimension / 2.0))
        )
    )
