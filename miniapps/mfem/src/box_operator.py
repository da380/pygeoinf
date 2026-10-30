"""An operator on one of the library's own spaces whose action is an MFEM solve.

``hilbert_space`` and ``matrix_space`` wrap an MFEM function space as a
Hilbert space. This module does the opposite: the space is the library's own
two-dimensional periodic box, whose vectors are grid arrays and whose
components are Fourier coefficients, and MFEM appears only inside the action
of an operator on it. The library never sees an MFEM space, an MFEM vector or
a finite element.

The operator solves the screened Poisson equation

.. code-block:: text

    (1 - s^2 Laplacian) u = f

on the periodic box, with ``s`` the screening length, in three steps:

1. **The load.** The field ``f`` is integrated against MFEM's basis
   functions by MFEM's quadrature rule, ``b_i = sum_q w_q phi_i(x_q) f(x_q)``,
   with ``f`` read at the quadrature points by the box's non-uniform
   transform. That is the right-hand side of the weak form itself, so the
   field is never interpolated onto the mesh.
2. **The solve.** ``K u = b`` with ``K = M + s^2 D``, the mass and diffusion
   matrices MFEM assembles on a periodic mesh of the same box, by MFEM's
   conjugate gradients with its symmetric Gauss-Seidel smoother.
3. **The evaluation.** The finite element solution read at the box's grid
   points, which is what a vector of the box is.

In components, with ``B`` the Fourier basis at the quadrature points, ``Q``
the shape functions times the quadrature weights, ``P`` the shape functions
at the grid points and ``T`` the box's map from grid values to components,
the operator is ``T P K^-1 Q B``. Its adjoint in the box's metric ``G`` is

.. code-block:: text

    A* = G^-1 (T P K^-1 Q B)^T G = G^-1 B^T Q^T K^-1 P^T T^T G

the transpose with the inverse Gram factor on the left and the Gram factor
on the right. ``K^-1`` is its own transpose because the system is symmetric,
so the adjoint costs one more solve of the same system; ``B^T`` is the box's
type-1 non-uniform transform, the exact transpose of the type-2 one that
gave ``B``; and ``T^T = (V / N) T^-1`` for a box of volume ``V`` on ``N``
grid points, because the box's basis is orthonormal on its own grid.

Each step is a :class:`~pygeoinf.algebra.operators.LinearOperator` of its
own, between the box and a Euclidean space of finite element coefficients,
so that the check ladder can be run on them one at a time; the operator is
their composition, and its adjoint is the library's composition of theirs.
The Euclidean dot product on the coefficients is the pairing of a load with
a solution, ``b . u == integral of f u``, which is all the composition needs.

Requires PyMFEM, which is the ``mfem`` extra of the package
(``poetry install --extras mfem``); its wheels are built for Linux only.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import scipy.sparse as sp

from pygeoinf.algebra.operators import LinearOperator
from pygeoinf.algebra.spaces import EuclideanSpace
from pygeoinf.numerics.solvers import ConvergenceError
from pygeoinf.symmetric_space.fourier import PeriodicBox
from pygeoinf.traits import Traits

from .matrix_space import _require_mfem

__all__ = ["ScreenedPoisson"]


class ScreenedPoisson:
    """The screened Poisson solve on a periodic box, by finite elements in MFEM.

    Builds a periodic mesh of quadrilaterals over the box, an ``H1`` space of
    the given order on it, the system matrix, MFEM's solver for it, and the
    two sparse transfer matrices between the box and the mesh; then exposes
    the three steps and their composition as operators on the box.

    The mesh need not match the box's grid: the transfers are evaluations of
    the box's Fourier series at MFEM's quadrature points and of MFEM's
    shape functions at the box's grid points, wherever those fall.
    """

    def __init__(
        self,
        space: PeriodicBox,
        /,
        *,
        elements: int,
        screening_length: float,
        order: int = 2,
        rtol: float = 1e-14,
        max_iterations: int = 2000,
    ) -> None:
        """
        Args:
            space: a two-dimensional periodic box: the domain and the
                codomain of the operator.
            elements: quadrilaterals along each side of the mesh.
            screening_length: ``s`` in ``(1 - s^2 Laplacian) u = f``, in the
                box's units.
            order: the polynomial order of the finite elements.
            rtol: the relative residual of MFEM's conjugate gradients. Tight
                by default, because the operator and its adjoint agree only
                as well as the solves do.
            max_iterations: the cap on each solve.

        Raises:
            ValueError: for a box that is not two-dimensional, a mesh of
                fewer than two elements a side, a non-positive screening
                length or an order below one.
        """
        mfem = _require_mfem()
        if not isinstance(space, PeriodicBox) or space.spatial_dimension != 2:
            raise ValueError(
                f"A two-dimensional periodic box is needed, got {space!r}."
            )
        if elements < 2:
            raise ValueError(
                f"At least two elements a side are needed, got {elements}."
            )
        if screening_length <= 0.0:
            raise ValueError(
                f"The screening length must be positive, got {screening_length}."
            )
        if order < 1:
            raise ValueError(f"The order must be at least one, got {order}.")

        self._space = space
        self._screening = float(screening_length)
        self._max_iterations = int(max_iterations)
        width, height = (float(length) for length in space.lengths)

        # The mesh is periodic in both directions: MFEM identifies the
        # vertices on opposite sides, so the finite element space carries
        # no boundary and its functions are periodic, as the box's are.
        square = mfem.Mesh.MakeCartesian2D(
            int(elements),
            int(elements),
            mfem.Element.QUADRILATERAL,
            False,
            width,
            height,
        )
        mapping = square.CreatePeriodicVertexMapping(
            [mfem.Vector([width, 0.0]), mfem.Vector([0.0, height])]
        )
        mesh = mfem.Mesh.MakePeriodic(square, mapping)
        collection = mfem.H1_FECollection(int(order), 2)
        elements_space = mfem.FiniteElementSpace(mesh, collection)
        self._size = int(elements_space.GetVSize())
        self._coefficients = EuclideanSpace(self._size)

        self._load_matrix, self._quadrature_points = _load_matrix(
            mfem, mesh, elements_space, 2 * int(order) + 2
        )
        self._prepared = space.prepare_points(self._quadrature_points)
        grid = np.stack(np.meshgrid(*space.grid_axes, indexing="ij"), axis=-1)
        self._evaluation_matrix = _evaluation_matrix(
            mfem, mesh, elements_space, grid.reshape(-1, 2)
        )
        self._cell = space.volume / float(np.prod(space.shape))

        form = mfem.BilinearForm(elements_space)
        form.AddDomainIntegrator(
            mfem.DiffusionIntegrator(mfem.ConstantCoefficient(self._screening**2))
        )
        form.AddDomainIntegrator(mfem.MassIntegrator())
        form.Assemble()
        form.Finalize()
        self._matrix = form.SpMat()
        # Everything MFEM holds a raw pointer to lives as long as this object:
        # the element space points to the mesh and the collection, the form
        # to the element space, and the system matrix belongs to the form.
        self._retained = (square, mesh, collection, elements_space, form)
        self._lock = threading.Lock()
        self.solves = 0
        self.iterations = 0
        self.rtol = float(rtol)
        self._solver = self._mfem_solver(self.rtol)

    # ----------------------------------------------------------------- #
    #                              Structure                            #
    # ----------------------------------------------------------------- #

    @property
    def space(self) -> PeriodicBox:
        """The box: the operator's domain and codomain."""
        return self._space

    @property
    def screening_length(self) -> float:
        """``s`` in ``(1 - s^2 Laplacian) u = f``."""
        return self._screening

    @property
    def size(self) -> int:
        """The number of finite element degrees of freedom."""
        return self._size

    @property
    def coefficients(self) -> EuclideanSpace:
        """The Euclidean space of finite element coefficients and loads."""
        return self._coefficients

    @property
    def quadrature_points(self) -> np.ndarray:
        """MFEM's quadrature points over the whole mesh, one row each."""
        return self._quadrature_points

    @property
    def load_matrix(self) -> sp.csr_matrix:
        """``Q``: shape function times quadrature weight, a dof by a point."""
        return self._load_matrix

    @property
    def evaluation_matrix(self) -> sp.csr_matrix:
        """``P``: shape functions at the box's grid points, a point by a dof."""
        return self._evaluation_matrix

    @property
    def cell_volume(self) -> float:
        """``V / N``: the box's volume over its number of grid points."""
        return self._cell

    def reset_counts(self) -> None:
        """Zero the counts of solves and of conjugate-gradient iterations."""
        self.solves = self.iterations = 0

    # ----------------------------------------------------------------- #
    #                              The steps                            #
    # ----------------------------------------------------------------- #

    def _mfem_solver(self, rtol: float) -> tuple[Any, ...]:
        """MFEM's conjugate gradients on the system, with the objects it points to.

        MFEM's solver holds raw pointers to its operator and its smoother and
        owns neither; the tuple returned keeps them alive with it.
        """
        mfem = _require_mfem()
        solver = mfem.CGSolver()
        smoother = mfem.GSSmoother(self._matrix)
        solver.SetOperator(self._matrix)
        solver.SetPreconditioner(smoother)
        solver.SetRelTol(float(rtol))
        solver.SetAbsTol(0.0)
        solver.SetMaxIter(self._max_iterations)
        solver.SetPrintLevel(-1)
        return solver, smoother, self._matrix, float(rtol)

    def _solve(self, load: np.ndarray, solver: tuple[Any, ...]) -> np.ndarray:
        """``K^-1 b`` by MFEM, from zero, into a fresh array.

        Serialized by a lock: MFEM's solver keeps its work vectors between
        calls, so two threads in one solver would share them.
        """
        mfem = _require_mfem()
        cg, rtol = solver[0], solver[-1]
        right_hand_side = mfem.Vector(self._size)
        right_hand_side.GetDataArray()[:] = np.asarray(load, dtype=float)
        solution = mfem.Vector(self._size)
        solution.Assign(0.0)
        with self._lock:
            cg.Mult(right_hand_side, solution)
            converged = cg.GetConverged()
            iterations = cg.GetNumIterations()
            self.solves += 1
            self.iterations += iterations
        if not converged:
            raise ConvergenceError(
                f"MFEM's conjugate gradients did not reach a relative residual "
                f"of {rtol:g} in {iterations} iterations."
            )
        return np.array(solution.GetDataArray(), copy=True)

    @property
    def sampling(self) -> LinearOperator:
        """The load: a field of the box to the weak form's right-hand side.

        ``x -> Q B c``, the field read at the quadrature points by the box's
        type-2 transform and summed against the weighted shape functions.
        Its adjoint ``G^-1 B^T Q^T`` is the type-1 transform of the weighted
        shape-function sums, which gives derivative components, and the
        inverse Gram factor that turns them into a vector of the box.
        """
        space, points, matrix = self._space, self._prepared, self._load_matrix

        def value(x: np.ndarray) -> np.ndarray:
            return matrix @ space.evaluate(x, points)

        def adjoint(b: np.ndarray) -> np.ndarray:
            derivative = space.accumulate(matrix.T @ np.asarray(b), points)
            return space.from_components(space.solve_gram(derivative))

        return LinearOperator.from_callables(
            space, self._coefficients, value, adjoint=adjoint
        )

    def solve_at(self, rtol: float, /) -> LinearOperator:
        """The solve ``b -> K^-1 b`` by MFEM, at a given relative residual.

        Self-adjoint and positive definite in the dot product, because ``K``
        is symmetric and positive definite: the operator is its own adjoint,
        up to the tolerance of the solve.

        Args:
            rtol: the relative residual of MFEM's conjugate gradients.

        Returns:
            The solve, on the coefficient space.
        """
        solver = self._solver if rtol == self.rtol else self._mfem_solver(rtol)

        def value(b: np.ndarray, _solver: tuple[Any, ...] = solver) -> np.ndarray:
            return self._solve(b, _solver)

        return LinearOperator.self_adjoint(
            self._coefficients, value, traits=Traits.POSITIVE_DEFINITE
        )

    @property
    def solve(self) -> LinearOperator:
        """The solve ``b -> K^-1 b`` at the tolerance the operator was built with."""
        return self.solve_at(self.rtol)

    @property
    def evaluation(self) -> LinearOperator:
        """A finite element function read at the box's grid points.

        ``u -> P u`` as grid values, which is a vector of the box. Its adjoint
        ``P^T T^T G`` takes a vector of the box to the grid values of its
        Gram image, ``(V / N) T^-1 G c``, and sums them against the shape
        functions.
        """
        space, matrix, cell = self._space, self._evaluation_matrix, self._cell

        def value(u: np.ndarray) -> np.ndarray:
            return space.from_grid_values((matrix @ np.asarray(u)).reshape(space.shape))

        def adjoint(y: np.ndarray) -> np.ndarray:
            weighted = space.from_components(space.apply_gram(space.to_components(y)))
            return matrix.T @ (cell * np.asarray(weighted).ravel())

        return LinearOperator.from_callables(
            self._coefficients, space, value, adjoint=adjoint
        )

    def operator_at(self, rtol: float, /) -> LinearOperator:
        """The whole solve, ``f -> u``, with MFEM's solves at a given tolerance.

        Args:
            rtol: the relative residual of MFEM's conjugate gradients.

        Returns:
            ``evaluation @ solve @ sampling``, an operator on the box.
        """
        return self.evaluation @ self.solve_at(rtol) @ self.sampling

    @property
    def operator(self) -> LinearOperator:
        """The whole solve, ``f -> u``, at the tolerance it was built with."""
        return self.operator_at(self.rtol)

    @property
    def continuum(self) -> LinearOperator:
        """The operator the finite elements approximate, applied exactly.

        On a periodic box the screened Poisson solve is the Fourier
        multiplier ``1 / (1 + s^2 |k|^2)``, which the box applies on its own
        coefficients. The finite element operator differs from it by the
        discretization, and that difference falls under mesh refinement.
        """
        scale = self._screening**2
        return self._space.invariant_operator(
            lambda eigenvalue: 1.0 / (1.0 + scale * eigenvalue)
        )

    def __repr__(self) -> str:
        return (
            f"ScreenedPoisson({self._space!r}, dofs={self._size}, "
            f"screening_length={self._screening})"
        )


def _load_matrix(
    mfem: Any, mesh: Any, elements_space: Any, quadrature_order: int
) -> tuple[sp.csr_matrix, np.ndarray]:
    """``Q`` and the quadrature points: ``Q[i, q] = w_q phi_i(x_q)``.

    One pass over the elements with MFEM's own rule, transformation and
    shape functions; ``w_q`` carries the Jacobian, so that ``Q f`` is the
    load ``integral of f phi_i`` once ``f`` holds the field's values at the
    points.
    """
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    points: list[np.ndarray] = []
    count = 0
    for element in range(mesh.GetNE()):
        transformation = mesh.GetElementTransformation(element)
        finite_element = elements_space.GetFE(element)
        rule = mfem.IntRules.Get(finite_element.GetGeomType(), quadrature_order)
        dofs = np.asarray(elements_space.GetElementDofs(element), dtype=int)
        shape = mfem.Vector(finite_element.GetDof())
        for index in range(rule.GetNPoints()):
            point = rule.IntPoint(index)
            transformation.SetIntPoint(point)
            finite_element.CalcShape(point, shape)
            weight = point.weight * transformation.Weight()
            points.append(np.array(transformation.Transform(point), dtype=float))
            rows.append(dofs)
            columns.append(np.full(dofs.size, count))
            values.append(weight * np.array(shape.GetDataArray(), copy=True))
            count += 1
    matrix = sp.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(int(elements_space.GetVSize()), count),
    )
    return matrix, np.stack(points)


def _evaluation_matrix(
    mfem: Any, mesh: Any, elements_space: Any, points: np.ndarray
) -> sp.csr_matrix:
    """``P``: ``P[p, i] = phi_i(x_p)``, from MFEM's point location.

    A point on an element edge belongs to either neighbour, and the
    functions are continuous there, so whichever MFEM reports is right.

    Raises:
        RuntimeError: if MFEM fails to place a point in the mesh.
    """
    found, located, reference = mesh.FindPoints(
        np.ascontiguousarray(points), warn=False
    )
    if found != len(points):
        raise RuntimeError(f"MFEM placed {found} of {len(points)} grid points.")
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for index, element in enumerate(located):
        finite_element = elements_space.GetFE(element)
        shape = mfem.Vector(finite_element.GetDof())
        finite_element.CalcShape(reference[index], shape)
        dofs = np.asarray(elements_space.GetElementDofs(element), dtype=int)
        rows.append(np.full(dofs.size, index))
        columns.append(dofs)
        values.append(np.array(shape.GetDataArray(), copy=True))
    return sp.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(points), int(elements_space.GetVSize())),
    )
