"""Polyhedral approximations of DLI feasible regions in property space.

Builds an incremental polyhedral approximation of the convex feasible
property region by caching directional DLI bounds. Each direction $q$ on
the unit sphere of property space yields one supporting hyperplane via:

    h_U(q) = inf_{λ ∈ D} φ(λ; q)

which translates to the half-space constraint ⟨q, p⟩ ≤ h_U(q).

The cached bounds are exported as a :class:`~pygeoinf.subsets.PolyhedralSet`,
which is directly accepted by :func:`~pygeoinf.plot.plot_slice` via its
fast exact ``scipy.spatial.HalfspaceIntersection`` path.

Typical usage::

    from pygeoinf.backus_gilbert import DualMasterCostFunction
    from pygeoinf.convex_analysis import BallSupportFunction
    from pygeoinf.convex_optimisation import ProximalBundleMethod
    from pygeoinf.polyhedral_approximation import PolyhedralApproximation
    from pygeoinf.plot import plot_slice
    import numpy as np

    cost = DualMasterCostFunction(
        data_space, property_space, model_space,
        forward_op, property_op,
        BallSupportFunction(model_space, model_space.zero, prior_radius),
        BallSupportFunction(data_space, data_space.zero, data_radius),
        observed_data,
        q_direction=property_space.basis_vector(0),
    )
    solver = ProximalBundleMethod(cost, tolerance=1e-4, max_iterations=200)

    approx = PolyhedralApproximation(property_space, cost, solver)
    approx.initialize("box")
    approx.refine("random_uniform", n_new=50)

    # Visualize Cap 1 vs Cap 2 slice — uses PolyhedralSet exact path
    subspace = np.eye(property_space.dim)[:, [0, 1]]
    fig, ax, _ = approx.plot(subspace=subspace)
"""

from __future__ import annotations

import numpy as np

from .subsets import HalfSpace, PolyhedralSet


def _is_primal_kkt_solver(solver) -> bool:
    """Check if solver is a PrimalKKTSolver."""
    return type(solver).__name__ == "PrimalKKTSolver"


class DirectionSampler:
    """Factory for generating direction sets on the unit sphere.

    All methods return an array of shape ``(n_directions, n_dims)`` with
    unit-norm rows. Directions can be passed to
    :meth:`PolyhedralApproximation.add_directions` directly.
    """

    @staticmethod
    def get(
        strategy: str,
        n_dims: int,
        n_new: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Dispatch to a named strategy.

        Args:
            strategy: One of ``"box"``, ``"simplex"``, ``"random_uniform"``.
            n_dims: Dimensionality of property space.
            n_new: Number of directions (required for ``"random_uniform"``).
            **kwargs: Forwarded to the selected sampler.

        Returns:
            Array of shape ``(n_directions, n_dims)`` of unit vectors.
        """
        if strategy == "box":
            return DirectionSampler.box(n_dims)
        elif strategy == "simplex":
            return DirectionSampler.simplex(n_dims, **kwargs)
        elif strategy == "random_uniform":
            if n_new is None:
                raise ValueError("n_new is required for strategy='random_uniform'")
            return DirectionSampler.random_uniform(n_dims, n_new, **kwargs)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Choose from 'box', 'simplex', 'random_uniform'."
            )

    @staticmethod
    def box(n_dims: int) -> np.ndarray:
        """Cardinal axis-aligned directions $\\pm e_i$.

        Returns $2 n$ unit vectors: the $n$ standard basis vectors and their
        negations. Produces an axis-aligned hyperbox approximation.

        Args:
            n_dims: Dimensionality of property space.

        Returns:
            Array of shape ``(2 * n_dims, n_dims)``.
        """
        return np.vstack([np.eye(n_dims), -np.eye(n_dims)])

    @staticmethod
    def simplex(
        n_dims: int,
        random_state: int | np.random.Generator | None = None,
        max_attempts: int = 1000,
        min_volume: float = 1e-3,
    ) -> np.ndarray:
        """Random unit directions forming a non-degenerate simplex.

        Generates $n + 1$ random unit vectors and resamples until the first
        $n$ rows have an absolute determinant above ``min_volume``, ensuring
        the directions span all of $\\mathbb{R}^n$.

        Args:
            n_dims: Dimensionality of property space.
            random_state: Seed or Generator for reproducibility.
            max_attempts: Maximum resampling attempts before raising.
            min_volume: Minimum absolute determinant of the first $n$ rows.

        Returns:
            Array of shape ``(n_dims + 1, n_dims)`` of unit vectors.

        Raises:
            RuntimeError: If a valid simplex is not found within ``max_attempts``.
        """
        rng = np.random.default_rng(random_state)
        for _ in range(max_attempts):
            D = rng.standard_normal((n_dims + 1, n_dims))
            norms = np.linalg.norm(D, axis=1, keepdims=True)
            D /= norms
            if np.abs(np.linalg.det(D[:n_dims])) > min_volume:
                return D
        raise RuntimeError(
            f"Could not find a valid simplex after {max_attempts} attempts. "
            "Try increasing max_attempts or lowering min_volume."
        )

    @staticmethod
    def random_uniform(
        n_dims: int,
        n_directions: int,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Uniformly random unit directions on the sphere.

        Draws directions from the isotropic Gaussian and normalizes, which
        gives a uniform distribution on the unit sphere.

        Args:
            n_dims: Dimensionality of property space.
            n_directions: Number of directions to generate.
            random_state: Seed or Generator for reproducibility.

        Returns:
            Array of shape ``(n_directions, n_dims)`` of unit vectors.
        """
        rng = np.random.default_rng(random_state)
        D = rng.standard_normal((n_directions, n_dims))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        return D


class PolyhedralApproximation:
    """Incremental polyhedral approximation of a DLI feasible region.

    Caches directional DLI bounds and exposes the result as a
    :class:`~pygeoinf.subsets.PolyhedralSet` for visualization via
    :func:`~pygeoinf.plot.plot_slice`.

    Each direction $q$ on the unit sphere in property space yields one
    cached support value $h_U(q)$ and a corresponding half-space:

    .. math::

        \\langle q, p \\rangle \\leq h_U(q)

    The polyhedral approximation is the intersection of all such half-spaces
    over all cached directions.

    Args:
        property_space: The :class:`~pygeoinf.hilbert_space.EuclideanSpace`
            in which the property vectors live.
        cost_function: A :class:`~pygeoinf.backus_gilbert.DualMasterCostFunction`
            instance. Its direction is updated in place via ``set_direction``
            before each solve.
        solver: An optimization routine with a ``solve(x0)`` method that
            returns a result object with an ``f_best`` attribute (e.g.
            :class:`~pygeoinf.convex_optimisation.ProximalBundleMethod` or
            :class:`~pygeoinf.convex_optimisation.SubgradientDescent`). The
            solver's oracle must be the same object as ``cost_function``.
    """

    def __init__(self, property_space, cost_function, solver, property_operator=None) -> None:
        self._property_space = property_space
        self._cost = cost_function
        self._solver = solver
        self._property_operator = property_operator  # for PrimalKKTSolver
        self._is_primal_kkt = _is_primal_kkt_solver(solver)
        # Maps tuple(rounded unit direction) -> h_U(q) value
        self._cache: dict[tuple, float] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def n_constraints(self) -> int:
        """Number of half-space constraints (one per cached direction)."""
        return len(self._cache)

    def initialize(self, strategy: str = "box") -> None:
        """Initialize the approximation with a named direction strategy.

        Args:
            strategy: ``"box"`` (axis-aligned hyperbox), ``"simplex"``
                ($n+1$ random simplex directions), or ``"random_uniform"``
                (not valid without ``n_new``; prefer :meth:`refine` for that).
        """
        if strategy == "random_uniform":
            raise ValueError(
                "Use refine('random_uniform', n_new=...) for random directions."
            )
        directions = DirectionSampler.get(strategy, self._property_space.dim)
        self.add_directions(directions)

    def refine(
        self,
        strategy: str = "random_uniform",
        n_new: int = 50,
        **kwargs,
    ) -> None:
        """Add more directions to refine the approximation.

        New directions that are near-duplicates of already cached ones are
        silently skipped.

        Args:
            strategy: Direction strategy — see :class:`DirectionSampler`.
            n_new: Number of new directions to generate (used by
                ``"random_uniform"``; ignored by ``"box"`` and ``"simplex"``).
            **kwargs: Forwarded to the sampler (e.g. ``random_state``).
        """
        directions = DirectionSampler.get(
            strategy, self._property_space.dim, n_new=n_new, **kwargs
        )
        self.add_directions(directions)

    def add_directions(self, directions: np.ndarray) -> None:
        """Solve and cache bounds for an explicit set of directions.

        Directions are normalized to unit length. Near-duplicates (within
        a rounded 12-decimal-place tolerance) of already-cached directions
        are skipped without re-solving.

        Args:
            directions: Array of shape ``(n, dim)`` or ``(dim,)`` containing
                direction vectors. Need not be pre-normalized.
        """
        directions = np.atleast_2d(np.asarray(directions, dtype=float))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        if np.any(norms < 1e-14):
            raise ValueError("Direction array contains a zero vector.")
        directions = directions / norms

        for q_vec in directions:
            key = tuple(np.round(q_vec, 12))
            if key in self._cache:
                continue
            self._cache[key] = self._solve_bound(q_vec)

    def as_polyhedral_set(self) -> PolyhedralSet:
        """Return the current approximation as a :class:`~pygeoinf.subsets.PolyhedralSet`.

        The returned object is accepted by :func:`~pygeoinf.plot.plot_slice`
        which uses the fast exact ``scipy.spatial.HalfspaceIntersection``
        code path for :class:`~pygeoinf.subsets.PolyhedralSet`.

        Returns:
            A :class:`~pygeoinf.subsets.PolyhedralSet` on ``property_space``
            defined by one ``<=`` half-space per cached direction.

        Raises:
            RuntimeError: If no directions have been added yet.
        """
        if not self._cache:
            raise RuntimeError(
                "No directions cached. Call initialize() or add_directions() first."
            )

        half_spaces = []
        for q_tuple, h_val in self._cache.items():
            q_vec = np.array(q_tuple)
            q = self._property_space.from_components(q_vec)
            half_spaces.append(
                HalfSpace(self._property_space, q, h_val, inequality_type="<=")
            )

        return PolyhedralSet(self._property_space, half_spaces)

    def plot(self, dims=None, **kwargs):
        """Plot a 2D slice of the polyhedral approximation.

        Constructs an :class:`~pygeoinf.subspaces.AffineSubspace` from two
        property-space coordinate indices and delegates to
        :func:`~pygeoinf.plot.plot_slice`, which uses the fast exact
        ``scipy.spatial.HalfspaceIntersection`` path for
        :class:`~pygeoinf.subsets.PolyhedralSet`.

        Args:
            dims: Pair of integer property indices to use as the slice axes,
                e.g. ``[0, 1]`` for the first two properties. Defaults to
                ``[0, 1]``.
            **kwargs: Forwarded to :func:`~pygeoinf.plot.plot_slice` (e.g.
                ``bounds``, ``alpha``, ``show_plot``).

        Returns:
            ``(fig, ax, payload)`` from :func:`~pygeoinf.plot.plot_slice`.
        """
        from .plot import plot_slice
        from .subspaces import AffineSubspace

        if dims is None:
            dims = [0, 1]
        if len(dims) != 2:
            raise ValueError("dims must be a pair of property indices.")

        n = self._property_space.dim
        e_i = self._property_space.from_components(np.eye(n)[dims[0]])
        e_j = self._property_space.from_components(np.eye(n)[dims[1]])
        subspace = AffineSubspace.from_tangent_basis(
            self._property_space, [e_i, e_j]
        )

        return plot_slice(
            self.as_polyhedral_set(), on_subspace=subspace, **kwargs
        )

    def plot_3d(self, dims=None, **kwargs):
        """Plot a 3D slice of the polyhedral approximation.

        Constructs an :class:`~pygeoinf.subspaces.AffineSubspace` from three
        property-space coordinate indices and delegates to
        :func:`~pygeoinf.plot.plot_slice`, which uses the fast exact
        ``scipy.spatial.HalfspaceIntersection`` path for
        :class:`~pygeoinf.subsets.PolyhedralSet`.

        Args:
            dims: Triple of integer property indices to use as the slice axes,
                e.g. ``[0, 1, 2]`` for the first three properties. Defaults to
                ``[0, 1, 2]``.
            **kwargs: Forwarded to :func:`~pygeoinf.plot.plot_slice` (e.g.
                ``bounds``, ``alpha``, ``show_plot``, ``backend``).

        Returns:
            ``(fig, ax, payload)`` from :func:`~pygeoinf.plot.plot_slice`.
        """
        from .plot import plot_slice
        from .subspaces import AffineSubspace

        if dims is None:
            dims = [0, 1, 2]
        if len(dims) != 3:
            raise ValueError("dims must be a triple of property indices.")

        n = self._property_space.dim
        e_i = self._property_space.from_components(np.eye(n)[dims[0]])
        e_j = self._property_space.from_components(np.eye(n)[dims[1]])
        e_k = self._property_space.from_components(np.eye(n)[dims[2]])
        subspace = AffineSubspace.from_tangent_basis(
            self._property_space, [e_i, e_j, e_k]
        )

        return plot_slice(
            self.as_polyhedral_set(), on_subspace=subspace, **kwargs
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _solve_bound(self, q_vec: np.ndarray) -> float:
        """Solve for h_U(q) using either PrimalKKTSolver or bundle method.

        Args:
            q_vec: Unit vector in property space (numpy array).

        Returns:
            The support value $h_U(q) = \\inf_\\lambda \\varphi(\\lambda; q)$.
        """
        q = self._property_space.from_components(q_vec)

        if self._is_primal_kkt:
            # PrimalKKTSolver: solve the primal problem directly
            # c = T^* q in model space
            # h_U(q) = max_u <c, u> subject to feasible set = <c, u*>
            c = self._property_operator.adjoint(q)
            result = self._solver.solve(c)
            model_space = self._property_operator.domain
            h_val = float(model_space.inner_product(c, result.m))
            return h_val
        else:
            # DualMasterCostFunction + bundle method
            self._cost.set_direction(q)
            x0 = self._cost.domain.zero
            result = self._solver.solve(x0)
            return float(result.f_best)
