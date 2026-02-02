import pygeoinf as pgf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use a non-interactive backend for environments without display

M = pgf.EuclideanSpace(2)
n_1 = np.asarray([1.0, 1.0]) / np.sqrt(2)
n_2 = np.asarray([1.0, -1.0]) / np.sqrt(2)
n_3 = np.asarray([-1.0, 0.0])
offset = 1.0
halfspace_1 = pgf.subsets.HalfSpace(M, n_1, offset)
halfspace_2 = pgf.subsets.HalfSpace(M, n_2, offset)
halfspace_3 = pgf.subsets.HalfSpace(M, n_3, offset)
polyhedra = pgf.subsets.PolyhedralSet(
    M,
    [halfspace_1, halfspace_2, halfspace_3]
)

subspace = pgf.subspaces.AffineSubspace.from_tangent_basis(
    domain=M,
    basis_vectors=[M.basis_vector(0),
                   M.basis_vector(1)]
)

pgf.plot.plot_subset_oracle(polyhedra,
                            on_subspace=subspace,
                            bounds=((-2, 2), (-2, 2)),
                            show_plot=True)