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
# Add two additional half-spaces to produce a more constrained polyhedron
n_4 = np.asarray([0.0, -1.0])
n_5 = np.asarray([0.0, 1.0])
offset4 = 2.0
offset5 = 2.0
halfspace_4 = pgf.subsets.HalfSpace(M, n_4, offset4)
halfspace_5 = pgf.subsets.HalfSpace(M, n_5, offset5)

polyhedra = pgf.subsets.PolyhedralSet(
    M,
    [halfspace_1, halfspace_2, halfspace_3, halfspace_4, halfspace_5]
)

subspace = pgf.subspaces.AffineSubspace.from_tangent_basis(
    domain=M,
    basis_vectors=[M.basis_vector(0),
                   M.basis_vector(1)],
    solver=None
)

plotter = pgf.visualization.SubspaceSlicePlotter(polyhedra, subspace)

fig, ax, mask = plotter.plot(bounds=[(-5.0, 5.0), (-5.0, 5.0)])