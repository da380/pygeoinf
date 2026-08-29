from common import *
from pygeoinf2 import LinearOperator, EuclideanSpace, DiagonalLinearOperator, Traits, DirectSum, BlockLinearOperator, JacobiPreconditioner, BandedPreconditioner, CGSolver, ColumnLinearOperator
from pygeoinf2.algebra.spaces import MassWeightedSpace
from pygeoinf2.testing import check_traits, check_operator, check_coordinates
rng = np.random.default_rng(3)
def ok(label, cond): print(("ok  " if cond else "FAIL"), label)
for n in (3, 30):
    D = make_dense_metric_space(n); G = D._gram
    M = rng.standard_normal((n, n))
    Ac = LinearOperator.from_matrix(D, D, M, form="components")
    Ag = LinearOperator.from_matrix(D, D, G @ M, form="galerkin")
    ok(f"n={n} form conversion components->galerkin", np.allclose(Ac.matrix(form="galerkin"), G @ M))
    ok(f"n={n} form conversion galerkin->components", np.allclose(Ag.matrix(form="components"), M))
    ok(f"n={n} diagonals(galerkin, offsets -1..1) from components-stored", np.allclose(Ac.diagonals(offsets=(-1, 0, 1), form="galerkin"), LinearOperator.from_callables(D, D, Ac, adjoint=Ac.adjoint).diagonals(offsets=(-1,0,1), form="galerkin")))
    x = rng.standard_normal(n)
    ok(f"n={n} galerkin-stored value == G^-1 M c", np.allclose(Ag(x), np.linalg.solve(G, G @ M @ x)))
    check_operator(Ac, rng=rng); check_operator(Ag, rng=rng)
    # DiagonalLinearOperator on dense metric: adjoint and diagonals fall back correctly
    d = rng.uniform(1, 2, n); Dd = DiagonalLinearOperator(D, d)
    check_operator(Dd, rng=rng)
    ok(f"n={n} Diagonal.diagonals(galerkin) on dense metric == diag(G diag(d))", np.allclose(Dd.diagonals(form="galerkin")[0], np.diag(G @ np.diag(d))))
    ok(f"n={n} Diagonal traits on dense metric are NONE", Dd.traits == Traits.NONE)
    # from_formal_adjoint: two dense spaces over the same vectors
    D2 = make_dense_metric_space(n)   # same gram; build a second, different one
    Q, _ = np.linalg.qr(rng.standard_normal((n, n))); lam = rng.uniform(0.5, 2, n)
    from conftest import DenseMetricSpace
    D2 = DenseMetricSpace((Q*lam) @ Q.T); G2 = D2._gram
    base = LinearOperator.from_matrix(EuclideanSpace(n), EuclideanSpace(n), M, form="components")
    lifted = LinearOperator.from_formal_adjoint(D, D2, base)
    y = rng.standard_normal(n)
    # adjoint in new metrics: G_X^-1 M^T G_Y
    ok(f"n={n} from_formal_adjoint adjoint == G_X^-1 M^T G_Y", np.allclose(lifted.adjoint(y), np.linalg.solve(G, M.T @ (G2 @ y))))
    check_operator(lifted, rng=rng)
    # MassWeightedSpace over euclid with dense mass; from_formal_adjoint coordinate-free route
    E = EuclideanSpace(n)
    Mass = LinearOperator.from_matrix(E, E, G, form="components", traits=Traits.POSITIVE_DEFINITE)
    MW = MassWeightedSpace(E, Mass)
    try:
        LinearOperator.from_formal_adjoint(MW, EuclideanSpace(n), base)
        print("from_formal_adjoint with an equal-but-distinct EuclideanSpace: ok")
    except ValueError as e:
        print("BUG n=%d: from_formal_adjoint(MassWeightedSpace(E), EuclideanSpace(n) == E but not E, op) raises: %s" % (n, str(e)[:60]))
    print("   MW.shares_vectors_with(EuclideanSpace(n)) =", MW.shares_vectors_with(EuclideanSpace(n)), "; MW.shares_vectors_with(E) =", MW.shares_vectors_with(E))
    base_E = LinearOperator.from_matrix(E, E, M, form="components")
    lifted2 = LinearOperator.from_formal_adjoint(MW, E, base_E)
    ok(f"n={n} from_formal_adjoint (MassWeighted domain) adjoint == G^-1 M^T", np.allclose(lifted2.adjoint(y), np.linalg.solve(G, M.T @ y), atol=1e-6))
    # block operator over dense summands
    S = DirectSum([D, D2])
    B = BlockLinearOperator([[Ac, LinearOperator.from_callables(D2, D, lambda v: v, adjoint=lambda w: np.linalg.solve(G2, G @ w))], [LinearOperator.zero(D, codomain=D2), LinearOperator.from_matrix(D2, D2, M, form="components")]])
    check_operator(B, rng=rng)
    ok(f"n={n} _CoordinateDirectSum apply_gram blockwise", np.allclose(S.apply_gram(np.concatenate([x, y])), np.concatenate([G @ x, G2 @ y])))
    # Jacobi on dense metric: preconditioner self-adjoint in the space inner product
    Spd = M @ M.T + n*np.eye(n)
    Apd = LinearOperator.from_matrix(D, D, Spd, form="galerkin", traits=Traits.POSITIVE_DEFINITE)
    P = JacobiPreconditioner()(Apd); check_traits(P, rng=rng)
    ok(f"n={n} Jacobi inverse diagonal == 1/diag(Galerkin)", np.allclose(P(D.from_components(np.linalg.solve(G, np.eye(n)[0]))), np.eye(n)[0] / Spd[0, 0]))
    res = CGSolver(rtol=1e-10, preconditioner=JacobiPreconditioner())(Apd).solve(x)
    ok(f"n={n} PCG(Jacobi) on dense metric solves", np.allclose(Apd(res.solution), x))
    # Banded on a NON-self-adjoint operator claims SELF_ADJOINT
    Pb = BandedPreconditioner(1, form="components")(Ac)
    try:
        check_traits(Pb, rng=rng); print("Banded on non-self-adjoint: check_traits passed (unexpected)")
    except AssertionError as e:
        print(f"n={n} S §3: BandedPreconditioner claims SELF_ADJOINT on a non-self-adjoint operator -> check_traits: {str(e)[:70]}")
print("done")
