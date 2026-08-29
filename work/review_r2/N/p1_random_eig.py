"""random_eig at rank 50 on 3000-dim operators: Euclidean, dense metric, sphere."""
from common import *
import cProfile, pstats, io
from pygeoinf2.tests.conftest import make_dense_metric_space
from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.traits import Traits
from pygeoinf2.numerics.randomised import random_eig, random_range
rng = np.random.default_rng(1)
n, k = 3000, 50

def spd(n, decay=0.9):
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = decay ** np.arange(n) * 100 + 1e-3
    return V @ (lam[:, None] * V.T), lam

def proto_random_eig(A_c_mul, space, k, oversampling=10, power=1):
    """Component-space prototype: Cholesky-QR twice on component arrays."""
    G = space.apply_gram
    def orth(Y):  # Y: n x m components -> G-orthonormal columns
        for _ in range(2):
            GY = np.stack([G(y) for y in Y.T], axis=1)
            C = Y.T @ GY
            R = np.linalg.cholesky(C)
            Y = np.linalg.solve(R, Y.T).T
        return Y
    m = k + oversampling
    Om = np.stack([space.white_noise_components(rng=rng) for _ in range(m)], axis=1)
    Y = A_c_mul(Om)
    Q = orth(Y)
    for _ in range(power):
        Z = orth(A_c_mul(Q))   # A self-adjoint: A* = A
        Q = orth(A_c_mul(Z))
    AQ = A_c_mul(Q)
    GAQ = np.stack([G(y) for y in AQ.T], axis=1)
    T = Q.T @ GAQ; T = 0.5*(T+T.T)
    vals, vecs = np.linalg.eigh(T)
    order = np.argsort(np.abs(vals))[::-1][:k]
    U = Q @ vecs[:, order]
    return vals[order], U

def run_case(name, space, op, A_c_mul, lam_true, block_apply):
    cnt = Counts(); opc = counting_operator(op, cnt)
    cnt.wrap_space(space)
    # interleaved timing
    for rep in range(2):
        t = time.perf_counter(); eig = random_eig(opc, rank=k, rng=rng); t_v2 = time.perf_counter() - t
        t = time.perf_counter(); vals, U = proto_random_eig(A_c_mul, space, k); t_p = time.perf_counter() - t
        print(f"{name} rep{rep}: v2 {t_v2:.3f}s  proto {t_p:.3f}s  ratio {t_v2/t_p:.1f}")
    print("  counts v2:", cnt)
    print("  v2 err top10:", np.max(np.abs(eig.eigenvalues[:10] - lam_true[:10])/lam_true[:10]),
          " proto err top10:", np.max(np.abs(vals[:10] - lam_true[:10])/lam_true[:10]))
    pr = cProfile.Profile(); pr.enable(); random_eig(opc, rank=k, rng=rng); pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(14); 
    print("\n".join(l for l in s.getvalue().splitlines() if "pygeoinf2" in l or "ncalls" in l)[:3000])

# (a) Euclidean
M, lam = spd(n)
spE = EuclideanSpace(n)
opE = LinearOperator.from_matrix(spE, spE, M, form="galerkin", traits=Traits.SELF_ADJOINT|Traits.POSITIVE_SEMIDEFINITE)
#run_case("euclid3000", spE, opE, lambda X: M @ X, lam, True)
# (b) dense metric, same spectrum: Galerkin S = G A_c with A_c = L^-T diag L^T? use S = G^{1/2}... simpler: generalized problem
from pygeoinf2.tests.conftest import DenseMetricSpace
root = np.eye(n) + (0.3/np.sqrt(n)) * np.tril(rng.standard_normal((n, n)), -1); spD = DenseMetricSpace(root @ root.T); G = spD.gram_matrix(); Lc = np.linalg.cholesky(G)
# A_c = L^-T V diag V^T L^T  -> G A_c = L V diag V^T L^T symmetric, eigenvalues lam in the space's metric
V, _ = np.linalg.qr(rng.standard_normal((n, n)))
Ac = np.linalg.solve(Lc.T, V @ (lam[:, None] * V.T) @ Lc.T)
S = G @ Ac
opD = LinearOperator.from_matrix(spD, spD, Ac, form="components", traits=Traits.SELF_ADJOINT|Traits.POSITIVE_SEMIDEFINITE)
run_case("densemetric3000", spD, opD, lambda X: Ac @ X, lam, True)
