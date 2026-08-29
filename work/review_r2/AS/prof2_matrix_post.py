from common import *
from scipy.linalg import cho_solve
from pygeoinf2 import LinearOperator, EuclideanSpace, Traits
n = 2000
rng = np.random.default_rng(0)
M = rng.standard_normal((n, n))
for name, Y in [("dense-metric", dense_space(n)), ("diag-metric", weighted_space(n)), ("euclid", EuclideanSpace(n))]:
    X = EuclideanSpace(n)
    A = LinearOperator.from_callables(X, Y, lambda c: Y.from_components(M @ c), adjoint=lambda y: M.T @ Y.apply_gram(Y.to_components(y)))
    G = Y._gram if name == "dense-metric" else (np.diag(Y.metric_values) if name=="diag-metric" else np.eye(n))
    def raw_columns():
        return np.column_stack([Y.to_components(A(X.basis_vector(j))) for j in range(n)])
    def post_loop():
        return np.column_stack([Y.apply_gram(M[:, j]) for j in range(n)])
    def post_vec():
        if name == "dense-metric": return G @ M
        if name == "diag-metric": return Y.metric_values[:, None] * M
        return M
    r = interleave({"columns(raw)": raw_columns, "post loop": post_loop, "post vectorised": post_vec, "matrix(galerkin)": lambda: A.matrix(form="galerkin", by="columns")}, repeats=3)
    print(name, {k: f"{v*1e3:.0f} ms" for k, v in r.items()})
    assert np.allclose(post_loop(), post_vec())
    # rows path, components form: solve_gram per column
    if name != "euclid":
        B = LinearOperator.from_callables(Y, X, lambda y: M @ Y.to_components(y), adjoint=lambda c: Y.from_components(Y.solve_gram(M.T @ c)))
        # B has codomain X (euclid) so solve_gram is trivial; use A with by="rows", form="components" instead
        def rows_raw():
            adj = A.adjoint
            return np.stack([X.apply_gram(X.to_components(adj(Y.basis_vector(i)))) for i in range(n)])
        Mg = G @ M
        def post_solve_loop():
            return np.column_stack([Y.solve_gram(Mg[:, j]) for j in range(n)])
        def post_solve_vec():
            if name == "dense-metric": return cho_solve((Y._chol, True), Mg)
            return Mg / Y.metric_values[:, None]
        r = interleave({"rows(raw)": rows_raw, "solve loop": post_solve_loop, "solve vectorised": post_solve_vec, "matrix(components,rows)": lambda: A.matrix(form="components", by="rows")}, repeats=2)
        print(name, {k: f"{v*1e3:.0f} ms" for k, v in r.items()})
        assert np.allclose(post_solve_loop(), post_solve_vec())
    # MatrixLinearOperator._in_form conversion
    Am = LinearOperator.from_matrix(X, Y, M, form="components")
    r = interleave({"_in_form(galerkin) loop": lambda: Am.matrix(form="galerkin"), "vectorised": post_vec}, repeats=3)
    print(name, "MatrixLinearOperator conversion", {k: f"{v*1e3:.0f} ms" for k, v in r.items()})
