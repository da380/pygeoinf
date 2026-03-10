"""
Tests for DualMasterCostFunction in backus_gilbert.py.

Covers:
- value_and_subgradient consistency with separate _mapping / _subgradient calls.
- Phase 1 guardrail tests: lock in expected oracle-path efficiency before
  production changes (Phases 2-3 of the DLI optimization plan).
"""

import numpy as np
import pytest

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.convex_analysis import BallSupportFunction
from pygeoinf.backus_gilbert import DualMasterCostFunction

np.random.seed(42)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_cost():
    """Small DualMasterCostFunction for n_data=4, n_model=5, n_prop=1."""
    rng = np.random.default_rng(42)
    n_data = 4
    n_model = 5
    n_prop = 1

    data_space = EuclideanSpace(n_data)
    model_space = EuclideanSpace(n_model)
    prop_space = EuclideanSpace(n_prop)

    G_matrix = rng.standard_normal((n_data, n_model))
    T_matrix = rng.standard_normal((n_prop, n_model))
    G = LinearOperator.from_matrix(model_space, data_space, G_matrix)
    T = LinearOperator.from_matrix(model_space, prop_space, T_matrix)

    model_prior = BallSupportFunction(model_space, model_space.zero, 1.0)
    data_error = BallSupportFunction(data_space, data_space.zero, 0.5)

    d_tilde = rng.standard_normal(n_data)
    q = rng.standard_normal(n_prop)

    cost = DualMasterCostFunction(
        data_space, prop_space, model_space,
        G, T, model_prior, data_error, d_tilde, q
    )
    return cost, data_space


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_value_and_subgradient_consistency():
    """value_and_subgradient must agree with separate value / subgradient calls.

    For 5 random λ points the joint call should return identical (f, g) to
    calling cost(lam) and cost.subgradient(lam) independently.
    """
    cost, data_space = _make_cost()
    rng = np.random.default_rng(7)

    for _ in range(5):
        lam_comps = rng.standard_normal(data_space.dim)
        lam = data_space.from_components(lam_comps)

        f_direct = cost(lam)
        g_direct = cost.subgradient(lam)

        f_joint, g_joint = cost.value_and_subgradient(lam)

        np.testing.assert_allclose(
            f_joint,
            f_direct,
            rtol=1e-10,
            err_msg=f"value_and_subgradient value differs from direct call: "
                    f"f_joint={f_joint}, f_direct={f_direct}",
        )
        np.testing.assert_allclose(
            data_space.to_components(g_joint),
            data_space.to_components(g_direct),
            rtol=1e-10,
            err_msg="value_and_subgradient subgradient differs from direct call",
        )


# ---------------------------------------------------------------------------
# Phase 1 guardrail test doubles
# ---------------------------------------------------------------------------

class _CountingBallSupportFunction(BallSupportFunction):
    """BallSupportFunction that counts scalar (_mapping) evaluations separately
    from support_point accesses.

    This lets guardrail tests verify that ``value_and_subgradient`` stops
    calling the scalar support function once the support-point identity
    ``h_S(q) = ⟨q, x*(q)⟩`` is exploited (Phase 2/3 goal).
    """

    def __init__(self, *args, **kwargs):
        self.scalar_eval_count = 0
        super().__init__(*args, **kwargs)

    def _mapping(self, q):
        self.scalar_eval_count += 1
        return super()._mapping(q)


class _AdjointCountingLinearOperator(LinearOperator):
    """LinearOperator subclass that counts accesses to the ``.adjoint`` property.

    ``LinearOperator.adjoint`` constructs a new operator on every access when
    ``_adjoint_base`` is None.  After Phase 3 the adjoint will be cached at
    construction time so subsequent oracle calls incur zero additional property
    accesses.
    """

    def __init__(self, domain, codomain, fwd_fn, adj_fn):
        self.adjoint_access_count = 0
        super().__init__(domain, codomain, fwd_fn, adjoint_mapping=adj_fn)

    @property
    def adjoint(self):
        self.adjoint_access_count += 1
        return super().adjoint


# ---------------------------------------------------------------------------
# Phase 1 guardrail tests (must FAIL against current production code)
# ---------------------------------------------------------------------------

def test_no_redundant_scalar_eval_when_support_point_exists():
    """GUARDRAIL: value_and_subgradient must NOT call the scalar support
    function when a support point is available.

    For any support function h_S the identity h_S(q) = ⟨q, x*(q)⟩ holds at
    every support point x*(q) ∈ ∂h_S(q).  Once x*(q) is computed, the scalar
    value is already known, so calling self._model_prior_support(hilbert_residual)
    and self._data_error_support(neg_lam) inside value_and_subgradient is
    redundant.

    This test currently FAILS against production code because value_and_subgradient
    makes a separate scalar evaluation after fetching the support point (see the
    ``term2 = self._model_prior_support(hilbert_residual)`` block in backus_gilbert.py).
    """
    rng = np.random.default_rng(99)
    n_data, n_model, n_prop = 4, 5, 1

    data_space = EuclideanSpace(n_data)
    model_space = EuclideanSpace(n_model)
    prop_space = EuclideanSpace(n_prop)

    G_matrix = rng.standard_normal((n_data, n_model))
    T_matrix = rng.standard_normal((n_prop, n_model))
    G = LinearOperator.from_matrix(model_space, data_space, G_matrix)
    T = LinearOperator.from_matrix(model_space, prop_space, T_matrix)

    model_prior = _CountingBallSupportFunction(model_space, model_space.zero, 1.0)
    data_error = _CountingBallSupportFunction(data_space, data_space.zero, 0.5)

    d_tilde = rng.standard_normal(n_data)
    q_vec = rng.standard_normal(n_prop)
    cost = DualMasterCostFunction(
        data_space, prop_space, model_space,
        G, T, model_prior, data_error, d_tilde, q_vec,
    )

    # Reset after construction in case any incidental evaluation occurred
    model_prior.scalar_eval_count = 0
    data_error.scalar_eval_count = 0

    lam = rng.standard_normal(n_data)
    cost.value_and_subgradient(lam)

    total = model_prior.scalar_eval_count + data_error.scalar_eval_count
    assert total == 0, (
        f"Expected 0 scalar support evaluations when support point is available "
        f"(use h_S(q) = <q, x*(q)> identity instead), but got {total} "
        f"(model_prior={model_prior.scalar_eval_count}, "
        f"data_error={data_error.scalar_eval_count})."
    )


def test_no_repeated_adjoint_fetch_across_oracle_calls():
    """GUARDRAIL: repeated value_and_subgradient calls must NOT re-fetch G.adjoint.

    ``LinearOperator.adjoint`` allocates a new operator object each time it is
    accessed (when ``_adjoint_base`` is None).  After Phase 3 the adjoint will
    be cached once in ``DualMasterCostFunction.__init__``, so subsequent oracle
    calls incur zero additional property accesses.

    This test currently FAILS against production code because value_and_subgradient
    calls ``self._G.adjoint(lam)`` on every invocation, re-fetching the property
    each time.
    """
    rng = np.random.default_rng(17)
    n_data, n_model, n_prop = 4, 5, 1

    data_space = EuclideanSpace(n_data)
    model_space = EuclideanSpace(n_model)
    prop_space = EuclideanSpace(n_prop)

    G_matrix = rng.standard_normal((n_data, n_model))
    T_matrix = rng.standard_normal((n_prop, n_model))

    def _fwd(x):
        return G_matrix @ x

    def _adj(y):
        return G_matrix.T @ y

    G_tracked = _AdjointCountingLinearOperator(model_space, data_space, _fwd, _adj)
    T = LinearOperator.from_matrix(model_space, prop_space, T_matrix)

    model_prior = BallSupportFunction(model_space, model_space.zero, 1.0)
    data_error = BallSupportFunction(data_space, data_space.zero, 0.5)

    d_tilde = rng.standard_normal(n_data)
    q_vec = rng.standard_normal(n_prop)
    cost = DualMasterCostFunction(
        data_space, prop_space, model_space,
        G_tracked, T, model_prior, data_error, d_tilde, q_vec,
    )

    # Reset after construction: only post-init accesses count for the oracle test
    G_tracked.adjoint_access_count = 0

    n_calls = 3
    lam = rng.standard_normal(n_data)
    for _ in range(n_calls):
        cost.value_and_subgradient(lam)

    assert G_tracked.adjoint_access_count == 0, (
        f"Expected 0 adjoint accesses after init (adjoint should be cached once "
        f"at construction time), but .adjoint was accessed "
        f"{G_tracked.adjoint_access_count} times across {n_calls} oracle calls."
    )
