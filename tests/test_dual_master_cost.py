"""
Tests for DualMasterCostFunction in backus_gilbert.py.

Covers:
- value_and_subgradient consistency with separate _mapping / _subgradient calls.
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
