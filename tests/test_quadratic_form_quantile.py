"""Tests for the weighted chi-square CDF/quantile module."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2

from pygeoinf.quadratic_form_quantile import (
    weighted_chi2_cdf,
    weighted_chi2_quantile,
)


@pytest.mark.parametrize("k", [1, 5, 50])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 0.99])
def test_equal_weights_match_chi2_imhof(k, p):
    """Equal weights of length k recover the standard chi-square quantile."""
    weights = np.ones(k)
    expected = chi2.ppf(p, df=k)
    got = weighted_chi2_quantile(weights, p, method="imhof", rtol=1e-9)
    assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("k", [1, 5, 50])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 0.99])
def test_equal_weights_match_chi2_ws(k, p):
    """Welch--Satterthwaite is exact for equal weights."""
    weights = np.ones(k)
    expected = chi2.ppf(p, df=k)
    got = weighted_chi2_quantile(weights, p, method="ws")
    assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("scale", [0.25, 1.0, 4.0])
def test_single_weight_is_scaled_chi2(scale):
    """A single weight w gives Q = w * chi2_1."""
    weights = np.array([scale])
    for p in (0.25, 0.5, 0.9):
        expected = scale * chi2.ppf(p, df=1)
        got = weighted_chi2_quantile(weights, p, method="imhof")
        assert_allclose(got, expected, rtol=1e-6)


def test_imhof_two_weights_matches_mc():
    """Imhof and Monte Carlo agree for a two-weight problem."""
    weights = np.array([4.0, 9.0])
    p = 0.85
    r_imhof = weighted_chi2_quantile(weights, p, method="imhof", rtol=1e-9)
    rng = np.random.default_rng(0)
    r_mc = weighted_chi2_quantile(
        weights, p, method="mc", n_samples=200_000, rng=rng
    )
    # Statistical uncertainty for MC quantile is roughly sigma_p / sqrt(N).
    # Allow 2% tolerance.
    assert_allclose(r_imhof, r_mc, rtol=2e-2)


def test_imhof_anisotropic_matches_mc():
    """Highly anisotropic weights: Imhof and MC still agree."""
    weights = np.concatenate([[100.0], np.ones(20)])
    p = 0.9
    r_imhof = weighted_chi2_quantile(weights, p, method="imhof", rtol=1e-9)
    rng = np.random.default_rng(1)
    r_mc = weighted_chi2_quantile(
        weights, p, method="mc", n_samples=200_000, rng=rng
    )
    assert_allclose(r_imhof, r_mc, rtol=3e-2)


def test_ws_brackets_imhof_anisotropic():
    """WS approximation should be close to but not exactly equal to Imhof."""
    weights = np.concatenate([[10.0], np.ones(30)])
    p = 0.9
    r_imhof = weighted_chi2_quantile(weights, p, method="imhof", rtol=1e-9)
    r_ws = weighted_chi2_quantile(weights, p, method="ws")
    # WS is moment-matched; expect within ~10% for moderate anisotropy.
    assert abs(r_ws - r_imhof) / r_imhof < 0.1


def test_imhof_decaying_weights():
    """Imhof handles a long sequence of decaying weights (function-space-like)."""
    n = 200
    weights = 1.0 / np.arange(1, n + 1) ** 2
    p = 0.9
    r_imhof = weighted_chi2_quantile(weights, p, method="imhof", rtol=1e-9)
    rng = np.random.default_rng(2)
    r_mc = weighted_chi2_quantile(
        weights, p, method="mc", n_samples=200_000, rng=rng
    )
    assert_allclose(r_imhof, r_mc, rtol=2e-2)


def test_saddlepoint_deep_tail():
    """Saddlepoint approximation tracks Imhof in the deep tail."""
    weights = np.array([1.0, 0.5, 0.25, 0.125])
    p = 0.999
    r_imhof = weighted_chi2_quantile(weights, p, method="imhof", rtol=1e-10)
    r_sp = weighted_chi2_quantile(weights, p, method="saddlepoint", rtol=1e-10)
    # Lugannani-Rice has O(1/n) relative error; 1% is the realistic budget
    # for a small heterogeneous spectrum.
    assert_allclose(r_sp, r_imhof, rtol=1e-2)


def test_cdf_increases_with_t():
    """Sanity check: CDF is monotonically non-decreasing in t."""
    weights = np.array([3.0, 1.0, 0.5])
    t_vals = np.linspace(0.1, 30.0, 50)
    cdf_vals = weighted_chi2_cdf(weights, t_vals, method="imhof")
    diffs = np.diff(cdf_vals)
    assert np.all(diffs >= -1e-10)


def test_cdf_quantile_roundtrip_imhof():
    """For weights w, F_imhof(quantile(p)) == p within tolerance."""
    weights = np.array([2.0, 1.0, 0.5, 0.25])
    for p in (0.3, 0.7, 0.95):
        r = weighted_chi2_quantile(weights, p, method="imhof", rtol=1e-10)
        p_back = weighted_chi2_cdf(weights, r, method="imhof", rtol=1e-10)
        assert_allclose(p_back, p, atol=1e-5)


def test_zero_weights_collapses_to_point_mass():
    """All-zero or empty weights produce a point mass at 0."""
    assert weighted_chi2_cdf(np.array([]), 1.0, method="imhof") == 1.0
    assert weighted_chi2_cdf(np.array([0.0, 0.0]), 1.0, method="imhof") == 1.0
    assert weighted_chi2_quantile(np.array([]), 0.5, method="imhof") == 0.0


def test_cdf_at_zero_is_zero():
    weights = np.array([1.0, 2.0])
    assert weighted_chi2_cdf(weights, 0.0, method="imhof") == 0.0
    assert weighted_chi2_cdf(weights, -1.0, method="imhof") == 0.0


@pytest.mark.parametrize("p", [-0.1, 0.0, 1.0, 1.1])
def test_invalid_probability(p):
    with pytest.raises(ValueError, match="probability"):
        weighted_chi2_quantile(np.ones(3), p)


def test_negative_weights_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        weighted_chi2_quantile(np.array([1.0, -1.0]), 0.5)


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        weighted_chi2_quantile(np.ones(2), 0.5, method="bogus")


# ---------------------------------------------------------------------------
# Auto-mode tests
# ---------------------------------------------------------------------------


def test_auto_select_method_isotropic_uses_saddlepoint():
    """Isotropic spectrum (nu_eff = k) uses saddlepoint for tol=0.01."""
    from pygeoinf.quadratic_form_quantile import _auto_select_method

    # 100 equal weights → nu_eff = 100 > 0.3/0.01 = 30
    w = np.ones(100)
    assert _auto_select_method(w, tol=1e-2) == "saddlepoint"


def test_auto_select_method_anisotropic_uses_imhof():
    """Anisotropic spectrum (nu_eff ≈ 2.5) uses Imhof for tol=0.01."""
    from pygeoinf.quadratic_form_quantile import _auto_select_method

    # 1/(j*pi)^2 spectrum → nu_eff ≈ 2.5
    w = 1.0 / (np.arange(1, 201) * np.pi) ** 2
    assert _auto_select_method(w, tol=1e-2) == "imhof"


def test_auto_select_method_loose_tol_uses_saddlepoint():
    """Even mildly anisotropic spectrum uses saddlepoint for loose tol."""
    from pygeoinf.quadratic_form_quantile import _auto_select_method

    # nu_eff ≈ 2.5; with tol=0.5 the saddlepoint threshold 0.3/0.5=0.6 is satisfied
    w = 1.0 / (np.arange(1, 201) * np.pi) ** 2
    assert _auto_select_method(w, tol=0.5) == "saddlepoint"


@pytest.mark.parametrize("k", [1, 10, 100])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.95])
def test_auto_mode_equal_weights_matches_chi2(k, p):
    """Auto mode on equal weights recovers the chi-square quantile."""
    weights = np.ones(k)
    expected = chi2.ppf(p, df=k)
    got = weighted_chi2_quantile(weights, p)  # default method="auto"
    assert_allclose(got, expected, rtol=1e-4)


def test_auto_mode_default_is_auto():
    """Calling without method= uses auto mode (doesn't raise, returns float)."""
    w = np.array([2.0, 1.0, 0.5])
    result = weighted_chi2_quantile(w, 0.9)
    assert isinstance(result, float)
    assert result > 0.0


def test_auto_cdf_default_is_auto():
    """weighted_chi2_cdf without method= uses auto mode."""
    w = np.array([2.0, 1.0, 0.5])
    q = weighted_chi2_quantile(w, 0.9, method="imhof")
    p = weighted_chi2_cdf(w, q)  # default method="auto"
    assert_allclose(p, 0.9, atol=1e-4)


def test_auto_mode_anisotropic_agrees_with_imhof():
    """Auto mode on anisotropic spectrum (selects Imhof) matches explicit Imhof."""
    # nu_eff ≈ 2.5 → auto selects Imhof → results should be identical
    w = 1.0 / (np.arange(1, 51) * np.pi) ** 2
    for p in (0.5, 0.9, 0.99):
        q_auto = weighted_chi2_quantile(w, p)
        q_imhof = weighted_chi2_quantile(w, p, method="imhof")
        assert_allclose(q_auto, q_imhof, rtol=1e-12)


def test_auto_mode_tol_explicit_method_independence():
    """Passing tol= is harmless when an explicit method is specified."""
    w = np.array([3.0, 1.0, 0.5])
    q1 = weighted_chi2_quantile(w, 0.8, method="imhof", tol=1e-6)
    q2 = weighted_chi2_quantile(w, 0.8, method="imhof", tol=0.5)
    assert_allclose(q1, q2, rtol=1e-12)

