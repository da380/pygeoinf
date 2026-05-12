"""PLI analogue of the Euclidean DLI ellipsoid demo.

Spaces:
- Model space H = R^3
- Data space D = R^2
- Property space P = R^1

This script mirrors the DLI setup but in Bayesian form:
- Model prior:      u ~ N(u0, C_u) with C_u^{-1} = A_B
- Data-error prior: e ~ N(0,  C_e) with C_e^{-1} = A_V
- Data model:       d = G u + e

The DLI ellipsoids are recovered as Gaussian level sets:
    (u-u0)^T A_B (u-u0) <= eta^2
    (e-0)^T A_V (e-0) <= r^2

Plot (single image, 3 panels):
1) model-space prior measure (prior mean, prior samples, true model),
2) predictive data measure (predictive mean, predictive samples,
    observed datum),
3) property posterior measure p = T u | d (Gaussian density).

Run:
    conda activate inferences3
    cd pygeoinf
    python work/pli_euclidean_gaussian_analogue.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf.forward_problem import LinearForwardProblem
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_bayesian import LinearBayesianInversion
from pygeoinf.linear_operators import LinearOperator
from pygeoinf.linear_solvers import CholeskySolver


def _diag_operator(space: EuclideanSpace, diag: np.ndarray) -> LinearOperator:
    """Return a self-adjoint diagonal operator on a Euclidean space."""
    return LinearOperator.self_adjoint_from_matrix(space, np.diag(diag))


def _sample_unit_ball_3d(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform samples in the 3D unit ball."""
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    radii = rng.random(n) ** (1.0 / 3.0)
    return x * radii[:, None]


def _ellipsoid_boundary_2d(
    center: np.ndarray,
    covariance: np.ndarray,
    kappa: float,
    n: int = 300,
) -> np.ndarray:
    """Boundary of (x-c)^T C^{-1} (x-c) = kappa^2 for SPD 2x2 C."""
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.vstack((np.cos(theta), np.sin(theta))).T
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    return center + kappa * (circle @ transform.T)


def _ellipsoid_surface_3d(
    center: np.ndarray,
    cov_diag: np.ndarray,
    kappa: float,
    nu: int = 48,
    nv: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Surface of (x-c)^T C^{-1} (x-c) = kappa^2 for diagonal C."""
    u = np.linspace(0.0, 2.0 * np.pi, nu)
    v = np.linspace(0.0, np.pi, nv)
    uu, vv = np.meshgrid(u, v)

    sphere = np.stack(
        [
            np.cos(uu) * np.sin(vv),
            np.sin(uu) * np.sin(vv),
            np.cos(vv),
        ],
        axis=-1,
    )

    transform = np.diag(np.sqrt(cov_diag))
    pts = center + kappa * np.einsum("...j,ij->...i", sphere, transform)
    return pts[..., 0], pts[..., 1], pts[..., 2]


def main() -> dict[str, float]:
    # ------------------------------------------------------------------
    # Spaces and linear maps (same geometry as DLI example)
    # ------------------------------------------------------------------
    model_space = EuclideanSpace(3)
    data_space = EuclideanSpace(2)
    property_space = EuclideanSpace(1)

    g_matrix = np.array(
        [
            [1.0, 0.5, -0.2],
            [-0.3, 1.0, 0.8],
        ],
        dtype=float,
    )
    t_matrix = np.array([[0.2, 1.0, -0.4]], dtype=float)

    g_op = LinearOperator.from_matrix(model_space, data_space, g_matrix)
    t_op = LinearOperator.from_matrix(model_space, property_space, t_matrix)

    # ------------------------------------------------------------------
    # Gaussian measures chosen to match DLI ellipsoid metrics
    # ------------------------------------------------------------------
    u0 = np.array([0.4, -0.2, 0.3], dtype=float)
    a_b_diag = np.array([9.0, 4.0, 16.0], dtype=float)
    c_u_diag = 1.0 / a_b_diag

    a_v_diag = np.array([25.0, 16.0], dtype=float)
    c_e_diag = 1.0 / a_v_diag

    model_prior = GaussianMeasure.from_standard_deviations(
        model_space,
        np.sqrt(c_u_diag),
        expectation=u0,
    )

    data_error_measure = GaussianMeasure.from_standard_deviations(
        data_space,
        np.sqrt(c_e_diag),
        expectation=data_space.zero,
    )

    # ------------------------------------------------------------------
    # Synthetic observation and Bayesian update
    # ------------------------------------------------------------------
    u_true = np.array([0.55, -0.1, 0.15], dtype=float)
    e_true = np.array([0.05, -0.04], dtype=float)
    d_tilde = g_op(u_true) + e_true

    forward_problem = LinearForwardProblem(
        g_op,
        data_error_measure=data_error_measure,
    )
    inversion = LinearBayesianInversion(forward_problem, model_prior)

    solver = CholeskySolver(galerkin=True)
    model_posterior = inversion.model_posterior_measure(d_tilde, solver)
    property_posterior = model_posterior.affine_mapping(operator=t_op)

    # Prior measure in data space (predictive): y = G u + e
    predictive_data_measure = inversion.data_prior_measure

    # Posterior property stats
    prop_mean = float(property_posterior.expectation[0])
    prop_var = float(
        property_posterior.covariance.matrix(dense=True, galerkin=True)[0, 0]
    )
    prop_std = float(np.sqrt(max(prop_var, 0.0)))
    true_prop = float(t_op(u_true)[0])

    # ------------------------------------------------------------------
    # Sampling for visual intuition
    # ------------------------------------------------------------------
    model_prior_samples = np.asarray(model_prior.samples(2000))
    predictive_data_samples = np.asarray(predictive_data_measure.samples(2000))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14.0, 4.8))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.1, 1.0, 1.0],
        wspace=0.17,
    )

    # Panel 1: model prior measure in R^3
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.scatter(
        model_prior_samples[:, 0],
        model_prior_samples[:, 1],
        model_prior_samples[:, 2],
        s=3,
        alpha=0.10,
        color="tab:blue",
        label="prior samples",
    )

    ax1.scatter(*u0, color="tab:blue", s=40, marker="o", label="prior mean")
    ax1.scatter(*u_true, color="black", s=55, marker="*", label="u_true")

    ax1.set_title("Model space R^3: prior Gaussian")
    ax1.set_xlabel("u1")
    ax1.set_ylabel("u2")
    ax1.set_zlabel("u3")
    ax1.legend(loc="upper left", fontsize=8)

    # Panel 2: predictive data measure in R^2
    ax2 = fig.add_subplot(gs[0, 1])

    predictive_data_mean = np.asarray(predictive_data_measure.expectation)

    ax2.scatter(
        predictive_data_samples[:, 0],
        predictive_data_samples[:, 1],
        s=3,
        alpha=0.12,
        color="tab:green",
        label="predictive samples",
    )
    ax2.scatter(
        *predictive_data_mean,
        color="tab:green",
        s=40,
        marker="o",
        label="predictive mean",
    )
    ax2.scatter(*d_tilde, color="black", s=40, marker="x", label=r"$\tilde d$")

    ax2.set_title("Data space R^2: predictive data measure")
    ax2.set_xlabel("y1")
    ax2.set_ylabel("y2")
    ax2.set_aspect("equal", adjustable="box")
    ax2.legend(loc="best", fontsize=8)

    # Panel 3: property posterior in R^1
    ax3 = fig.add_subplot(gs[0, 2])

    x_min = min(true_prop, prop_mean - 4.0 * prop_std)
    x_max = max(true_prop, prop_mean + 4.0 * prop_std)
    x = np.linspace(x_min, x_max, 500)
    if prop_std > 0:
        pdf = (1.0 / (np.sqrt(2.0 * np.pi) * prop_std)) * np.exp(
            -0.5 * ((x - prop_mean) / prop_std) ** 2
        )
    else:
        pdf = np.zeros_like(x)
        pdf[np.argmin(np.abs(x - prop_mean))] = 1.0

    ax3.plot(x, pdf, color="tab:red", lw=2, label="posterior density")
    ax3.axvline(prop_mean, color="tab:red", ls="--", lw=1.8,
                label="posterior mean")
    ax3.axvline(
        prop_mean - prop_std,
        color="tab:red",
        ls=":",
        lw=1.5,
        label="mean ± 1 sigma",
    )
    ax3.axvline(prop_mean + prop_std, color="tab:red", ls=":", lw=1.5)
    ax3.axvline(
        true_prop,
        color="black",
        ls="-",
        lw=1.5,
        label="true property",
    )

    ax3.set_title("Property space R^1: posterior measure")
    ax3.set_xlabel("property value")
    ax3.set_ylabel("density")
    ax3.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Probabilistic Linear Inversion analogue "
        "(R^3 -> R^2 -> R^1)",
        y=0.96,
    )
    fig.subplots_adjust(
        left=0.03,
        right=0.995,
        bottom=0.10,
        top=0.84,
        wspace=0.17,
    )

    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "pli_euclidean_gaussian_analogue.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {out_path}")
    print(f"Posterior property mean: {prop_mean:.6f}")
    print(f"Posterior property std:  {prop_std:.6f}")
    print(f"True property:           {true_prop:.6f}")

    return {
        "posterior_property_mean": prop_mean,
        "posterior_property_std": prop_std,
        "true_property": true_prop,
    }


if __name__ == "__main__":
    main()
