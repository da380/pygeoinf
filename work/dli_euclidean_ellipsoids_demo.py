"""Simple DLI example on Euclidean spaces using ellipsoids only.

Model space:    R^3
Data space:     R^2
Property space: R^1

This script builds:
- A model-prior ellipsoid B in R^3
- A data-error ellipsoid V in R^2
- A linear forward map G: R^3 -> R^2
- A 1D property map T: R^3 -> R^1

It then computes deterministic admissible bounds on the property using
PrimalKKTSolver and saves one figure with three panels:
1) prior ellipsoid + admissible samples in model space,
2) shifted data ellipsoid d_tilde + V in data space,
3) inferred property interval in R^1.

Run:
    conda activate inferences3
    cd pygeoinf
    python work/dli_euclidean_ellipsoids_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf.convex_analysis import EllipsoidSupportFunction
from pygeoinf.convex_optimisation import PrimalKKTSolver
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator


def _diag_spd_ops(
    space: EuclideanSpace,
    diag: np.ndarray,
) -> tuple[LinearOperator, LinearOperator, LinearOperator]:
    """Build diagonal SPD metric operators A, A^{-1}, and A^{-1/2}."""
    a = LinearOperator.self_adjoint_from_matrix(space, np.diag(diag))
    a_inv = LinearOperator.self_adjoint_from_matrix(space, np.diag(1.0 / diag))
    a_inv_sqrt = LinearOperator.self_adjoint_from_matrix(
        space,
        np.diag(1.0 / np.sqrt(diag)),
    )
    return a, a_inv, a_inv_sqrt


def _sample_unit_ball_3d(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample points uniformly from the 3D unit ball."""
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    radii = rng.random(n) ** (1.0 / 3.0)
    return x * radii[:, None]


def _ellipsoid_boundary_2d(
    center: np.ndarray,
    radius: float,
    a_inv_sqrt_diag: np.ndarray,
    n: int = 300,
) -> np.ndarray:
    """Return boundary points of a 2D ellipsoid with center and metric."""
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.vstack((np.cos(theta), np.sin(theta))).T
    transform = np.diag(a_inv_sqrt_diag)
    return center + radius * (circle @ transform.T)


def _ellipsoid_surface_3d(
    center: np.ndarray,
    radius: float,
    a_inv_sqrt_diag: np.ndarray,
    nu: int = 48,
    nv: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return parameterized ellipsoid surface coordinates in 3D."""
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
    transform = np.diag(a_inv_sqrt_diag)
    pts = center + radius * np.einsum("...j,ij->...i", sphere, transform)
    return pts[..., 0], pts[..., 1], pts[..., 2]


def main() -> dict[str, float]:
    rng = np.random.default_rng(7)

    # Spaces
    model_space = EuclideanSpace(3)
    data_space = EuclideanSpace(2)
    property_space = EuclideanSpace(1)

    # Linear maps G: R^3 -> R^2 and T: R^3 -> R^1
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

    # Prior ellipsoid B = {u : (u-u0)^T A_B (u-u0) <= eta^2}
    u0 = np.array([0.4, -0.2, 0.3], dtype=float)
    eta = 1.2
    a_b_diag = np.array([9.0, 4.0, 16.0], dtype=float)
    a_b, a_b_inv, a_b_inv_sqrt = _diag_spd_ops(model_space, a_b_diag)
    b_support = EllipsoidSupportFunction(
        model_space,
        center=u0,
        radius=eta,
        shape_operator=a_b,
        inverse_operator=a_b_inv,
        inverse_sqrt_operator=a_b_inv_sqrt,
    )

    # Synthetic feasible truth and observed data
    u_true = np.array([0.55, -0.1, 0.15], dtype=float)

    # Data ellipsoid V = {e : e^T A_V e <= r^2} (centered at zero)
    r = 0.7
    a_v_diag = np.array([25.0, 16.0], dtype=float)
    a_v, a_v_inv, a_v_inv_sqrt = _diag_spd_ops(data_space, a_v_diag)
    v_support = EllipsoidSupportFunction(
        data_space,
        center=data_space.zero,
        radius=r,
        shape_operator=a_v,
        inverse_operator=a_v_inv,
        inverse_sqrt_operator=a_v_inv_sqrt,
    )

    e_true = np.array([0.05, -0.04], dtype=float)
    d_tilde = g_op(u_true) + e_true

    # DLI solve for property bounds in R^1
    solver = PrimalKKTSolver(b_support, v_support, g_op, d_tilde)
    q = property_space.basis_vector(0)
    c = t_op.adjoint(q)

    res_upper = solver.solve(c)
    res_lower = solver.solve(-c)

    upper = float(t_op(res_upper.m)[0])
    lower = float(t_op(res_lower.m)[0])
    true_prop = float(t_op(u_true)[0])

    # Sample prior points for visualization and identify admissible samples
    n_samples = 9000
    z = _sample_unit_ball_3d(n_samples, rng)
    a_b_inv_sqrt_diag = 1.0 / np.sqrt(a_b_diag)
    prior_samples = u0 + eta * (z * a_b_inv_sqrt_diag[None, :])

    data_residuals = (prior_samples @ g_matrix.T) - d_tilde[None, :]
    data_quad = np.sum(
        (data_residuals @ np.diag(a_v_diag)) * data_residuals,
        axis=1,
    )
    admissible_mask = data_quad <= r**2
    admissible_samples = prior_samples[admissible_mask]
    admissible_data = admissible_samples @ g_matrix.T

    # Plot three panels in one figure
    fig = plt.figure(figsize=(14.0, 4.8))

    # Panel 1: model-space prior ellipsoid and admissible samples
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.1, 1.0, 1.0],
        wspace=0.16,
    )

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    sx, sy, sz = _ellipsoid_surface_3d(u0, eta, a_b_inv_sqrt_diag)
    ax1.plot_surface(sx, sy, sz, alpha=0.18, color="tab:blue", linewidth=0)
    if admissible_samples.shape[0] > 0:
        ax1.scatter(
            admissible_samples[:, 0],
            admissible_samples[:, 1],
            admissible_samples[:, 2],
            s=2,
            alpha=0.35,
            color="tab:green",
            label="admissible samples",
        )
    ax1.scatter(*u_true, color="black", s=50, marker="*", label="u_true")
    ax1.scatter(*res_upper.m, color="tab:red", s=32, label="u_max")
    ax1.scatter(*res_lower.m, color="tab:orange", s=32, label="u_min")
    ax1.set_title("Model space R^3: prior ellipsoid B")
    ax1.set_xlabel("u1")
    ax1.set_ylabel("u2")
    ax1.set_zlabel("u3")
    ax1.legend(loc="upper left", fontsize=8)

    # Panel 2: data-space shifted ellipsoid d_tilde + V and forward images
    ax2 = fig.add_subplot(gs[0, 1])
    y_boundary = _ellipsoid_boundary_2d(
        center=d_tilde,
        radius=r,
        a_inv_sqrt_diag=(1.0 / np.sqrt(a_v_diag)),
    )
    ax2.plot(
        y_boundary[:, 0],
        y_boundary[:, 1],
        color="tab:purple",
        lw=2,
        label=r"$\tilde d + V$",
    )

    # Forward map of prior/admissible samples
    g_prior = prior_samples @ g_matrix.T
    ax2.scatter(
        g_prior[:, 0],
        g_prior[:, 1],
        s=2,
        alpha=0.08,
        color="tab:blue",
        label="G(B) samples",
    )
    if admissible_data.shape[0] > 0:
        ax2.scatter(
            admissible_data[:, 0],
            admissible_data[:, 1],
            s=4,
            alpha=0.45,
            color="tab:green",
            label="consistent",
        )

    y_upper = g_op(res_upper.m)
    y_lower = g_op(res_lower.m)
    ax2.scatter(*d_tilde, color="black", s=40, marker="x", label=r"$\tilde d$")
    ax2.scatter(*y_upper, color="tab:red", s=28, label="G(u_max)")
    ax2.scatter(*y_lower, color="tab:orange", s=28, label="G(u_min)")
    ax2.set_title("Data space R^2: shifted ellipsoid")
    ax2.set_xlabel("y1")
    ax2.set_ylabel("y2")
    ax2.set_aspect("equal", adjustable="box")
    ax2.legend(loc="best", fontsize=8)

    # Panel 3: property interval in R^1
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hlines(
        0.0,
        lower,
        upper,
        color="tab:red",
        lw=8,
        alpha=0.7,
        label="DLI admissible interval",
    )
    ax3.scatter([lower, upper], [0.0, 0.0], color="tab:red", s=44)
    ax3.scatter(
        [true_prop],
        [0.0],
        color="black",
        marker="*",
        s=70,
        label="true property",
    )
    ax3.set_title("Property space R^1: inferred bounds")
    ax3.set_xlabel("property value")
    ax3.set_yticks([])
    pad = 0.12 * max(1e-6, upper - lower)
    ax3.set_xlim(lower - pad, upper + pad)
    ax3.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Deterministic Linear Inference with Ellipsoids "
        "(R^3 -> R^2 -> R^1)",
        y=1.02,
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
    out_path = output_dir / "dli_euclidean_ellipsoids.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {out_path}")
    print(f"Property interval: [{lower:.6f}, {upper:.6f}]")
    print(f"True property: {true_prop:.6f}")
    print(
        "Admissible sample count: "
        f"{admissible_samples.shape[0]} / {n_samples}"
    )

    return {
        "lower": lower,
        "upper": upper,
        "true_property": true_prop,
        "admissible_count": float(admissible_samples.shape[0]),
    }


if __name__ == "__main__":
    main()
