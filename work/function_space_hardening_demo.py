"""Function-space hardening demo (theory figures).

Reproduces theory figures for the weakened-ellipsoid credible set
$\\theta \\in \\{0.2, 0.5, 0.8\\}$ on the basis-free Lebesgue space with
covariance $C = (-\\Delta)^{-1}$ on $[0, 1]$ (Dirichlet BCs).

For each $\\theta$ the script plots the calibrated radius $r_p(\\theta)$
as a function of confidence $p$ for the weakened weighted chi-square
distribution $\\sum_j \\lambda_j^{1 - \\theta} Z_j^2$.

Also benchmarks the Imhof and Wood–Saddlepoint quantile backends on
InverseLaplacian spectra of size {50, 500, 5000}.

Outputs go to ``pygeoinf/work/figures/function_space_hardening/``.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pygeoinf.quadratic_form_quantile import weighted_chi2_quantile


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def laplacian_dirichlet_spectrum(num_modes: int) -> np.ndarray:
    """Analytic eigenvalues of $(-\\Delta)^{-1}$ on $[0,1]$ with Dirichlet BCs."""
    j = np.arange(1, num_modes + 1, dtype=float)
    return 1.0 / (j * np.pi) ** 2


def figures_dir() -> Path:
    here = Path(__file__).resolve().parent
    out = here / "figures" / "function_space_hardening"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Theory figure: radius vs confidence for varying theta
# ---------------------------------------------------------------------------


def figure_radius_vs_confidence(spectrum_size: int = 500) -> Path:
    """Plot $r_p(\\theta)$ for $\\theta \\in \\{0.0, 0.2, 0.5, 0.8\\}$."""
    eigenvalues = laplacian_dirichlet_spectrum(spectrum_size)
    confidences = np.linspace(0.5, 0.99, 25)
    thetas = [0.0, 0.2, 0.5, 0.8]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for theta in thetas:
        weights = eigenvalues ** (1.0 - theta)
        radii = np.array(
            [
                np.sqrt(
                    max(
                        weighted_chi2_quantile(weights, float(p), method="imhof"),
                        0.0,
                    )
                )
                for p in confidences
            ]
        )
        ax.plot(confidences, radii, marker="o", ms=3, label=fr"$\theta = {theta}$")

    ax.set_xlabel("confidence $p$")
    ax.set_ylabel(r"radius $r_p(\theta)$")
    ax.set_title(
        r"Weakened-ellipsoid radius for $C = (-\Delta)^{-1}$ on $[0,1]$"
        f" (N = {spectrum_size})"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = figures_dir() / "radius_vs_confidence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Backend benchmark: Imhof vs WS
# ---------------------------------------------------------------------------


def benchmark_backends() -> dict:
    """Time Imhof and Wood–Saddlepoint quantile on Laplacian spectra."""
    sizes = (50, 500, 5000)
    probability = 0.9
    results: dict[str, dict[int, float]] = {"imhof": {}, "ws": {}}

    print(f"{'N':>6} | {'imhof [s]':>10} | {'ws [s]':>10} | "
          f"{'imhof q':>10} | {'ws q':>10}")
    print("-" * 60)
    for n in sizes:
        weights = laplacian_dirichlet_spectrum(n)

        t0 = time.perf_counter()
        q_imhof = weighted_chi2_quantile(weights, probability, method="imhof")
        t_imhof = time.perf_counter() - t0

        t0 = time.perf_counter()
        q_ws = weighted_chi2_quantile(weights, probability, method="ws")
        t_ws = time.perf_counter() - t0

        results["imhof"][n] = t_imhof
        results["ws"][n] = t_ws
        print(
            f"{n:>6d} | {t_imhof:>10.4f} | {t_ws:>10.4f} | "
            f"{q_imhof:>10.5f} | {q_ws:>10.5f}"
        )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def benchmark_all_methods(n_modes: int = 200, n_mc_ref: int = 500_000) -> None:
    """Precision and speed of all four quantile methods.

    Uses Monte Carlo with ``n_mc_ref`` samples as the ground truth.
    Compares Imhof, Welch–Satterthwaite, and Lugannani–Rice saddlepoint
    across different tail probabilities and spectrum shapes (theta).

    Args:
        n_modes: Number of eigenvalues in the Laplacian spectrum.
        n_mc_ref: MC reference sample count.
    """
    import itertools

    probs  = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    thetas = [0.0, 0.2, 0.5, 0.8]
    methods = ["imhof", "ws", "saddlepoint"]

    base_spectrum = laplacian_dirichlet_spectrum(n_modes)

    rng_ref = np.random.default_rng(0)

    # ── Pre-compute MC reference quantiles (one RNG draw per theta) ──────────
    print(f"\nComputing MC reference ({n_mc_ref:,} samples, N={n_modes}) …", flush=True)
    ref: dict[float, dict[float, float]] = {}
    for theta in thetas:
        weights = base_spectrum ** (1.0 - theta)
        from pygeoinf.quadratic_form_quantile import weighted_chi2_quantile
        ref[theta] = {
            p: weighted_chi2_quantile(weights, p, method="mc",
                                      n_samples=n_mc_ref, rng=rng_ref)
            for p in probs
        }

    # ── Time each approximate method at each (theta, p) ─────────────────────
    n_repeats = 5   # average over this many calls for timing

    results: dict = {}   # [method][theta][p] = (q, rel_err, t_sec)
    for method in methods:
        results[method] = {}
        for theta in thetas:
            weights = base_spectrum ** (1.0 - theta)
            results[method][theta] = {}
            for p in probs:
                times = []
                for _ in range(n_repeats):
                    t0 = time.perf_counter()
                    q = weighted_chi2_quantile(weights, p, method=method)
                    times.append(time.perf_counter() - t0)
                rel_err = abs(q - ref[theta][p]) / max(abs(ref[theta][p]), 1e-300)
                results[method][theta][p] = (q, rel_err, np.mean(times))

    # ── Print precision table (relative error vs MC) ─────────────────────────
    print(f"\n{'─'*90}")
    print(f"  Precision: relative error vs MC reference  (N={n_modes}, {n_mc_ref:,} samples)")
    print(f"{'─'*90}")
    hdr = f"  {'theta':>5}  {'p':>6}  " + "  ".join(f"{'|'+m:>14}" for m in methods)
    print(hdr)
    print(f"{'─'*90}")
    for theta in thetas:
        for p in probs:
            row = f"  {theta:>5.1f}  {p:>6.3f}  "
            for method in methods:
                _, rel_err, _ = results[method][theta][p]
                row += f"  {rel_err:>14.2e}"
            print(row)
        print()

    # ── Print speed table ─────────────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print(f"  Speed: mean call time [ms]  (N={n_modes}, averaged over {n_repeats} repeats)")
    print(f"{'─'*90}")
    print(hdr)
    print(f"{'─'*90}")
    for theta in thetas:
        for p in probs:
            row = f"  {theta:>5.1f}  {p:>6.3f}  "
            for method in methods:
                _, _, t_sec = results[method][theta][p]
                row += f"  {t_sec*1000:>14.3f}"
            print(row)
        print()

    # ── Figure: relative error heatmap ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5),
                              sharey=True)
    import matplotlib.colors as mcolors

    vmin, vmax = -6, 0    # log10 error range
    cmap = "RdYlGn_r"

    for ax, method in zip(axes, methods):
        err_grid = np.array([
            [np.log10(max(results[method][th][p][1], 1e-16))
             for p in probs]
            for th in thetas
        ])
        im = ax.imshow(err_grid, aspect="auto", vmin=vmin, vmax=vmax,
                       cmap=cmap, origin="upper")
        ax.set_xticks(range(len(probs)))
        ax.set_xticklabels([str(p) for p in probs], rotation=45, ha="right")
        ax.set_title(method, fontsize=12, fontweight="bold")
        if ax is axes[0]:
            ax.set_yticks(range(len(thetas)))
            ax.set_yticklabels([f"θ={th}" for th in thetas])
        ax.set_xlabel("p")

    fig.colorbar(im, ax=axes[-1], label="log₁₀ |relative error|")
    fig.suptitle(
        f"Relative error vs MC reference  (N={n_modes},  {n_mc_ref:,} MC samples)\n"
        "Green = accurate, Red = inaccurate",
        fontsize=11,
    )
    plt.tight_layout()
    out = figures_dir() / "method_accuracy_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved accuracy heatmap to {out}")


def main() -> None:
    print("=== Function-space hardening demo ===")
    out = figure_radius_vs_confidence()
    print(f"Saved theory figure to {out}")

    print("\n=== Imhof vs Wood–Saddlepoint benchmark ===")
    benchmark_backends()

    print("\n=== All-method precision & speed benchmark ===")
    benchmark_all_methods()


if __name__ == "__main__":
    main()
