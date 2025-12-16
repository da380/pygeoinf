"""
Lightweight checker to study positive-definiteness of the prior covariance
operator C_0 under different integration configurations.

This script constructs the model Lebesgue space and the prior covariance
operator (either inverse Laplacian or Bessel-Sobolev inverse) using the
same building blocks as `PLIExperiment.setup_prior`, but stops before any
inference or expensive sampling. It computes the dense Galerkin matrix of
C_0 and inspects eigenvalues to report (and optionally save) negative
or near-zero eigenvalues.

Use this to quickly explore how `integration_method` and `integration_n_points`
affect the positive-definiteness of the assembled matrix.

Example:
    python check_C0_pd.py --n-points 32 64 128 --methods midpoint trapezoid --prior bessel_sobolev

"""

from pathlib import Path
import argparse
import json
import numpy as np
import itertools
import csv
from typing import List

from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval import Lebesgue
from pygeoinf.interval.configs import (
    IntegrationConfig, LebesgueIntegrationConfig, ParallelConfig
)
from pygeoinf.interval.boundary_conditions import BoundaryConditions
from pygeoinf.interval.operators import Laplacian, BesselSobolevInverse, InverseLaplacian
from pygeoinf.interval.KL_sampler import KLSampler
from pygeoinf.interval.demos.pli_demos.pli_config import PLIConfig, get_fast_config


def build_C0_matrix(cfg: PLIConfig, verbose: bool = True) -> np.ndarray:
    """Construct the prior covariance operator C_0 and return its dense matrix.

    Args:
        cfg: PLIConfig instance with parameters set
        verbose: If True, print progress

    Returns:
        Dense numpy array representing C_0 in the chosen Galerkin basis
    """
    if verbose:
        print(f"Building Lebesgue space with N={cfg.N}, integration: {cfg.integration_method}, n_points={cfg.integration_n_points}")

    # Domain and Lebesgue space
    domain = IntervalDomain(cfg.domain_a, cfg.domain_b)
    integration_cfg = IntegrationConfig(method=cfg.integration_method, n_points=cfg.integration_n_points)
    lebesgue_integration_cfg = LebesgueIntegrationConfig(
        inner_product=integration_cfg,
        dual=integration_cfg,
        general=integration_cfg
    )
    parallel_cfg = ParallelConfig(enabled=True, n_jobs=12)

    M = Lebesgue(
        cfg.N,
        domain,
        basis=cfg.basis,
        integration_config=lebesgue_integration_cfg,
        parallel_config=parallel_cfg
    )

    # Boundary conditions
    bc = BoundaryConditions(bc_type=cfg.bc_type, left=cfg.bc_left, right=cfg.bc_right)

    # Build covariance operator
    if cfg.prior_type == 'inverse_laplacian':
        C_0 = InverseLaplacian(
            M, bc, cfg.alpha_computed,
            method=cfg.method, dofs=cfg.dofs
        )
    elif cfg.prior_type == 'bessel_sobolev':
        L = Laplacian(
            M, bc, cfg.alpha_computed,
            method=cfg.method, dofs=cfg.dofs,
            integration_config=integration_cfg
        )
        C_0 = BesselSobolevInverse(
            M, M, cfg.k, cfg.s, L,
            dofs=cfg.dofs, n_samples=cfg.n_samples,
            use_fast_transforms=cfg.use_fast_transforms
        )
    else:
        raise ValueError(f"Unknown prior_type: {cfg.prior_type}")

    # Try to extract dense Galerkin matrix
    # Use same call-signature as elsewhere in the codebase
    try:
        mat = C_0.matrix(dense=True, galerkin=True, parallel=True, n_jobs=12)
    except TypeError:
        # Some operator implementations might accept fewer args
        mat = C_0.matrix(dense=True, galerkin=True)

    mat = np.asarray(mat)
    return mat


def analyze_matrix(mat: np.ndarray, tol: float = 1e-10, sym_tol: float = 1e-12) -> dict:
    """Compute eigenvalues and report PD statistics.

    This function first checks how symmetric the assembled matrix is by
    computing the relative Frobenius norm of (mat - mat.T). If the matrix
    is noticeably non-symmetric (relative error > ``sym_tol``) a warning
    is reported in the returned dictionary. The matrix is then symmetrized
    before eigenvalue computation to avoid spurious complex eigenvalues.

    Args:
        mat: Dense matrix (expected symmetric up to numerical error)
        tol: Eigenvalue threshold to consider negative or zero
        sym_tol: Relative symmetry tolerance (Frobenius norm of antisymmetric
                 part divided by Frobenius norm of matrix)

    Returns:
        Dictionary with min eigenvalue, count of eigenvalues < -tol,
        count between -tol and tol, symmetry diagnostics, and full
        eigenvalues array.
    """
    # Compute symmetry diagnostics BEFORE symmetrizing
    try:
        frob_mat = np.linalg.norm(mat, ord='fro')
        frob_diff = np.linalg.norm(mat - mat.T, ord='fro')
    except Exception:
        # Fall back to elementwise norms if Frobenius fails
        frob_mat = np.sqrt(np.sum(mat ** 2))
        frob_diff = np.sqrt(np.sum((mat - mat.T) ** 2))

    rel_sym_err = frob_diff / (frob_mat + 1e-30)

    # If the matrix is far from symmetric, surface a warning in the output
    symmetry_ok = bool(rel_sym_err <= sym_tol)
    if not symmetry_ok:
        print(f"WARNING: assembled matrix is not symmetric (rel_err={rel_sym_err:.3e})")

    # Symmetrize for eigen decomposition (safe even if small asymmetry)
    mat_sym = 0.5 * (mat + mat.T)

    # Compute eigenvalues (use eigh for symmetric)
    eigs = np.linalg.eigvalsh(mat_sym)
    min_eig = float(np.min(eigs))
    neg_count = int(np.sum(eigs < -tol))
    near_zero_count = int(np.sum(np.abs(eigs) <= tol))

    return {
        'min_eig': min_eig,
        'neg_count': neg_count,
        'near_zero_count': near_zero_count,
        'eigenvalues': eigs,
        'symmetry': {
            'frob_norm': float(frob_mat),
            'frob_diff': float(frob_diff),
            'rel_sym_err': float(rel_sym_err),
            'symmetry_ok': symmetry_ok,
            'sym_tol': float(sym_tol)
        }
    }


def run_checks(
    base_cfg: PLIConfig,
    n_points_list: List[int],
    methods: List[str],
    output_dir: Path,
    tol: float = 1e-10,
    sym_tol: float = 1e-12,
    prior_type: str = None,
    dofs_list: List[int] = None,
    n_samples_list: List[int] = None,
    save_matrices: bool = False,
):
    """Run checks across grid of integration configs and save summary.

    Args:
        base_cfg: Base PLIConfig to modify per run
        n_points_list: list of integration n_points values
        methods: list of integration methods
        output_dir: directory to store results
        tol: numerical tolerance for eigenvalues
        prior_type: if provided, override base_cfg.prior_type
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    # Default to base config values if lists not provided
    if dofs_list is None:
        dofs_list = [int(base_cfg.dofs)]
    if n_samples_list is None:
        n_samples_list = [int(base_cfg.n_samples)]

    combinations = list(itertools.product(n_points_list, methods, dofs_list, n_samples_list))
    print(f"Running {len(combinations)} configurations")

    for idx, (n_points, method, dofs, n_samples) in enumerate(combinations):
        print(f"\n[{idx+1}/{len(combinations)}] method={method}, n_points={n_points}, dofs={dofs}, n_samples={n_samples}")
        # Create a copy of the base config and apply overrides. Use the
        # provided `copy` method to avoid passing derived fields (like 'k')
        # into the PLIConfig constructor which would raise TypeError.
        cfg = base_cfg.copy()
        cfg.integration_n_points = int(n_points)
        cfg.integration_method = method
        cfg.dofs = int(dofs)
        cfg.n_samples = int(n_samples)
        if prior_type is not None:
            # Use copy to ensure type-consistent construction
            cfg = cfg.copy(prior_type=prior_type)

        try:
            mat = build_C0_matrix(cfg, verbose=False)
            analysis = analyze_matrix(mat, tol=tol, sym_tol=sym_tol)

            row = {
                'method': method,
                'n_points': n_points,
                'dofs': int(dofs),
                'n_samples': int(n_samples),
                'min_eig': analysis['min_eig'],
                'neg_count': analysis['neg_count'],
                'near_zero_count': analysis['near_zero_count'],
                'mat_shape': mat.shape[0],
                'symmetry_rel_err': analysis.get('symmetry', {}).get('rel_sym_err'),
                'symmetry_ok': analysis.get('symmetry', {}).get('symmetry_ok'),
                'symmetry_frob_diff': analysis.get('symmetry', {}).get('frob_diff')
            }

            # Save eigenvalues and matrix for this run (optional)
            prefix = f"C0_{method}_n{n_points}_d{dofs}_s{n_samples}"
            if save_matrices:
                np.save(output_dir / f"{prefix}_eigs.npy", analysis['eigenvalues'])
                np.save(output_dir / f"{prefix}_mat.npy", mat)

            summary_rows.append(row)

            print(f"  min_eig={row['min_eig']:.3e}, neg_count={row['neg_count']}, near_zero={row['near_zero_count']}")

        except Exception as e:
            print(f"  FAILED to build/test C_0: {e}")
            summary_rows.append({
                'method': method,
                'n_points': n_points,
                'dofs': int(dofs) if 'dofs' in locals() else None,
                'n_samples': int(n_samples) if 'n_samples' in locals() else None,
                'min_eig': None,
                'neg_count': None,
                'near_zero_count': None,
                'mat_shape': None,
                'error': str(e)
            })

    # Write summary CSV
    csv_path = output_dir / 'summary.csv'
    keys = [
        'method', 'n_points', 'dofs', 'n_samples', 'mat_shape', 'min_eig', 'neg_count', 'near_zero_count',
        'symmetry_rel_err', 'symmetry_ok', 'symmetry_frob_diff', 'error'
    ]
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for r in summary_rows:
            # Ensure all keys present
            row_out = {k: r.get(k, '') for k in keys}
            writer.writerow(row_out)

    print(f"\nSummary saved to {csv_path}")
    return summary_rows


def plot_summary(summary_rows: List[dict], output_dir: Path) -> None:
    """Create simple visualizations from the summary rows and save PNGs.

    Generates three plots saved to the `output_dir`:
    - symmetry_rel_err vs n_points (colored by method, marker size ~ n_samples)
    - min_eig vs n_points (colored by method)
    - neg_count vs n_points (colored by method)
    """
    # Import matplotlib lazily to avoid hard dependency at import-time
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available — skipping plots")
        return

    # Filter out failed runs
    rows = [r for r in summary_rows if r.get('error') in (None, '', False)]
    if len(rows) == 0:
        print("No successful runs to plot")
        return

    # Extract fields safely
    n_points = [int(r['n_points']) for r in rows]
    methods = [r['method'] for r in rows]
    dofs = [int(r.get('dofs') or 0) for r in rows]
    n_samples = [int(r.get('n_samples') or 0) for r in rows]
    sym_err = [float(r.get('symmetry_rel_err') or np.nan) for r in rows]
    min_eig = [float(r.get('min_eig') if r.get('min_eig') is not None else np.nan) for r in rows]
    neg_count = [int(r.get('neg_count') if r.get('neg_count') is not None else -1) for r in rows]

    # Unique methods for color mapping
    uniq_methods = sorted(set(methods))
    cmap = plt.get_cmap('tab10')
    method_color = {m: cmap(i % 10) for i, m in enumerate(uniq_methods)}

    # Symmetry plot
    plt.figure(figsize=(8, 5))
    for m in uniq_methods:
        idxs = [i for i, mm in enumerate(methods) if mm == m]
        xs = [n_points[i] for i in idxs]
        ys = [sym_err[i] for i in idxs]
        sizes = [max(20, min(200, int(n_samples[i] / 2))) for i in idxs]
        plt.scatter(xs, ys, label=m, color=method_color[m], s=sizes, alpha=0.7)
    plt.xlabel('integration n_points')
    plt.ylabel('relative symmetry error')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / 'symmetry_rel_err.png', dpi=150)
    plt.close()

    # Minimum eigenvalue plot
    plt.figure(figsize=(8, 5))
    for m in uniq_methods:
        idxs = [i for i, mm in enumerate(methods) if mm == m]
        xs = [n_points[i] for i in idxs]
        ys = [min_eig[i] for i in idxs]
        plt.plot(xs, ys, marker='o', label=m, color=method_color[m])
    plt.xlabel('integration n_points')
    plt.ylabel('min eigenvalue')
    plt.xscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'min_eig_vs_npoints.png', dpi=150)
    plt.close()

    # Negative count plot
    plt.figure(figsize=(8, 5))
    for m in uniq_methods:
        idxs = [i for i, mm in enumerate(methods) if mm == m]
        xs = [n_points[i] for i in idxs]
        ys = [neg_count[i] for i in idxs]
        plt.plot(xs, ys, marker='s', label=m, color=method_color[m])
    plt.xlabel('integration n_points')
    plt.ylabel('negative eigenvalue count')
    plt.xscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'neg_count_vs_npoints.png', dpi=150)
    plt.close()

    print(f"Saved plots to {output_dir}")


def parse_list_arg(arg: str) -> List[str]:
    if arg is None:
        return []
    return [a.strip() for a in arg.split(',') if a.strip()]


def main():
    parser = argparse.ArgumentParser(description="Check positive-definiteness of C_0 under integration configs")
    parser.add_argument('--n-points', nargs='+', type=int, default=[32, 64, 128, 256, 512, 1024],
                        help='List of integration n_points to try')
    parser.add_argument('--methods', nargs='+', type=str, default=['simpson', 'trapz'],
                        help='List of integration methods to try')
    parser.add_argument('--dofs', nargs='+', type=int, default=[32, 64, 128],
                        help='List of dofs (model-space basis size) to try')
    parser.add_argument('--n-samples', nargs='+', type=int, default=[32, 64, 128, 256, 512, 1024],
                        help='List of n_samples (fast transform sampling) to try')
    parser.add_argument('--output-dir', type=str, default='c0_checks', help='Directory to save outputs')
    parser.add_argument('--tol', type=float, default=1e-10, help='Eigenvalue numerical tolerance')
    parser.add_argument('--sym-tol', type=float, default=1e-12, help='Relative symmetry tolerance for matrix assembly')
    parser.add_argument('--prior', type=str, default=None, help='Override prior type (inverse_laplacian or bessel_sobolev)')
    parser.add_argument('--config-json', type=str, default=None, help='Optional base config JSON file (from previous runs)')
    parser.add_argument('--save-matrices', action='store_true',
                        help='Save full matrices and eigenvalues for each run (can be large)')
    parser.add_argument('--plot', action='store_true', help='Create summary plots (requires matplotlib)')
    args = parser.parse_args()

    # Load base config
    if args.config_json:
        cfg_path = Path(args.config_json)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config JSON not found: {cfg_path}")
        # Use the classmethod load which strips derived parameters and
        # converts lists to tuples where appropriate.
        base_cfg = PLIConfig.load(cfg_path)
    else:
        # Use the provided convenience factory
        base_cfg = get_fast_config()

    # Run checks
    out = run_checks(
        base_cfg,
        n_points_list=args.n_points,
        methods=args.methods,
        output_dir=Path(args.output_dir),
        tol=args.tol,
        sym_tol=args.sym_tol,
        prior_type=args.prior,
        dofs_list=args.dofs,
        n_samples_list=args.n_samples,
        save_matrices=args.save_matrices,
    )

    # Also save a small JSON summary
    summary_json = Path(args.output_dir) / 'summary.json'
    with open(summary_json, 'w') as f:
        json.dump(out, f, indent=2, default=lambda o: None)

    # Optionally plot summary
    if args.plot:
        plot_summary(out, Path(args.output_dir))


if __name__ == '__main__':
    main()
