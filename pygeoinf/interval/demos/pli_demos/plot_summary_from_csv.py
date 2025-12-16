"""
Quick plotting helper for `summary.csv` produced by `check_C0_pd.py`.

Usage:
    python plot_summary_from_csv.py --csv path/to/summary.csv --outdir ./figs

Produces:
 - fraction_symmetry_ok_by_npoints.png  (line plot: fraction symmetric vs n_points per method)
 - symmetry_rel_err_scatter.png         (scatter: symmetry_rel_err vs n_points colored by method)
 - min_eig_vs_npoints.png              (line plot: min eigenvalue vs n_points per method)

The script prefers `pandas` + `matplotlib` but will fall back to the csv module if pandas
is not available.
"""

from pathlib import Path
import argparse
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_summary(csv_path):
    csv_path = Path(csv_path)
    if pd is not None:
        df = pd.read_csv(csv_path)
    else:
        # minimal fallback
        import csv as _csv
        rows = []
        with open(csv_path, newline='') as fh:
            reader = _csv.DictReader(fh)
            for r in reader:
                rows.append(r)
        # convert to structured dict-of-lists
        df = {}
        for k in rows[0].keys():
            df[k] = [r[k] for r in rows]
        # wrap simple namespace-like access
        class SimpleDF(dict):
            def __getattr__(self, item):
                return self[item]
        df = SimpleDF(df)
    return df


def ensure_numeric(df, col):
    """Return a numpy array of numeric values for column `col` from dataframe-like `df`.
    Handles strings like 'True'/'False' for boolean columns."""
    if pd is not None and isinstance(df, pd.DataFrame):
        s = df[col]
        # coerce booleans stored as strings
        if s.dtype == object:
            s2 = s.replace({'True': True, 'False': False, 'true': True, 'false': False})
            try:
                return s2.astype(float).to_numpy()
            except Exception:
                # attempt boolean
                return s2.astype(bool).to_numpy()
        else:
            return s.to_numpy()
    else:
        arr = np.array(df[col])
        # try to convert
        try:
            return arr.astype(float)
        except Exception:
            # booleans?
            lowered = np.char.lower(arr.astype(str))
            return np.where(lowered == 'true', True, False)


def plot_fraction_symmetry_ok(df, outdir: Path):
    """Plot fraction of runs with symmetry_ok == True vs n_points for each method."""
    if plt is None:
        print("matplotlib not available — skipping plots")
        return

    if pd is not None and isinstance(df, pd.DataFrame):
        grouping = df.groupby(['n_points', 'method'])['symmetry_ok'].apply(lambda x: np.mean(x.astype(bool))).unstack('method')
        ax = grouping.plot(marker='o', logx=True, figsize=(8,5))
        ax.set_xlabel('integration n_points')
        ax.set_ylabel('fraction symmetry_ok')
        ax.grid(True, which='both', ls='--', alpha=0.4)
        out = outdir / 'fraction_symmetry_ok_by_npoints.png'
        ax.get_figure().tight_layout()
        ax.get_figure().savefig(out, dpi=150)
        print(f"Saved {out}")
    else:
        # fallback: manual aggregation
        n_points = [int(x) for x in df['n_points']]
        methods = list(set(df['method']))
        pts = sorted(set(n_points))
        data = {m: [] for m in methods}
        for p in pts:
            idxs = [i for i, v in enumerate(n_points) if v == p]
            for m in methods:
                vals = [df['symmetry_ok'][i] for i in idxs if df['method'][i] == m]
                # coerce
                vals_bool = [1 if str(v).lower() in ('1','true','yes') else 0 for v in vals]
                data[m].append(np.mean(vals_bool) if len(vals_bool) else np.nan)
        plt.figure(figsize=(8,5))
        for m in methods:
            plt.plot(pts, data[m], marker='o', label=m)
        plt.xscale('log')
        plt.xlabel('integration n_points')
        plt.ylabel('fraction symmetry_ok')
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.4)
        out = outdir / 'fraction_symmetry_ok_by_npoints.png'
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")


def plot_symmetry_rel_err_scatter(df, outdir: Path):
    """Scatter plot symmetry_rel_err vs n_points, colored by method."""
    if plt is None:
        return
    if pd is not None and isinstance(df, pd.DataFrame):
        fig, ax = plt.subplots(figsize=(8,5))
        methods = sorted(df['method'].unique())
        cmap = plt.get_cmap('tab10')
        for i, m in enumerate(methods):
            sub = df[df['method'] == m]
            ax.scatter(sub['n_points'], sub['symmetry_rel_err'].astype(float), label=m, color=cmap(i%10), alpha=0.7)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('integration n_points')
        ax.set_ylabel('relative symmetry error')
        ax.grid(True, which='both', ls='--', alpha=0.4)
        ax.legend()
        fig.tight_layout()
        out = outdir / 'symmetry_rel_err_scatter.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")
    else:
        # fallback simple scatter
        n_points = np.array([int(x) for x in df['n_points']])
        methods = list(set(df['method']))
        cmap = plt.get_cmap('tab10')
        plt.figure(figsize=(8,5))
        for i, m in enumerate(methods):
            idxs = [j for j, mm in enumerate(df['method']) if mm == m]
            xs = n_points[idxs]
            ys = np.array([float(df['symmetry_rel_err'][j]) if df['symmetry_rel_err'][j] not in (None, '') else np.nan for j in idxs])
            plt.scatter(xs, ys, label=m, color=cmap(i%10), alpha=0.7)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('integration n_points')
        plt.ylabel('relative symmetry error')
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.4)
        out = outdir / 'symmetry_rel_err_scatter.png'
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")


def plot_min_eig(df, outdir: Path):
    if plt is None:
        return
    if pd is not None and isinstance(df, pd.DataFrame):
        fig, ax = plt.subplots(figsize=(8,5))
        methods = sorted(df['method'].unique())
        cmap = plt.get_cmap('tab10')
        for i, m in enumerate(methods):
            sub = df[df['method'] == m]
            ax.plot(sub['n_points'], sub['min_eig'].astype(float), marker='o', label=m, color=cmap(i%10))
        ax.set_xscale('log')
        ax.set_xlabel('integration n_points')
        ax.set_ylabel('min eigenvalue')
        ax.grid(True, which='both', ls='--', alpha=0.4)
        ax.legend()
        fig.tight_layout()
        out = outdir / 'min_eig_vs_npoints.png'
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")
    else:
        # fallback: simple plotting
        n_points = np.array([int(x) for x in df['n_points']])
        methods = list(set(df['method']))
        cmap = plt.get_cmap('tab10')
        plt.figure(figsize=(8,5))
        for i, m in enumerate(methods):
            idxs = [j for j, mm in enumerate(df['method']) if mm == m]
            xs = n_points[idxs]
            ys = np.array([float(df['min_eig'][j]) if df['min_eig'][j] not in (None, '') else np.nan for j in idxs])
            plt.plot(xs, ys, marker='o', label=m, color=cmap(i%10))
        plt.xscale('log')
        plt.xlabel('integration n_points')
        plt.ylabel('min eigenvalue')
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.4)
        out = outdir / 'min_eig_vs_npoints.png'
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description='Plot summary.csv produced by check_C0_pd.py')
    parser.add_argument('--csv', type=str, default='summary.csv', help='Path to summary.csv')
    parser.add_argument('--outdir', type=str, default='figs', help='Directory to save figures')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_summary(csv_path)
    plot_fraction_symmetry_ok(df, outdir)
    plot_symmetry_rel_err_scatter(df, outdir)
    plot_min_eig(df, outdir)


if __name__ == '__main__':
    main()
