"""Plotting cost: v2 plot vs v1 plot, transforms per call, cartopy overhead."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from yg_util import TransformCounter, bench, fmt
import numpy as np
import time
import cProfile, pstats, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pygeoinf2.symmetric_space.sphere import Sobolev as Sob2
from pygeoinf2 import plotting
import pygeoinf.symmetric_space.sphere as sph1

rng = np.random.default_rng(6)

for lmax in (128, 256):
    X = Sob2(lmax, 2.0, 0.2); X1 = sph1.Sobolev(lmax, 2.0, 0.2)
    x = X.heat_measure(0.3).sample(rng=rng)
    x1 = X1.from_components(X.to_components(x))
    with TransformCounter() as c:
        ax, im = plotting.plot(X, x); plt.close(ax.figure)
    print(f"lmax {lmax}: transforms in plotting.plot: {c}")
    def v2_plot():
        ax, im = plotting.plot(X, x, coasts=True); plt.close(ax.figure)
    def v2_plot_noc():
        ax, im = plotting.plot(X, x, colorbar=False); plt.close(ax.figure)
    def v2_plot_draw():
        ax, im = plotting.plot(X, x); ax.figure.canvas.draw(); plt.close(ax.figure)
    def v1_plot():
        ax, im = sph1.plot(x1, coasts=True, colorbar=True); plt.close(ax.figure)
    def v1_plot_draw():
        ax, im = sph1.plot(x1, colorbar=True, gridlines=False); ax.figure.canvas.draw(); plt.close(ax.figure)
    def subplots_only():
        fig, ax = plotting.subplots(X); plt.close(fig)
    def pcolormesh_plain():
        fig, ax = plt.subplots(); ax.pcolormesh(X.grid_values(x)); plt.close(fig)
    print(fmt(bench({
        "v2 plot(coasts, colorbar)": v2_plot,
        "v2 plot(no colorbar)": v2_plot_noc,
        "v2 plot + canvas.draw": v2_plot_draw,
        "v1 plot(coasts, colorbar, gridlines)": v1_plot,
        "v1 plot + canvas.draw (no gridlines)": v1_plot_draw,
        "subplots(space) only": subplots_only,
        "plain pcolormesh, no cartopy": pcolormesh_plain,
    }, reps=3)))

X = Sob2(256, 2.0, 0.2); x = X.heat_measure(0.3).sample(rng=rng)
pr = cProfile.Profile(); pr.enable(); ax, im = plotting.plot(X, x); ax.figure.canvas.draw(); pr.disable(); plt.close(ax.figure)
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
print("\n".join(s.getvalue().splitlines()[:34]))
