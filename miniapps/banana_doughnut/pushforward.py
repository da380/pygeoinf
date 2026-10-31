"""Part D: the two theories' laws of the delays over the fan of distances.

At each band, ray theory's functionals of every ray of the scale, stacked,
are one operator from the model space onto ``R^N``, ``N`` the distances, and
finite-frequency theory's another. A prior pushed forward through each is a
Gaussian on ``R^N``: the law of the ``N`` delays each theory predicts before
any datum. How far apart the two laws are is their Kullback-Leibler
divergence, taken both ways: ``KL(FF || ray)`` is the expected
log-likelihood ratio in favor of finite-frequency theory's law when
noise-free delays obey it, how readily such delays would tell the two
theories apart, and ``KL(ray || FF)`` the same the other way. It compares
the two laws, not the two delays of one model: it is zero where the laws
are the same, whatever the delays do draw by draw, and it weighs each
combination of the delays by how much it varies, so that combinations the
prior holds nearly fixed, differences between neighboring rays under a
smooth prior, count for much.

Both push-forwards are marginals of one: the prior pushed forward through
every functional of the scale at once, ray theory's of every ray and
finite-frequency theory's of every ray in every band, a Gaussian on ``R^(N
(1 + B))`` whose covariance is assembled once; the projection onto one
theory's delays in one band pushes it forward to that theory's law, which is
the push-forward through that theory's stacked operator. Its blocks also
give the correlation of the two delays at each distance, part C's, which
the draws are checked against.

Three priors of the scale: the smoothest and the roughest isotropic one, by
the distance at which their correlation from the reference ray's turning
point falls to ``1/e`` along the shell, and the first anisotropic pair. For
each, a figure: the ray delay against the finite-frequency delay of every
ray for draws of the model from the prior, each through both theories'
functionals, a panel a band, a color a distance, and the diagonal
``dT_FF = dT_ray``. The numbers go to the printout and to a table beside
the figures.
"""

from __future__ import annotations

import csv
import pathlib
import time
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

import matplotlib.pyplot as plt

from gradients import prior_tag
from priors import short
from pygeoinf import EuclideanSpace, GaussianMeasure, LinearOperator
from src.common import parse, save
from src.correlations import describe
from src.functionals import thread_map
from src.problem import FULL, QUICK, Problem, Scale, build

#: The table's name in the output directory.
TABLE = "pushforward.csv"
#: How far the scatter panels reach beyond the largest delay drawn.
MARGIN = 1.08


@dataclass(frozen=True)
class Divergence:
    """The two theories' laws of the delays of every ray, in one band under
    one prior.

    Attributes:
        period: the band's dominant period.
        radial_length: the prior's nominal length scale along a radius.
        lateral_length: its nominal length scale across one.
        shell_distance: where the prior's correlation from the reference
            ray's turning point falls to ``1/e`` along the shell.
        radial_distance: where it does along the radius, NaN if not inside
            the mantle.
        kl_ff_ray: ``KL(FF || ray)``, finite-frequency theory's law of the
            delays against ray theory's.
        kl_ray_ff: ``KL(ray || FF)``.
        distances: the rays' distances, in degrees.
        std_ray: the ray delay's standard deviation at each distance.
        std_ff: the finite-frequency delay's at each.
        correlation: the two delays' correlation at each, from the
            push-forward.
        sample_correlation: their sample correlation at each over the draws.
        draws: how many draws.
    """

    period: float
    radial_length: float
    lateral_length: float
    shell_distance: float
    radial_distance: float
    kl_ff_ray: float
    kl_ray_ff: float
    distances: tuple[float, ...]
    std_ray: tuple[float, ...]
    std_ff: tuple[float, ...]
    correlation: tuple[float, ...]
    sample_correlation: tuple[float, ...]
    draws: int

    @property
    def key(self) -> tuple[float, float]:
        """The prior's nominal lengths."""
        return (self.radial_length, self.lateral_length)


def chosen(problem: Problem, /) -> list[tuple[float, float]]:
    """The priors of part D, by their nominal lengths: the smoothest and the
    roughest isotropic prior of the scale, by the distance along the shell
    at which their correlation from the reference ray's turning point falls
    to ``1/e``, and the scale's first anisotropic pair."""
    scale = problem.scale
    isotropic = sorted(
        {(float(length), float(length)) for length in scale.prior_lengths},
        key=lambda key: problem.correlation(key).shell_distance,
    )
    keys = [isotropic[-1], isotropic[0]]
    if scale.prior_pairs:
        radial, lateral = scale.prior_pairs[0]
        keys.append((float(radial), float(lateral)))
    return list(dict.fromkeys(keys))


def projection(total: int, start: int, count: int, /) -> LinearOperator:
    """The projection of ``R^total`` onto ``count`` of its coordinates from
    ``start`` on."""
    matrix = np.zeros((count, total))
    matrix[np.arange(count), start + np.arange(count)] = 1.0
    return LinearOperator.from_matrix(
        EuclideanSpace(total), EuclideanSpace(count), matrix, form="components"
    )


def joint_law(problem: Problem, key: tuple[float, float], /) -> GaussianMeasure:
    """The prior of those nominal lengths pushed forward through every
    functional of the scale, :meth:`~src.problem.Problem.stacked_operator`,
    its covariance assembled.

    Where the two theories' delays nearly coincide, in the narrow bands,
    that covariance is singular to rounding, and the factorization that
    comes with the assembly clips the negative eigenvalues the rounding
    leaves, which the library warns of; the clipping is at the level of
    rounding and is expected here, so that warning is not shown.
    """
    pushed = problem.prior(*key).push_forward(problem.stacked_operator())
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The covariance matrix has small negative eigenvalues"
        )
        return pushed.with_dense_covariance()


def laws(
    problem: Problem, joint: GaussianMeasure, band: int, /
) -> tuple[GaussianMeasure, GaussianMeasure]:
    """Ray theory's and finite-frequency theory's laws of the delays of
    every ray in one band, by its index: the joint law pushed forward onto
    each theory's block."""
    count = len(problem.paths)
    total = count * (1 + len(problem.bands))
    ray = joint.push_forward(projection(total, 0, count))
    finite = joint.push_forward(projection(total, count * (1 + band), count))
    return ray, finite


def draw_delays(
    problem: Problem,
    key: tuple[float, float],
    count: int,
    /,
    *,
    seed: Any,
    n_jobs: int | None = None,
) -> np.ndarray:
    """The delays of every functional of the scale, in the order of
    :meth:`~src.problem.Problem.stacked_operator`, for draws of the model
    from a prior: each draw from its own generator spawned from the seed, so
    that they do not depend on ``n_jobs``.

    Returns:
        ``(count, N (1 + B))``.
    """
    prior = problem.prior(*key)
    stack = problem.stacked_operator()
    streams = np.random.SeedSequence(seed).spawn(int(count))
    delays = thread_map(
        lambda stream: stack(prior.sample(rng=np.random.default_rng(stream))),
        streams,
        n_jobs=n_jobs,
    )
    return np.array(delays, dtype=float).reshape(int(count), len(problem.stacked_keys))


def divergences(
    problem: Problem,
    keys: Sequence[tuple[float, float]],
    /,
    *,
    draws: int,
    n_jobs: int | None = None,
) -> tuple[list[Divergence], dict]:
    """For each prior and band: the two laws' divergences both ways, and the
    correlation of the two delays at each distance from the push-forward and
    from draws.

    The rows of every functional are formed first, in ``n_jobs`` threads,
    then each prior's joint law, a push-forward a prior in a thread each,
    then each prior's draws in ``n_jobs`` threads, the draws of the prior
    whose index in the scale's priors is ``k`` from the seed ``(seed, k)``.
    With fewer than two draws the sample correlations are NaN.

    Returns:
        The numbers, prior by prior and band by band within each, and the
        delays drawn under each prior, by its nominal lengths, as
        :func:`draw_delays` gives them.
    """
    scale = problem.scale
    problem.rows(problem.stacked_keys, n_jobs=n_jobs)
    for key in keys:
        problem.prior(*key)
        problem.correlation(key)
    joints = dict(
        zip(keys, thread_map(lambda key: joint_law(problem, key), keys, n_jobs=n_jobs))
    )
    order = scale.prior_scales
    delays = {
        key: draw_delays(
            problem,
            key,
            draws,
            seed=[scale.seed, order.index(key) if key in order else len(order)],
            n_jobs=n_jobs,
        )
        for key in keys
    }
    sampled = int(draws) > 1
    count = len(problem.paths)
    rays = np.arange(count)
    found = []
    for key in keys:
        joint = joints[key]
        covariance = np.asarray(joint.covariance.matrix(form="components"))
        measured = problem.correlation(key)
        for band, period in enumerate(scale.periods):
            ray, finite = laws(problem, joint, band)
            ff = count * (1 + band) + rays
            a, b = covariance[rays, rays], covariance[ff, ff]
            c = covariance[rays, ff]
            sample = [
                float(np.corrcoef(delays[key][:, i], delays[key][:, j])[0, 1])
                if sampled
                else float("nan")
                for i, j in zip(rays, ff)
            ]
            found.append(
                Divergence(
                    period=float(period),
                    radial_length=key[0],
                    lateral_length=key[1],
                    shell_distance=measured.shell_distance,
                    radial_distance=measured.radial_distance,
                    kl_ff_ray=float(finite.kl_divergence(ray)),
                    kl_ray_ff=float(ray.kl_divergence(finite)),
                    distances=tuple(float(path.distance) for path in problem.paths),
                    std_ray=tuple(float(x) for x in np.sqrt(a)),
                    std_ff=tuple(float(x) for x in np.sqrt(b)),
                    correlation=tuple(float(x) for x in c / np.sqrt(a * b)),
                    sample_correlation=tuple(sample),
                    draws=int(draws),
                )
            )
    return found, delays


# --------------------------------------------------------------------------- #
#                                  Drawing                                    #
# --------------------------------------------------------------------------- #


def scatter_figure(
    problem: Problem,
    key: tuple[float, float],
    delays: np.ndarray,
    /,
    *,
    outdir: pathlib.Path,
) -> pathlib.Path:
    """Under one prior, the ray delay against the finite-frequency delay of
    every ray for each draw, a panel a band and a color a distance, with
    the diagonal."""
    count, bands = len(problem.paths), len(problem.bands)
    fig, axes = plt.subplots(
        1, bands, figsize=(4.2 * bands + 0.6, 5.0), squeeze=False, layout="constrained"
    )
    for band, ax in enumerate(axes[0]):
        ff = count * (1 + band)
        reach = MARGIN * float(
            np.abs(delays[:, list(range(count)) + list(range(ff, ff + count))]).max()
        )
        line = np.array([-reach, reach])
        ax.plot(line, line, "k--", lw=0.8, label="δT_FF = δT_ray")
        for index, path in enumerate(problem.paths):
            ax.plot(
                delays[:, index],
                delays[:, ff + index],
                "o",
                ms=2.2,
                alpha=0.55,
                mec="none",
                color=f"C{index}",
                label=f"{path.distance:g}°",
            )
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        # Square, so that with equal limits the two delays share one scale.
        ax.set_box_aspect(1.0)
        ax.set_xlabel("δT_ray", fontsize=8)
        ax.set_ylabel("δT_FF", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(style="sci", scilimits=(-2, 2))
        ax.set_title(f"T0 = {problem.bands[band].period:g}", fontsize=9)
    axes[0, 0].legend(fontsize=7, loc="upper left", markerscale=2.0)
    name = f"pushforward_{prior_tag(key)}.png"
    save(fig, outdir / name, title=f"Delays of prior draws, {short(key)}")
    return outdir / name


def write_table(
    problem: Problem, rows: Sequence[Divergence], path: pathlib.Path, /
) -> None:
    """The numbers as a table with a title, one row a band and prior, the
    standard deviations and correlations of each distance in columns of
    their own."""
    scale = problem.scale
    distances = rows[0].distances
    fields = [
        "period",
        "radial_length",
        "lateral_length",
        "shell_distance",
        "radial_distance",
        "kl_ff_ray",
        "kl_ray_ff",
        "draws",
    ]
    for name in ("std_ray", "std_ff", "correlation", "sample_correlation"):
        fields += [f"{name}_{distance:g}deg" for distance in distances]
    with open(path, "w", newline="") as stream:
        stream.write(
            "# Banana-doughnut, part D: the laws of the ray and finite-frequency "
            "delays of the rays at "
            + ", ".join(f"{d:g}" for d in distances)
            + " degrees, under Sobolev priors of order "
            f"{scale.prior_order:g} and pointwise std {scale.prior_std:g}, on "
            f"H^{scale.order:g} of length scale {scale.length_scale:g} at lmax "
            f"{scale.lmax}, the surface by "
            f"{'images' if scale.surface == 'fold' else 'clipping'}\n"
        )
        stream.write(
            "# kl_ff_ray is KL(FF || ray) and kl_ray_ff KL(ray || FF) between the "
            "two Gaussians on R^N, N the distances; shell_distance and "
            "radial_distance are where the prior's correlation from the turning "
            f"point of the ray at {scale.reference_distance:g} degrees falls to "
            "1/e along the shell and along the radius (nan: not inside the "
            "mantle), in units of the radius; standard deviations in units of "
            "time; correlation from the push-forward, sample_correlation over "
            "the draws\n"
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values: dict[str, Any] = {
                "period": row.period,
                "radial_length": row.radial_length,
                "lateral_length": row.lateral_length,
                "shell_distance": row.shell_distance,
                "radial_distance": row.radial_distance,
                "kl_ff_ray": row.kl_ff_ray,
                "kl_ray_ff": row.kl_ray_ff,
                "draws": row.draws,
            }
            for name in ("std_ray", "std_ff", "correlation", "sample_correlation"):
                for distance, value in zip(distances, getattr(row, name)):
                    values[f"{name}_{distance:g}deg"] = value
            writer.writerow({k: f"{v:.10g}" for k, v in values.items()})
    print(f"table written to {path}")


def print_numbers(rows: Sequence[Divergence], /) -> None:
    """The divergences, and the correlations at each distance."""
    distances = ", ".join(f"{d:g}" for d in rows[0].distances)
    print(
        f"  {'prior':>9s} {'T0':>6s} {'KL(FF||ray)':>12s} {'KL(ray||FF)':>12s}  "
        f"1 - correlation at {distances} degrees, push-forward / draws"
    )
    for row in rows:
        prior = (
            f"{row.radial_length:g}"
            if row.radial_length == row.lateral_length
            else f"{row.radial_length:g},{row.lateral_length:g}"
        )
        pairs = "  ".join(
            f"{1.0 - exact:.2e}/{1.0 - sampled:.2e}"
            for exact, sampled in zip(row.correlation, row.sample_correlation)
        )
        print(
            f"  {prior:>9s} {row.period:6g} {row.kl_ff_ray:12.3e} "
            f"{row.kl_ray_ff:12.3e}  {pairs}"
        )


def run(
    scale: Scale, /, *, outdir: pathlib.Path, problem: Problem | None = None
) -> dict:
    """The two laws at every band under the smoothest, the roughest and an
    anisotropic prior: the divergences, the draws' scatter, the table."""
    if problem is None:
        problem = build(scale=scale)
    print(
        f"\n=== Part D: the two theories' laws over the distances ({scale.name} "
        "scale) ==="
    )
    t0 = time.perf_counter()
    keys = chosen(problem)
    found, delays = divergences(problem, keys, draws=scale.draws, n_jobs=scale.n_jobs)
    print(
        f"{len(problem.paths)} distances, {len(problem.bands)} bands, "
        f"{len(keys)} priors: each prior pushed forward through all "
        f"{len(problem.stacked_keys)} functionals at once and drawn "
        f"{scale.draws} times, in {time.perf_counter() - t0:.1f} s"
    )
    for key in keys:
        print(f"  {short(key)}: {describe(problem.correlation(key))}")
    print_numbers(found)
    write_table(problem, found, outdir / TABLE)
    figures = [scatter_figure(problem, key, delays[key], outdir=outdir) for key in keys]
    return {"divergences": found, "figures": figures, "table": outdir / TABLE}


def main() -> None:
    args = parse(__doc__, quick=QUICK, full=FULL)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
