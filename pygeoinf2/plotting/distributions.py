"""
Drawing a measure on a low-dimensional space: marginals and corner plots.

The end of the estimator chain. An inversion on a function space produces a
posterior nobody can look at; :meth:`push_forward` turns it into a measure on a
handful of properties, and this draws *that* — which is what makes the property
space the thing worth asking about rather than a convenience.

Two entry points, and both take either kind of measure:

* a :class:`~pygeoinf2.probability.gaussian.GaussianMeasure` is drawn exactly,
  from its mean and covariance;
* any measure that can be *sampled* is drawn from draws instead — histograms
  and kernel density in place of curves and ellipses. That covers the posterior
  of a non-linear problem, and the randomise-then-optimise sampler of §18.7,
  neither of which has a covariance to hand.

**Components, not fields.** Everything here is about the components of a
vector, so the covariance that matters is the covariance *of the components*,
``G^-1 C_gal G^-1`` — not the covariance operator's component matrix, which is
a different thing on a space whose basis is not orthonormal, by 75% on the
weighted space in the test suite.
:meth:`~pygeoinf2.probability.gaussian.GaussianMeasure.as_multivariate_normal`
already computes it, so this asks for that rather than doing it again.

See DESIGN.md section 30.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["plot_densities", "plot_corner", "moments"]

_PRIOR_COLOURS = ("green", "orange", "purple", "brown", "olive", "teal")
_POSTERIOR_COLOURS = ("tab:blue", "tab:red", "darkgreen", "tab:orange", "purple")


def moments(
    measure: Any,
    /,
    *,
    samples: int = 20000,
    rng: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """The mean and covariance of a measure's components, and its draws.

    Exactly for a Gaussian, and from *samples* draws for anything that can be
    sampled. The draws come back too, because a non-Gaussian measure should be
    drawn as what it is rather than as the Gaussian with the same first two
    moments — the moments alone would hide precisely what makes it interesting.

    Args:
        measure: a Gaussian measure, or any measure that can be sampled.
        samples: draws to take, when sampling is the route.
        rng: the generator for those draws.

    Returns:
        ``(mean, covariance, draws)`` in components, with ``draws`` None when
        the measure was Gaussian and nothing had to be sampled.
    """
    from ..probability.gaussian import GaussianMeasure

    if isinstance(measure, GaussianMeasure) and measure.covariance is not None:
        frozen = measure.as_multivariate_normal()
        return np.atleast_1d(frozen.mean), np.atleast_2d(frozen.cov), None

    if not measure.can_sample:
        raise TypeError(
            f"{type(measure).__name__} is neither a Gaussian measure with a "
            f"covariance nor something that can be sampled, so there is "
            f"nothing to draw. Supply a covariance, a covariance factor, or a "
            f"sampler."
        )
    space = measure.domain
    draws = np.array(
        [space.to_components(measure.sample(rng=rng)) for _ in range(samples)]
    )
    return draws.mean(axis=0), np.atleast_2d(np.cov(draws.T)), draws


def _span(
    mean: np.ndarray,
    deviation: np.ndarray,
    truth: np.ndarray | None,
    width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Axis limits wide enough for the distribution *and* the truth.

    A truth marker outside the frame is worse than no marker: it says the fit
    is good by not being visible. So the window opens until it is in, with a
    little room to spare.
    """
    reach = np.full_like(deviation, width, dtype=float)
    if truth is not None:
        live = deviation > 0.0
        zed = np.zeros_like(reach)
        zed[live] = np.abs(truth[live] - mean[live]) / deviation[live]
        reach = np.maximum(reach, 1.05 * zed)
    return mean - reach * deviation, mean + reach * deviation


def _grid(
    limits: tuple[float, float],
    mean: float,
    deviation: float,
    width: float,
    *,
    points: int = 2000,
) -> np.ndarray:
    """Points spanning the shared window, concentrated on one curve's own.

    A single grid over the union of every curve's window resolves the widest
    and aliases the rest. With a prior a thousand times wider than its
    posterior — the case the twin axis exists for — a uniform 2000 points put
    several posterior standard deviations between samples, so the posterior
    peak was simply missed. Raising the count cannot fix it: the ratio, not the
    absolute width, is what defeats a shared grid.

    So each curve gets the shared span for its extent and a fine grid over its
    own few standard deviations for its shape, which is where all of its mass
    is. That is ~160 points per standard deviation regardless of the ratio.
    """
    low, high = limits
    coarse = np.linspace(low, high, points)
    if deviation <= 0.0:
        return coarse
    own_low = max(low, mean - width * deviation)
    own_high = min(high, mean + width * deviation)
    if own_high <= own_low:
        return coarse
    fine = np.linspace(own_low, own_high, points)
    return np.unique(np.concatenate([coarse, fine]))


def _density(
    axis: Any,
    limits: tuple[float, float],
    mean: float,
    deviation: float,
    draws: np.ndarray | None,
    *,
    colour: str,
    label: str | None,
    style: str,
    fill: bool,
    width: float,
    points: int = 2000,
) -> None:
    """One marginal, exactly or from draws, on a grid that resolves it."""
    values = _grid(limits, mean, deviation, width, points=points)
    if draws is None:
        from scipy.stats import norm

        density = norm.pdf(values, loc=mean, scale=deviation)
    else:
        from scipy.stats import gaussian_kde

        density = gaussian_kde(draws)(values)
    axis.plot(values, density, color=colour, lw=2, linestyle=style, label=label)
    if fill:
        axis.fill_between(values, density, color=colour, alpha=0.15)


def plot_densities(
    posterior: Any,
    /,
    *,
    prior: Any = None,
    truth: float | None = None,
    index: int = 0,
    ax: Any = None,
    labels: Sequence[str] | None = None,
    prior_labels: Sequence[str] | None = None,
    width: float = 6.0,
    fill: bool = False,
    samples: int = 20000,
    rng: Any = None,
    xlabel: str = "property value",
) -> Any:
    """One component's marginal, for one or several measures.

    v1's ``plot_1d_distributions``. Priors go on a second y-axis, because a
    prior is usually far wider than the posterior it is being compared with and
    sharing an axis makes the posterior a spike — the comparison worth seeing
    is of *shape and position*, not of height.

    Args:
        posterior: a measure, or several.
        prior: a measure, or several, drawn dotted on a right-hand axis.
        truth: a value to mark with a vertical line.
        index: which component to draw, for measures on more than one.
        ax: axes to draw on. The current axes if omitted.
        labels, prior_labels: legend entries. Means are used if omitted.
        width: half-width of the window, in standard deviations.
        fill: shade under the curves.
        samples, rng: for measures that must be sampled.
        xlabel: label for the shared x-axis.

    Returns:
        The axes drawn on, or ``(posterior_axes, prior_axes)`` when priors were
        given and a second axis was made.
    """
    import matplotlib.pyplot as plt

    posteriors = (
        list(posterior) if isinstance(posterior, (list, tuple)) else [posterior]
    )
    priors = (
        []
        if prior is None
        else (list(prior) if isinstance(prior, (list, tuple)) else [prior])
    )

    def summarise(measures: Sequence[Any]) -> list[tuple[float, float, Any]]:
        summary = []
        for one in measures:
            mean, covariance, draws = moments(one, samples=samples, rng=rng)
            if not 0 <= index < mean.size:
                raise IndexError(
                    f"Component {index} is out of range for a measure on "
                    f"{mean.size} components."
                )
            summary.append(
                (
                    float(mean[index]),
                    float(np.sqrt(covariance[index, index])),
                    None if draws is None else draws[:, index],
                )
            )
        return summary

    posterior_summary = summarise(posteriors)
    prior_summary = summarise(priors)
    every = posterior_summary + prior_summary
    live = [(m, s) for m, s, _ in every if s > 0.0]
    if not live:
        raise ValueError("Every measure given has zero variance in this component.")

    means = np.array([m for m, _ in live])
    deviations = np.array([s for _, s in live])
    truths = None if truth is None else np.full(means.shape, float(truth))
    lower, upper = _span(means, deviations, truths, width)
    # The grid has to resolve the *narrowest* curve, not the widest. A fixed
    # count over the union of the windows aliases a posterior much sharper than
    # its prior — the case the twin axis exists for: with a prior 1000x wider,
    # 2000 points put roughly six posterior standard deviations between
    # samples, so the posterior peak is missed altogether. v1 asks for 25
    # points per standard deviation of the narrowest peak; so does this.
    limits = (float(lower.min()), float(upper.max()))

    axis = plt.gca() if ax is None else ax
    axis.set_xlabel(xlabel)
    axis.grid(True, linestyle="--", alpha=0.5)

    prior_axis = None
    if prior_summary:
        prior_axis = axis.twinx()
        prior_axis.set_ylabel("prior density", color=_PRIOR_COLOURS[0])
        prior_axis.tick_params(axis="y", labelcolor=_PRIOR_COLOURS[0])
        for position, (mean, deviation, draws) in enumerate(prior_summary):
            label = (
                prior_labels[position]
                if prior_labels is not None and position < len(prior_labels)
                else f"prior (mean {mean:.4g})"
            )
            _density(
                prior_axis,
                limits,
                mean,
                deviation,
                draws,
                colour=_PRIOR_COLOURS[position % len(_PRIOR_COLOURS)],
                label=label,
                style=":",
                fill=fill,
                width=width,
            )

    axis.set_ylabel("posterior density")
    for position, (mean, deviation, draws) in enumerate(posterior_summary):
        label = (
            labels[position]
            if labels is not None and position < len(labels)
            else f"posterior (mean {mean:.4g})"
        )
        _density(
            axis,
            limits,
            mean,
            deviation,
            draws,
            colour=_POSTERIOR_COLOURS[position % len(_POSTERIOR_COLOURS)],
            label=label,
            style="-",
            fill=fill,
            width=width,
        )

    if truth is not None:
        axis.axvline(truth, color="black", lw=1.5, linestyle="--", label="truth")

    handles, texts = axis.get_legend_handles_labels()
    if prior_axis is not None:
        extra, extra_texts = prior_axis.get_legend_handles_labels()
        handles, texts = handles + extra, texts + extra_texts
    axis.legend(handles, texts, loc="upper right", fontsize="small")
    return axis if prior_axis is None else (axis, prior_axis)


def _contour_levels(
    mean: np.ndarray, covariance: np.ndarray, truth: np.ndarray | None, sigmas: int
) -> np.ndarray:
    """Sigma levels, opened up until the truth is inside one of them.

    Otherwise a truth that lies outside every drawn contour is indistinguishable
    from one that lies just outside the last, and the picture reads as a worse
    fit than it is -- or a better one.
    """
    reach = sigmas
    if truth is not None:
        deviation = np.sqrt(np.diag(covariance))
        live = deviation > 0.0
        if covariance.shape[0] > 1:
            for i in range(covariance.shape[0]):
                for j in range(i):
                    block = covariance[np.ix_([j, i], [j, i])] + 1e-12 * np.identity(2)
                    offset = np.array([truth[j] - mean[j], truth[i] - mean[i]])
                    reach = max(
                        reach,
                        float(np.sqrt(offset @ np.linalg.solve(block, offset))),
                    )
        elif np.any(live):
            reach = max(reach, float(np.abs(truth[0] - mean[0]) / deviation[0]))
    return np.arange(1, min(15, int(np.ceil(reach))) + 1, dtype=float)


def plot_corner(
    posterior: Any,
    /,
    *,
    prior: Any = None,
    truth: Any = None,
    labels: Sequence[str] | None = None,
    axes: Any = None,
    figsize: tuple[float, float] | None = None,
    sigmas: int = 3,
    width: float = 3.75,
    fill: bool = False,
    colormap: str = "Blues",
    colour: str = "darkblue",
    samples: int = 20000,
    rng: Any = None,
) -> Any:
    """The joint distribution of a measure's components, panel by panel.

    v1's ``plot_corner_distributions``. Marginals on the diagonal, pairwise
    contours below it, and nothing above — which is the point of the shape: a
    corner plot shows every pair once, and the empty half is what tells you so.

    Args:
        posterior: a Gaussian measure, or any measure that can be sampled.
        prior: an optional measure, drawn dotted on the diagonal for
            comparison. Its width is usually what the posterior is to be read
            against.
        truth: the true values, marked on every panel.
        labels: one per component. ``x[i]`` if omitted.
        axes: an ``N x N`` array of axes to draw on. Made if omitted.
        figsize: size of the figure made, when *axes* is not given.
        sigmas: how many contours to draw, before opening up for the truth.
        width: half-width of each panel, in standard deviations.
        fill: shade the contours rather than drawing them as lines.
        colormap, colour: for the filled and unfilled cases respectively.
        samples, rng: for measures that must be sampled.

    Returns:
        The ``N x N`` array of axes.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import chi2

    mean, covariance, draws = moments(posterior, samples=samples, rng=rng)
    size = mean.size
    if size < 2:
        raise ValueError(
            f"A corner plot needs at least two components; this measure has "
            f"{size}. Use plot_densities for one."
        )
    deviation = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    truth_values = None if truth is None else np.atleast_1d(np.asarray(truth, float))
    if truth_values is not None and truth_values.size != size:
        raise ValueError(f"{truth_values.size} true values for {size} components.")
    if labels is None:
        labels = [f"x[{index}]" for index in range(size)]

    prior_summary = None
    if prior is not None:
        prior_mean, prior_covariance, prior_draws = moments(
            prior, samples=samples, rng=rng
        )
        prior_summary = (
            prior_mean,
            np.sqrt(np.clip(np.diag(prior_covariance), 0.0, None)),
            prior_draws,
        )

    if axes is None:
        if figsize is None:
            figsize = (2.6 * size, 2.6 * size)
        _, axes = plt.subplots(size, size, figsize=figsize, squeeze=False)
    axes = np.asarray(axes)

    lower, upper = _span(mean, deviation, truth_values, width)
    levels = _contour_levels(mean, covariance, truth_values, sigmas)

    for row in range(size):
        for column in range(size):
            axis = axes[row, column]
            if column > row:
                axis.set_axis_off()
                continue

            if row == column:
                panel = (float(lower[row]), float(upper[row]))
                _density(
                    axis,
                    panel,
                    mean[row],
                    deviation[row],
                    None if draws is None else draws[:, row],
                    colour=colour,
                    label=None,
                    style="-",
                    fill=fill,
                    width=width,
                    points=500,
                )
                if prior_summary is not None:
                    prior_mean, prior_deviation, prior_draws = prior_summary
                    if prior_deviation[row] > 0.0:
                        twin = axis.twinx()
                        twin.set_yticks([])
                        _density(
                            twin,
                            panel,
                            prior_mean[row],
                            prior_deviation[row],
                            None if prior_draws is None else prior_draws[:, row],
                            colour=_PRIOR_COLOURS[0],
                            label=None,
                            style=":",
                            fill=False,
                            width=width,
                            points=500,
                        )
                axis.set_yticks([])
                if truth_values is not None:
                    axis.axvline(truth_values[row], color="black", lw=1.2, ls="--")
                axis.set_xlim(lower[row], upper[row])
            else:
                horizontal = np.linspace(lower[column], upper[column], 160)
                vertical = np.linspace(lower[row], upper[row], 160)
                mesh_x, mesh_y = np.meshgrid(horizontal, vertical)
                if draws is None:
                    block = covariance[np.ix_([column, row], [column, row])]
                    block = block + 1e-15 * np.trace(block) * np.identity(2)
                    offset = np.stack(
                        [mesh_x - mean[column], mesh_y - mean[row]], axis=-1
                    )
                    inverse = np.linalg.inv(block)
                    field = np.sqrt(
                        np.clip(
                            np.einsum("...i,ij,...j->...", offset, inverse, offset),
                            0.0,
                            None,
                        )
                    )
                    contour_levels = levels
                else:
                    from scipy.stats import gaussian_kde

                    density = gaussian_kde(draws[:, [column, row]].T)(
                        np.vstack([mesh_x.ravel(), mesh_y.ravel()])
                    ).reshape(mesh_x.shape)
                    # Levels chosen so each encloses the same probability the
                    # corresponding sigma contour would, which is what makes a
                    # sampled panel readable beside a Gaussian one.
                    order = np.sort(density.ravel())[::-1]
                    mass = np.cumsum(order) / order.sum()
                    wanted = chi2.cdf(levels**2, df=2)
                    contour_levels = np.array(
                        [order[np.searchsorted(mass, m)] for m in wanted]
                    )
                    field = density
                    contour_levels = np.sort(contour_levels)
                if fill:
                    # The fill shows *density*, in both branches. `field` is a
                    # Mahalanobis distance in the Gaussian one, which grows
                    # away from the mean: filling it directly painted the
                    # Gaussian panels darkest where the measure is least
                    # likely, and the sampled panels darkest where it is most,
                    # so the same argument read opposite ways in one figure.
                    if draws is None:
                        shading = np.exp(-0.5 * field**2)
                        edges = np.sort(np.exp(-0.5 * np.asarray(levels) ** 2))
                    else:
                        shading = field
                        edges = np.sort(contour_levels)
                    peak = float(shading.max())
                    if edges.size and peak > edges[-1]:
                        edges = np.concatenate([edges, [peak]])
                        axis.contourf(
                            mesh_x, mesh_y, shading, levels=edges, cmap=colormap
                        )
                    else:
                        axis.contourf(
                            mesh_x,
                            mesh_y,
                            shading,
                            levels=len(levels),
                            cmap=colormap,
                        )
                else:
                    axis.contour(
                        mesh_x,
                        mesh_y,
                        field,
                        levels=contour_levels,
                        colors=colour,
                        linewidths=1.0,
                    )
                axis.plot(mean[column], mean[row], "o", color=colour, ms=3)
                if truth_values is not None:
                    axis.plot(
                        truth_values[column],
                        truth_values[row],
                        "*",
                        color="black",
                        ms=10,
                    )
                axis.set_xlim(lower[column], upper[column])
                axis.set_ylim(lower[row], upper[row])

            if row == size - 1:
                axis.set_xlabel(labels[column])
            else:
                axis.set_xticklabels([])
            if column == 0 and row > 0:
                axis.set_ylabel(labels[row])
            elif column > 0:
                axis.set_yticklabels([])

    return axes
