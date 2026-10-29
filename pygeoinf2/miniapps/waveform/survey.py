"""The experiment: a grid, a survey, a true model, a prior and the data.

A two-dimensional section 2 km wide and 0.8 km deep, its top a free surface;
shots fired a couple of cells below the surface and recorded by a line of
receivers at the same depth across the section, every step of a one-second
record. The unknown is ``m = 1 / c^2``, the slowness squared, in s²/km²: a
velocity increasing linearly with depth and three Gaussian anomalies, two
fast and one slow. The background is known and is the prior's expectation;
the anomalies are what the data are asked for.

The grid is bigger than the section: a sponge lies beyond it on the left,
the right and below, where the waves that leave the section are damped
before the grid's edge can send them back. The model space is the whole
grid. Below the section the velocity stops increasing, leveling off within
a fifth of a kilometer, so that the rock under the sponge is no faster than
the time step allows for the prior's draws.

Three sizes. ``PROOF`` is the small grid the adjoints are proved on, a
sliver of section inside its sponge. ``QUICK`` is the section at a 20 m
spacing and a 2.5 Hz wavelet, ``FULL`` at a 10 m spacing and a 5 Hz wavelet:
in both, the shortest wavelength that matters (at two and a half times the
peak frequency, in the slowest rock) spans fifteen cells or more.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from pygeoinf2 import GaussianMeasure, LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.symmetric_space.fourier import Sobolev

try:
    import acoustic
except ImportError:  # the built module sits next to this file
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    try:
        import acoustic
    finally:
        sys.path.pop(0)

# The stability limit of the scheme, c dt / h, for the fourth-order Laplacian
# in two dimensions: dt^2 c^2 (32 / 3) / h^2 <= 4.
CFL_LIMIT = float(np.sqrt(3.0 / 8.0))
# The sponge: its width in cells and the rate of its Cerjan taper,
# exp(-(rate (width - d))^2) at d cells from the grid's edge, 0.964 at the
# edge. Measured against a grid too big to reflect, it sends back about half
# a percent of the data's rms at both scales; it is as wide as the full
# scale's memory allows.
SPONGE = 24
SPONGE_RATE = 0.008
DEPTH_CELLS = 2

# The background velocity, km/s, increasing linearly with depth to the
# section's bottom and leveling off below it over a length, km; and the
# anomalies: (x, z, radius, dc), km and km/s, Gaussian in the velocity, x
# from the section's left edge.
SURFACE_VELOCITY = 1.8
VELOCITY_GRADIENT = 0.6
SECTION_DEPTH = 0.8
LEVELING_LENGTH = 0.2
ANOMALIES = (
    (0.70, 0.25, 0.08, +0.25),
    (1.25, 0.32, 0.10, -0.20),
    (0.98, 0.46, 0.07, +0.20),
)
# The properties asked of the posterior: the mean of m over boxes
# (x0, x1, z0, z1), km. One over each anomaly and one over plain background.
BOXES = (
    ("fast, shallow", (0.60, 0.80, 0.15, 0.35)),
    ("slow", (1.13, 1.37, 0.20, 0.44)),
    ("fast, deep", (0.90, 1.06, 0.38, 0.54)),
    ("background", (0.45, 0.65, 0.40, 0.55)),
)

# The prior: m - m0 has the identity covariance on the model space, the
# Sobolev space of this order and length scale, scaled to this pointwise
# standard deviation.
ORDER = 2.0
LENGTH_SCALE = 0.1
PRIOR_STD = 0.02
NOISE_FRACTION = 0.03
LEVEL = 0.95


@dataclass(frozen=True)
class Scale:
    """One size of the problem, and how hard its solvers may work.

    ``nx`` and ``nz`` count the cells of the section; the grid adds
    ``sponge`` cells on the left, the right and below.
    """

    name: str
    nx: int
    nz: int
    h: float
    dt: float
    steps: int
    peak_frequency: float
    shots: tuple[float, ...]
    receiver_spacing: int
    sponge: int = SPONGE
    sponge_rate: float = SPONGE_RATE
    newton_iterations: int = 10
    exact_iterations: int = 10
    polish_iterations: int = 2
    cg_iterations: int = 40
    polish_cg_iterations: int = 200
    laplace_cg_iterations: int = 300
    laplace_rank: int = 60
    occam_iterations: int = 10
    occam_newton_iterations: int = 6
    occam_cg_iterations: int = 40
    n_jobs: int | None = None
    curvature_jobs: int | None = None


PROOF = Scale(
    name="proof",
    nx=24,
    nz=12,
    h=0.01,
    dt=0.0018,
    steps=300,
    peak_frequency=5.0,
    shots=(0.06, 0.18),
    receiver_spacing=2,
    # A narrower sponge than the others: the identities hold for any taper.
    sponge=20,
    sponge_rate=0.015,
)
QUICK = Scale(
    name="quick",
    nx=100,
    nz=40,
    h=0.02,
    dt=0.0036,
    steps=400,
    peak_frequency=2.5,
    shots=(0.6, 1.0, 1.4),
    receiver_spacing=2,
    n_jobs=3,
    curvature_jobs=3,
)
FULL = Scale(
    name="full",
    nx=200,
    nz=80,
    h=0.01,
    dt=0.0018,
    steps=600,
    peak_frequency=5.0,
    shots=(0.5, 1.0, 1.5),
    receiver_spacing=3,
    newton_iterations=10,
    exact_iterations=3,
    polish_cg_iterations=60,
    laplace_cg_iterations=200,
    laplace_rank=100,
    occam_cg_iterations=25,
    n_jobs=3,
    # One linearized field at a time beside the three forward fields: four
    # stored fields of 124 MB.
    curvature_jobs=1,
)


def ricker(times: np.ndarray, peak_frequency: float) -> np.ndarray:
    """The Ricker wavelet of a peak frequency, delayed by 1.2 periods."""
    arg = (np.pi * peak_frequency * (times - 1.2 / peak_frequency)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


@dataclass
class Survey:
    """A grid, its sponge, the shots, the receivers and the wavelet.

    The grid holds the section and, beyond it on the left, the right and
    below, ``pad`` cells of the same medium and then the sponge; ``pad`` is
    zero but for a reference run on a grid too big to reflect within the
    record. Positions ``x`` run from the section's left edge, ``z`` down from
    the surface, both in km.

    ``model`` is the C++ solver, from ``backend``: the module ``acoustic`` by
    default, or ``acoustic_mpi``, whose model splits the grid across the ranks
    of an MPI run.
    """

    scale: Scale
    backend: Any = None
    pad: int = 0
    model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        s = self.scale
        backend = acoustic if self.backend is None else self.backend
        self.model = backend.Model(
            *self.shape,
            s.h,
            s.dt,
            s.steps,
            s.sponge,
            s.sponge_rate,
            self.receiver_x.tolist(),
            [DEPTH_CELLS] * self.receiver_x.size,
        )

    @property
    def margin(self) -> int:
        """The cells beyond the section on each absorbing side."""
        return self.pad + self.scale.sponge

    @property
    def shape(self) -> tuple[int, int]:
        """The grid's cells, ``(nx, nz)``."""
        return (self.scale.nx + 2 * self.margin, self.scale.nz + self.margin)

    @property
    def lengths(self) -> tuple[float, float]:
        """The grid's width and depth, km."""
        return (self.shape[0] * self.scale.h, self.shape[1] * self.scale.h)

    @property
    def section_lengths(self) -> tuple[float, float]:
        """The section's width and depth, km."""
        return (self.scale.nx * self.scale.h, self.scale.nz * self.scale.h)

    @property
    def window(self) -> tuple[slice, slice]:
        """The section's cells within the grid."""
        return (
            slice(self.margin, self.margin + self.scale.nx),
            slice(0, self.scale.nz),
        )

    @cached_property
    def x(self) -> np.ndarray:
        """The grid's columns, km from the section's left edge."""
        return (np.arange(self.shape[0]) - self.margin) * self.scale.h

    @cached_property
    def z(self) -> np.ndarray:
        """The grid's rows, km below the surface."""
        return np.arange(self.shape[1]) * self.scale.h

    @cached_property
    def times(self) -> np.ndarray:
        """The time of each recorded sample: step n records u^{n+1}."""
        return (np.arange(self.scale.steps) + 1) * self.scale.dt

    @cached_property
    def receiver_x(self) -> np.ndarray:
        """Receiver columns: every few cells across the section."""
        return self.margin + np.arange(0, self.scale.nx, self.scale.receiver_spacing)

    @cached_property
    def source_x(self) -> np.ndarray:
        """Source columns, the nearest to the asked positions."""
        shots = np.rint(np.asarray(self.scale.shots) / self.scale.h).astype(int)
        return self.margin + shots

    @property
    def n_shots(self) -> int:
        return self.source_x.size

    @property
    def n_receivers(self) -> int:
        return self.receiver_x.size

    @property
    def data_shape(self) -> tuple[int, int, int]:
        return (self.n_shots, self.scale.steps, self.n_receivers)

    @property
    def data_size(self) -> int:
        return int(np.prod(self.data_shape))

    @cached_property
    def wavelet(self) -> np.ndarray:
        """The source time function at the times the steps inject it."""
        return ricker(
            np.arange(self.scale.steps) * self.scale.dt, self.scale.peak_frequency
        )

    @property
    def field_megabytes(self) -> float:
        """The size of one stored field, all steps of the grid."""
        return self.scale.steps * self.shape[0] * self.shape[1] * 8 / 1e6

    def interior(self) -> np.ndarray:
        """The cells of the section below the surface row."""
        mask = np.zeros(self.shape, dtype=bool)
        mask[self.window] = True
        mask[:, 0] = False
        return mask

    def boxes_inside(self) -> bool:
        """Whether every box of :data:`BOXES` lies inside the section."""
        width, depth = self.section_lengths
        return all(
            0.0 <= x0 and x1 <= width and z1 <= depth for _, (x0, x1, _, z1) in BOXES
        )

    def cfl(self, m: np.ndarray) -> float:
        """``c_max dt / h`` for a model."""
        return float(self.scale.dt / (np.sqrt(np.min(m)) * self.scale.h))


def background_velocity(survey: Survey) -> np.ndarray:
    """The known background: a velocity increasing linearly with depth.

    Below the section it levels off, smoothly: the depth it sees there is
    ``SECTION_DEPTH + L tanh((z - SECTION_DEPTH) / L)``, which meets the line
    with its slope and its curvature.
    """
    _, z = np.meshgrid(survey.x, survey.z, indexing="ij")
    below = np.maximum(z - SECTION_DEPTH, 0.0)
    seen = np.minimum(z, SECTION_DEPTH) + LEVELING_LENGTH * np.tanh(
        below / LEVELING_LENGTH
    )
    return SURFACE_VELOCITY + VELOCITY_GRADIENT * seen


def true_velocity(survey: Survey) -> np.ndarray:
    """The background and the three anomalies."""
    x, z = np.meshgrid(survey.x, survey.z, indexing="ij")
    velocity = background_velocity(survey)
    for x0, z0, radius, dc in ANOMALIES:
        velocity = velocity + dc * np.exp(-((x - x0) ** 2 + (z - z0) ** 2) / radius**2)
    return velocity


def box_weights(survey: Survey, box: tuple[float, float, float, float]) -> np.ndarray:
    """Equal weights on the cells of a box, summing to one."""
    x, z = np.meshgrid(survey.x, survey.z, indexing="ij")
    x0, x1, z0, z1 = box
    inside = (x >= x0) & (x <= x1) & (z >= z0) & (z <= z1)
    if not inside.any():
        raise ValueError(f"The box {box} holds no cell of the grid.")
    return inside / inside.sum()


def box_average_operator(space: Sobolev, survey: Survey) -> LinearOperator:
    """The mean of m over each of :data:`BOXES`, into Euclidean space.

    A weighted sum of grid values; its adjoint is the representer of that
    sum in the model space's metric, taken from the derivative's components.
    """
    weights = np.stack([box_weights(survey, box) for _, box in BOXES])
    codomain = EuclideanSpace(len(BOXES))
    scale = weights[0].size / space.volume

    def value(m: np.ndarray) -> np.ndarray:
        return np.tensordot(weights, np.asarray(m), axes=2)

    def derivative_components(y: np.ndarray) -> np.ndarray:
        return scale * space.to_components(np.tensordot(y, weights, axes=1))

    return LinearOperator.from_derivative_callables(
        space, codomain, value, derivative_components
    )


@dataclass
class Experiment:
    """Everything ``build`` makes: the survey, the spaces, the prior, the truth."""

    survey: Survey
    space: Sobolev
    background: np.ndarray
    truth: np.ndarray
    prior: GaussianMeasure
    probes: GaussianMeasure

    def relative_error(self, model: np.ndarray) -> float:
        """``|m - truth| / |m0 - truth|`` over the section: one at the background."""
        inside = self.survey.interior()
        miss = np.asarray(model)[inside] - self.truth[inside]
        anomaly = self.background[inside] - self.truth[inside]
        return float(np.linalg.norm(miss) / np.linalg.norm(anomaly))


def model_space(survey: Survey) -> Sobolev:
    """The Sobolev box over the whole grid: vectors are (nx, nz) arrays of m."""
    return Sobolev(survey.shape, ORDER, LENGTH_SCALE, lengths=survey.lengths)


def build(scale: Scale = FULL, /, *, backend: Any = None) -> Experiment:
    """The survey, the model space, the background, the truth and the prior.

    Args:
        scale: the size of the problem.
        backend: the module whose ``Model`` solves the wave equation:
            ``acoustic`` if omitted, or ``acoustic_mpi``.
    """
    survey = Survey(scale, backend)
    space = model_space(survey)
    background = 1.0 / background_velocity(survey) ** 2
    truth = 1.0 / true_velocity(survey) ** 2
    probes = space.sobolev_measure(0.0, LENGTH_SCALE, pointwise_std=PRIOR_STD)
    prior = space.sobolev_measure(
        0.0, LENGTH_SCALE, pointwise_std=PRIOR_STD, expectation=background
    )
    return Experiment(survey, space, background, truth, prior, probes)
