"""The estimators' companions: a logging Newton-CG and the saved mode.

The Gauss-Newton runs ask ``MaximumAPosteriori`` and ``Occam`` for the
Gauss-Newton Hessian (``gauss_newton=True``), ``2 (P + J* R^-1 J)`` on the
objective and ``2 J* R^-1 J`` scaled as Occam scales its misfit constraint,
since the forward operator is used without its second derivative there.

:class:`LoggedNewtonCG` is the library's Newton-CG with a record of the value
and the gradient norm at every point where it linearizes the objective, which
is once per iteration. :func:`map_mode` runs the Gauss-Newton MAP that part C
starts with, or reads it back from the file part C writes; part C's script,
``newton.py``, is beside this package, where a script run from the app finds
it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from typing import Any

import numpy as np

from pygeoinf import Functional, QuadraticModel
from pygeoinf.numerics.optimization import NewtonCG

from .survey import FULL, PROOF, QUICK, Experiment, Scale


class LoggedNewtonCG(NewtonCG):
    """Newton-CG keeping ``(value, gradient norm)`` at each linearization."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.log: list[tuple[float, float]] = []

    def minimize(self, functional: Functional, x0: Any, /) -> Any:
        space = functional.domain
        log = self.log

        def linearize(model: Any) -> QuadraticModel:
            local = functional.at(model)
            log.append((float(local.value), float(space.norm(local.gradient))))
            return local

        logged = Functional.from_callables(
            space,
            functional,
            linearize=linearize,
            hessian=functional.hessian if functional.has_hessian else None,
        )
        return super().minimize(logged, x0)


def newton(
    experiment: Experiment,
    scale: Scale,
    /,
    *,
    iterations: int | None = None,
    forcing: float = 0.1,
    ftol: float = 1e-5,
    cg_iterations: int | None = None,
) -> LoggedNewtonCG:
    """Newton-CG as the parts use it, preconditioned by the prior covariance.

    It stops on the iteration cap, or when an iteration lowers the objective
    by less than ``ftol`` of it; the default is about one unit of chi-squared
    here, far below its spread under the noise, ``sqrt(2 n)``.
    """
    return LoggedNewtonCG(
        max_iterations=scale.newton_iterations if iterations is None else iterations,
        max_cg_iterations=(
            scale.cg_iterations if cg_iterations is None else cg_iterations
        ),
        preconditioner=experiment.prior.covariance,
        forcing=forcing,
        rtol=0.0,
        gtol=0.0,
        ftol=ftol,
    )


def modes_path(outdir: pathlib.Path, scale: Scale, /) -> pathlib.Path:
    """Where part C leaves its modes for parts D and E.

    Named by the preset and the seed; a scale that departs from its preset
    in anything else, by an option given on the command line, adds a digest
    of itself, so that a mode is never read back under another problem.
    """
    stem = f"modes_{scale.name}_{scale.seed}"
    preset = {s.name: s for s in (PROOF, QUICK, FULL)}.get(scale.name)
    if preset is None or dataclasses.replace(scale, seed=preset.seed) != preset:
        stem += "_" + hashlib.blake2b(repr(scale).encode(), digest_size=4).hexdigest()
    return outdir / f"{stem}.npz"


def map_mode(state: Any, /, *, outdir: pathlib.Path) -> np.ndarray:
    """The polished MAP mode of part C: read back, or recomputed and saved."""
    experiment = state.experiment
    scale = experiment.survey.scale
    path = modes_path(outdir, scale)
    if path.exists():
        print(f"the MAP mode read from {path.name}")
        return np.load(path)["polish"]
    print(f"{path.name} not found; running part C's Newton iterations for the mode")
    import newton as part_c

    return part_c.map_answer(state, outdir=outdir, figure=False)["polish"]["model"]
