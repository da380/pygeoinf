"""The estimators' companions: Gauss-Newton variants, a logging Newton-CG, the saved mode.

``MaximumAPosteriori`` and ``Occam`` carry the exact Hessian of their
objectives when the forward operator has a second derivative and no Hessian
otherwise, so a Newton method on the Gauss-Newton Hessian needs the objective
to carry that Hessian itself. :class:`GaussNewtonMAP` attaches the
estimator's own :meth:`~pygeoinf2.inference.laplace.MaximumAPosteriori.hessian`,
which for a forward operator without a second derivative is
``2 (P + J* R^-1 J)``; :class:`GaussNewtonOccam` attaches ``2 J* R^-1 J``,
scaled as Occam scales its misfit constraint.

:class:`LoggedNewtonCG` is the library's Newton-CG with a record of the value
and the gradient norm at every point where it linearizes the objective, which
is once per iteration. :func:`map_mode` runs the Gauss-Newton MAP that part C
starts with, or reads it back from the file part C writes.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from pygeoinf2 import Functional, MaximumAPosteriori, Occam, QuadraticModel
from pygeoinf2.numerics.optimization import NewtonCG, gauss_newton_hessian

from survey import Experiment, Scale


class GaussNewtonMAP(MaximumAPosteriori):
    """The MAP estimator whose objective carries the Gauss-Newton Hessian.

    Built on a forward problem whose operator has no second derivative, for
    which :meth:`hessian` is ``2 (P + J* R^-1 J)``.
    """

    def objective(self, data: Any, /) -> Functional:
        base = super().objective(data)

        def linearize(model: Any) -> QuadraticModel:
            local = base.at(model)
            return QuadraticModel(
                model, local.value, local.derivative, self.hessian(data, model)
            )

        return Functional.from_callables(
            self.model_space,
            base,
            linearize=linearize,
            hessian=lambda model: self.hessian(data, model),
        )


class GaussNewtonOccam(Occam):
    """Occam's inversion whose misfit constraint carries the Gauss-Newton Hessian.

    The constraint is ``chi^2 / chi^2_crit - 1``; its Gauss-Newton Hessian is
    ``2 J* R^-1 J / chi^2_crit``. Built on a forward problem whose operator has
    no second derivative.
    """

    def misfit_constraint(self, data: Any, /) -> Functional:
        base = super().misfit_constraint(data)
        problem = self.forward_problem
        weight = 2.0 / self.target

        def hessian(model: Any) -> Any:
            return weight * gauss_newton_hessian(
                problem.forward_operator,
                model,
                weighting=problem.error_measure.precision,
            )

        def linearize(model: Any) -> QuadraticModel:
            local = base.at(model)
            return QuadraticModel(model, local.value, local.derivative, hessian(model))

        return Functional.from_callables(
            self.model_space, base, linearize=linearize, hessian=hessian
        )


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


def modes_path(outdir: pathlib.Path, scale: Scale, seed: int) -> pathlib.Path:
    """Where part C leaves its modes for part D."""
    return outdir / f"modes_{scale.name}_{seed}.npz"


def map_mode(state: Any, /, *, outdir: pathlib.Path, seed: int) -> np.ndarray:
    """The polished MAP mode of part C: read back, or recomputed and saved."""
    experiment = state.experiment
    scale = experiment.survey.scale
    path = modes_path(outdir, scale, seed)
    if path.exists():
        print(f"the MAP mode read from {path.name}")
        return np.load(path)["polish"]
    print(f"{path.name} not found; running part C's Newton iterations for the mode")
    import newton as part_c

    return part_c.map_answer(state, outdir=outdir, seed=seed, figure=False)["polish"][
        "model"
    ]
