"""
The observation model: what the data are, and how uncertain.

A ``ForwardProblem`` is the operator and the data uncertainty, and nothing
else. The prior and the property operator belong to the *estimator*, because
the prior is what selects the method — see DESIGN.md section 18.7. One problem
can then be attacked several ways without being rebuilt.

The data uncertainty may be a probability measure or a convex set. Those are
the two data relations of section 18.1, and they are interconvertible but not
canonically: a Gaussian hardens into an ellipsoid at a chosen chi-squared
level, while the reverse restores detail the set never carried.
"""

from __future__ import annotations

from typing import Any, Sequence

from numpy.random import Generator
from scipy.stats import chi2

from ..algebra.direct_sum import ColumnLinearOperator
from ..algebra.operators import LinearOperator, Operator
from ..algebra.spaces import HilbertSpace
from ..geometry.sets import Subset
from ..probability.base import ProbabilityMeasure
from ..probability.gaussian import GaussianMeasure

__all__ = ["ForwardProblem", "LinearForwardProblem"]


class ForwardProblem:
    """An operator from models to data, with the data's uncertainty."""

    def __init__(
        self,
        forward_operator: Operator,
        /,
        *,
        error: ProbabilityMeasure | Subset | None = None,
    ) -> None:
        """
        Args:
            forward_operator: maps the model space to the data space.
            error: the data uncertainty — a measure on the data space, a convex
                subset of it, or ``None`` for error-free data.
        """
        self._forward_operator = forward_operator
        if error is not None:
            domain = getattr(error, "domain", None)
            if domain is not None and domain != forward_operator.codomain:
                raise ValueError(
                    "The data uncertainty lives on "
                    f"{domain!r}, not on the data space "
                    f"{forward_operator.codomain!r}."
                )
        self._error = error

    @property
    def forward_operator(self) -> Operator:
        """The operator taking a model to its predicted data."""
        return self._forward_operator

    @property
    def model_space(self) -> HilbertSpace:
        """The operator's domain."""
        return self._forward_operator.domain

    @property
    def data_space(self) -> HilbertSpace:
        """The operator's codomain."""
        return self._forward_operator.codomain

    @property
    def has_error(self) -> bool:
        """True unless the data are taken to be exact."""
        return self._error is not None

    @property
    def error(self) -> ProbabilityMeasure | Subset:
        """The data uncertainty, as given."""
        if self._error is None:
            raise AttributeError("This problem has no data uncertainty.")
        return self._error

    @property
    def error_measure(self) -> ProbabilityMeasure:
        """The data uncertainty, when it is a measure.

        Raises:
            AttributeError: if the problem has no uncertainty at all.
            TypeError: if it has one but it is a *set*. That is not a missing
                attribute, it is a different kind of problem: bounded errors
                are handled by the Backus-Gilbert routes, which give a range
                of admissible values rather than a distribution over them.
                Saying so by type, and by name, saves the caller working out
                which of the two situations they are in.
        """
        if self._error is None:
            raise AttributeError("This problem has no data uncertainty.")
        if not isinstance(self._error, ProbabilityMeasure):
            raise TypeError(
                "This method needs a Gaussian error measure; the problem's "
                f"uncertainty is a set ({type(self._error).__name__}). Bounded "
                "errors are handled by the Backus-Gilbert routes -- see "
                "pygeoinf2.inference.backus."
            )
        return self._error

    @property
    def error_set(self) -> Subset:
        """The data uncertainty, when it is a set.

        Raises:
            AttributeError: if the problem has no uncertainty at all.
            TypeError: if it has one but it is a measure. The counterpart of
                :attr:`error_measure`, and refused the same way.
        """
        if self._error is None:
            raise AttributeError("This problem has no data uncertainty.")
        if not isinstance(self._error, Subset):
            raise TypeError(
                "This method needs a set-valued error; the problem's "
                f"uncertainty is a measure ({type(self._error).__name__}). Use "
                "the Gaussian routes -- see pygeoinf2.inference.gaussian."
            )
        return self._error

    def __repr__(self) -> str:
        kind = "exact"
        if isinstance(self._error, ProbabilityMeasure):
            kind = "measure"
        elif isinstance(self._error, Subset):
            kind = "set"
        return (
            f"{type(self).__name__}({self.model_space!r} -> "
            f"{self.data_space!r}, error={kind})"
        )

    # ----------------------------------------------------------------- #
    #                          Data and misfit                          #
    # ----------------------------------------------------------------- #

    def synthetic_data(self, model: Any, /, *, rng: Generator | None = None) -> Any:
        """Data predicted from a model, plus one draw of the error."""
        predicted = self._forward_operator(model)
        if not self.has_error:
            return predicted
        return self.data_space.add(predicted, self.error_measure.sample(rng=rng))

    def chi_squared(self, model: Any, data: Any, /) -> float:
        """The weighted misfit of a model against data.

        With a Gaussian error this is the Mahalanobis distance of the residual;
        without an error measure it is the squared norm.
        """
        residual = self.data_space.subtract(data, self._forward_operator(model))
        return self.chi_squared_from_residual(residual)

    def chi_squared_from_residual(self, residual: Any, /) -> float:
        """The weighted misfit of a residual."""
        if not self.has_error:
            return self.data_space.squared_norm(residual)
        measure = self.error_measure
        if not isinstance(measure, GaussianMeasure):
            raise TypeError(
                "A chi-squared statistic needs a Gaussian error measure; this "
                f"one is a {type(measure).__name__}."
            )
        centred = self.data_space.subtract(residual, measure.expectation)
        return measure.mahalanobis_squared(
            self.data_space.add(centred, measure.expectation)
        )

    def critical_chi_squared(self, /, *, level: float = 0.95) -> float:
        """The chi-squared threshold at a given confidence level."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"A confidence level lies in (0, 1), got {level}.")
        return float(chi2.ppf(level, self.data_space.dim))

    def chi_squared_test(
        self, model: Any, data: Any, /, *, level: float = 0.95
    ) -> bool:
        """Whether a model is compatible with the data at a confidence level.

        The boolean on top of :meth:`consistency_set`; the set is the object,
        and this is the question people usually ask of it (§18.11).
        """
        return self.chi_squared(model, data) < self.critical_chi_squared(level=level)

    def consistency_set(self, model: Any, /, *, level: float = 0.95) -> Subset:
        """The data compatible with a model, as an ellipsoid in the data space.

        This is the hardening of §18.1: a Gaussian error measure at a chosen
        chi-squared level. It discards information and says so by being a
        separate, named step rather than something a constructor does quietly.
        """
        return self.error_measure.credible_set(level=level).translate(
            self._forward_operator(model)
        )


class LinearForwardProblem(ForwardProblem):
    """A forward problem whose operator is linear."""

    def __init__(
        self,
        forward_operator: LinearOperator,
        /,
        *,
        error: ProbabilityMeasure | Subset | None = None,
    ) -> None:
        if not isinstance(forward_operator, LinearOperator):
            raise TypeError(
                "A LinearForwardProblem needs a LinearOperator; got a "
                f"{type(forward_operator).__name__}."
            )
        super().__init__(forward_operator, error=error)

    @staticmethod
    def from_direct_sum(
        problems: Sequence[LinearForwardProblem], /
    ) -> LinearForwardProblem:
        """One problem from several observing a common model.

        The joint inversion: the data space becomes the direct sum, and the
        errors are taken to be independent between data sets.

        Either every problem carries an error measure or none does. A mixture
        of the two is refused rather than resolved: the only way to combine
        them is to drop the errors that exist, which would make noisy data look
        exact and quietly overweight them in every inversion built on the
        result.

        Args:
            problems: the problems to join. They must share a model space.

        Returns:
            The joint problem, on the direct sum of the data spaces.

        Raises:
            ValueError: if no problems are given, if they do not share a model
                space, or if some carry an error measure and others do not.
        """
        problems = tuple(problems)
        if not problems:
            raise ValueError("At least one problem is needed.")
        model_space = problems[0].model_space
        if any(problem.model_space != model_space for problem in problems):
            raise ValueError("The problems must share a model space.")

        operator = ColumnLinearOperator(
            [problem.forward_operator for problem in problems]
        )
        carries_error = [problem.has_error for problem in problems]
        if all(carries_error):
            from ..probability.gaussian import GaussianMeasure as Gaussian

            error = Gaussian.from_product(
                [problem.error_measure for problem in problems]
            )
        elif any(carries_error):
            without = [i for i, has in enumerate(carries_error) if not has]
            raise ValueError(
                f"Every problem needs an error measure, or none may: "
                f"problems {without} have none while the others do. Joining "
                f"them can only be done by discarding the errors that exist, "
                f"which would present noisy data as exact. Give the exact "
                f"members an error measure — a small one says 'exact' without "
                f"lying — or build the joint problem without errors."
            )
        else:
            error = None
        return LinearForwardProblem(operator, error=error)

    def data_measure_from_model(self, model: Any, /) -> GaussianMeasure:
        """The distribution of data predicted by one model."""
        return self.error_measure.translate(self._forward_operator(model))

    def data_measure_from_model_measure(
        self, measure: ProbabilityMeasure, /
    ) -> ProbabilityMeasure:
        """The distribution of data induced by a distribution of models."""
        if measure.domain != self.model_space:
            raise ValueError("The measure is not defined on the model space.")
        pushed = measure.push_forward(self._forward_operator)
        return pushed + self.error_measure if self.has_error else pushed

    def joint_measure(self, measure: ProbabilityMeasure, /) -> ProbabilityMeasure:
        """The joint distribution of model and data, on their direct sum.

        Sampling this is how a consistent ``(model, data)`` pair is made: the
        two are drawn together rather than the data being predicted from an
        already-drawn model and given an independent error afterwards, which
        would be the same thing only by accident.
        """
        model_space, data_space = self.model_space, self.data_space
        if not self.has_error:
            return measure.push_forward(
                ColumnLinearOperator(
                    [LinearOperator.identity(model_space), self._forward_operator]
                )
            )

        from ..algebra.direct_sum import BlockLinearOperator
        from ..probability.gaussian import GaussianMeasure as Gaussian

        block = BlockLinearOperator(
            [
                [
                    LinearOperator.identity(model_space),
                    LinearOperator.zero(data_space, codomain=model_space),
                ],
                [self._forward_operator, LinearOperator.identity(data_space)],
            ]
        )
        joint = Gaussian.from_product([measure, self.error_measure])
        return joint.push_forward(block)

    def synthetic_model_and_data(
        self, prior: ProbabilityMeasure, /, *, rng: Generator | None = None
    ) -> tuple[Any, Any]:
        """A model drawn from the prior, and data consistent with it."""
        sample = self.joint_measure(prior).sample(rng=rng)
        return sample[0], sample[1]

    # ----------------------------------------------------------------- #
    #                        Transformed problems                       #
    # ----------------------------------------------------------------- #

    def parameterised(
        self, parameterisation: LinearOperator, /
    ) -> LinearForwardProblem:
        """The same data, seen through a restricted model space."""
        if parameterisation.codomain != self.model_space:
            raise ValueError("The parameterisation must map into the model space.")
        return LinearForwardProblem(
            self._forward_operator @ parameterisation, error=self._error
        )

    def data_reduced(
        self,
        reduction: LinearOperator,
        /,
        *,
        error: ProbabilityMeasure | Subset | None = None,
    ) -> LinearForwardProblem:
        """The same model, seen through fewer or combined data.

        Args:
            reduction: maps the data space to the reduced one.
            error: the reduced error. Defaults to pushing the current one
                forward, which is right for a measure and is the only thing
                that can be done automatically.
        """
        if reduction.domain != self.data_space:
            raise ValueError("The reduction must map from the data space.")
        if error is None and self.has_error:
            error = self.error_measure.push_forward(reduction)
        return LinearForwardProblem(reduction @ self._forward_operator, error=error)
