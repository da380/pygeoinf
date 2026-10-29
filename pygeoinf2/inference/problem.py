"""
The observation model: what the data are, and how uncertain.

A ``ForwardProblem`` is the operator and the data uncertainty, and nothing
else. The prior and the property operator belong to the *estimator*, because
the prior is what selects the method. One problem can then be attacked several
ways without being rebuilt.

The data uncertainty may be a probability measure or a convex set. The two are
interconvertible, but not canonically: a Gaussian hardens into an ellipsoid at
a chosen chi-squared level, while going the other way invents detail the set
never carried.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from numpy.random import Generator

from ..algebra.direct_sum import ColumnLinearOperator
from ..algebra.operators import LinearOperator, Operator
from ..algebra.spaces import HilbertSpace
from ..geometry.sets import Subset
from ..probability.base import ProbabilityMeasure
from ..probability.gaussian import GaussianMeasure

if TYPE_CHECKING:  # pragma: no cover
    from ..algebra.operators import Functional

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
        """The data uncertainty, as given -- a measure or a set.

        Returns:
            Whichever was supplied. Use :attr:`error_measure` or
            :attr:`error_set` when only one kind will do; each says by type
            which situation it is refusing.

        Raises:
            ValueError: if the problem has no uncertainty at all.
        """
        if self._error is None:
            raise ValueError("This problem has no data uncertainty.")
        return self._error

    @property
    def error_measure(self) -> ProbabilityMeasure:
        """The data uncertainty, when it is a measure.

        Raises:
            ValueError: if the problem has no uncertainty at all.
            TypeError: if it has one but it is a *set*. That is not a missing
                ingredient, it is a different kind of problem: bounded errors
                are handled by the Backus-Gilbert routes, which give a range
                of admissible values rather than a distribution over them.
                Saying so by type, and by name, saves the caller working out
                which of the two situations they are in.
        """
        if self._error is None:
            raise ValueError("This problem has no data uncertainty.")
        if not isinstance(self._error, ProbabilityMeasure):
            raise TypeError(
                "This method needs a measure-valued error; the problem's "
                f"uncertainty is a set ({type(self._error).__name__}). Bounded "
                "errors are handled by the Backus-Gilbert routes -- see "
                "pygeoinf2.inference.backus."
            )
        return self._error

    @property
    def error_set(self) -> Subset:
        """The data uncertainty, when it is a set.

        Raises:
            ValueError: if the problem has no uncertainty at all.
            TypeError: if it has one but it is a measure. The counterpart of
                :attr:`error_measure`, and refused the same way.
        """
        if self._error is None:
            raise ValueError("This problem has no data uncertainty.")
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

        With a Gaussian error this is the squared Mahalanobis distance of the
        residual; without an error measure it is the squared norm. It is
        computed on a data space that discretizes a function space as on any
        other: a misfit is a potential, not a density on the data space. Its
        value there depends on the discretization, and a limit under
        refinement needs the error measure to be trace class, which is the
        caller's to ensure.
        """
        residual = self.data_space.subtract(data, self._forward_operator(model))
        return self.chi_squared_from_residual(residual)

    def chi_squared_from_residual(self, residual: Any, /) -> float:
        """The weighted misfit of a residual.

        Args:
            residual: the difference between predicted and observed data.

        Returns:
            The squared Mahalanobis distance of the residual under the error
            measure, or its plain squared norm when the data are taken as
            exact.

        Raises:
            TypeError: if the uncertainty is a set rather than a Gaussian
                measure. A chi-squared statistic needs a distribution.
        """
        if not self.has_error:
            return self.data_space.squared_norm(residual)
        measure = self.error_measure
        if not isinstance(measure, GaussianMeasure):
            raise TypeError(
                "A chi-squared statistic needs a Gaussian error measure; this "
                f"one is a {type(measure).__name__}."
            )
        # unsafe=True passes the gate that refuses densities on function
        # spaces: a misfit is a likelihood potential, not a density on the
        # data space, and it is well defined whenever the residual lies in
        # the error's Cameron-Martin space. The measure subtracts its own
        # expectation.
        return measure.mahalanobis_squared(residual, unsafe=True)

    def critical_chi_squared(self, /, *, level: float = 0.95) -> float:
        """The chi-squared threshold at a given confidence level.

        Args:
            level: the confidence level, in ``(0, 1)``.

        Returns:
            The critical value, on as many degrees of freedom as there are
            data.

        Raises:
            ValueError: for a level outside ``(0, 1)``.
        """
        # Imported here rather than at module scope: `scipy.stats` is slow to
        # import, and this is the only thing in the file that wants it.
        from scipy.stats import chi2

        if not 0.0 < level < 1.0:
            raise ValueError(f"A confidence level lies in (0, 1), got {level}.")
        return float(chi2.ppf(level, self.data_space.dim))

    def chi_squared_test(
        self, model: Any, data: Any, /, *, level: float = 0.95
    ) -> bool:
        """Whether a model is compatible with the data at a confidence level.

        The boolean on top of :meth:`consistency_set`; the set is the object,
        and this is the question usually asked of it.

        Args:
            model: the model to test.
            data: the observations.
            level: the confidence level.

        Returns:
            Whether the misfit falls below the critical value.
        """
        return self.chi_squared(model, data) < self.critical_chi_squared(level=level)

    def potential(self, data: Any, /, *, temperature: float = 1.0) -> "Functional":
        """The likelihood potential ``Phi(m) == chi^2(m, d) / (2 T)``.

        The negative log likelihood up to its constant, which is the object a
        function-space sampler and a tempering schedule consume: the
        posterior's density relative to the prior is ``exp(-Phi)`` up to the
        evidence. Where the forward operator has a derivative, the potential
        is built from one linearization per point serving the value and the
        derivative together, so a caller asking for both pays for one forward
        evaluation, not two. As for :meth:`chi_squared`, on a data space that
        discretizes a function space a limit under refinement needs the error
        measure to be trace class.

        Args:
            data: the observations.
            temperature: divides the potential. One is the likelihood; larger
                flattens it, which is what tempering and annealed schedules
                sweep.

        Returns:
            The potential, a functional on the model space with a derivative
            wherever the forward operator has one.

        Raises:
            ValueError: for a non-positive temperature, or when the problem
                has no data uncertainty: exact data have no likelihood, and a
                posterior conditioned on them is singular with respect to
                the prior, with no density against it.
            TypeError: for a set-valued data uncertainty, which has no
                likelihood.
        """
        from ..algebra.linearization import QuadraticModel
        from ..algebra.operators import Functional, LinearFunctional

        if temperature <= 0.0:
            raise ValueError(f"A temperature is positive, got {temperature}.")
        if self._error is None:
            raise ValueError(
                "The problem has no data uncertainty, so there is no "
                "likelihood and no potential: exact data condition the prior "
                "onto the models that fit them, which has no density against "
                "it. Give the problem an error measure, or use chi_squared "
                "for the misfit itself."
            )
        if not isinstance(self._error, ProbabilityMeasure):
            raise TypeError(
                "A potential needs a measure-valued data uncertainty; a set "
                "has no likelihood. Use the feasibility machinery instead."
            )

        space = self.model_space
        data_space = self.data_space
        forward = self._forward_operator
        error = self.error_measure
        scale = 1.0 / (2.0 * temperature)

        def value(model: Any) -> float:
            return scale * self.chi_squared(model, data)

        def linearize(model: Any) -> QuadraticModel:
            linearization = forward.at(model)
            residual = data_space.subtract(data, linearization.value)
            weighted = error.precision(data_space.subtract(residual, error.expectation))
            gradient = space.scale(
                -2.0 * scale, linearization.derivative.adjoint(weighted)
            )
            return QuadraticModel(
                model,
                scale * self.chi_squared_from_residual(residual),
                LinearFunctional.from_representer(space, gradient),
            )

        if forward.has_derivative:
            return Functional.from_callables(space, value, linearize=linearize)
        return Functional.from_callables(space, value)

    def consistency_set(self, model: Any, /, *, level: float = 0.95) -> Subset:
        """The data compatible with a model, as an ellipsoid in the data space.

        The error measure hardened into a set: its credible ellipsoid at
        *level*, centered on the data the model predicts. It discards
        information, and says so by being a separate, named step rather than
        something a constructor does quietly. Its members are the data whose
        misfit against the model passes :meth:`chi_squared_test` at the same
        level, and like the misfit it is built on a data space that
        discretizes a function space as on any other; a limit under
        refinement needs the error measure to be trace class.

        Args:
            model: the model whose predictions the set is centered on.
            level: the confidence level.

        Returns:
            An ellipsoid in the data space.

        Raises:
            ValueError: if the problem has no data uncertainty, or for a
                level outside ``(0, 1)``.
            TypeError: if the uncertainty is a set rather than a measure.
        """
        # unsafe=True as for the misfit, which the set is the sublevel set of.
        return self.error_measure.credible_set(level=level, unsafe=True).translate(
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
            from ..probability.base import product

            error = product([problem.error_measure for problem in problems])
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
        """The distribution of data induced by a distribution of models.

        Args:
            measure: a measure on the model space.

        Returns:
            Its push-forward through the forward operator, convolved with the
            data error when there is one.

        Raises:
            ValueError: if the measure is not on the model space.
        """
        if measure.domain != self.model_space:
            raise ValueError("The measure is not defined on the model space.")
        pushed = measure.push_forward(self._forward_operator)
        return pushed + self.error_measure if self.has_error else pushed

    def joint_measure(self, measure: ProbabilityMeasure, /) -> ProbabilityMeasure:
        """The joint distribution of model and data, on their direct sum.

        The model is drawn from *measure* and the data are its prediction plus
        an independent draw of the error, so one sample of this is a
        consistent ``(model, data)`` pair. The independent product of the
        two measures is pushed through ``(m, e) -> (m, A m + e)``: Gaussian
        when both are Gaussian, and otherwise a measure that samples -- a
        mixture prior, say -- without claiming a closed form.

        Args:
            measure: the measure on the model space.

        Returns:
            The joint measure on the direct sum of the model and data spaces.

        Raises:
            ValueError: if the measure is not on the model space.
        """
        model_space, data_space = self.model_space, self.data_space
        if measure.domain != model_space:
            raise ValueError("The measure is not defined on the model space.")
        if not self.has_error:
            return measure.push_forward(
                ColumnLinearOperator(
                    [LinearOperator.identity(model_space), self._forward_operator]
                )
            )

        from ..algebra.direct_sum import BlockLinearOperator
        from ..probability.base import product

        block = BlockLinearOperator(
            [
                [
                    LinearOperator.identity(model_space),
                    LinearOperator.zero(data_space, codomain=model_space),
                ],
                [self._forward_operator, LinearOperator.identity(data_space)],
            ]
        )
        joint = product([measure, self.error_measure])
        return joint.push_forward(block)

    def synthetic_model_and_data(
        self, prior: ProbabilityMeasure, /, *, rng: Generator | None = None
    ) -> tuple[Any, Any]:
        """A model drawn from the prior, and data consistent with it.

        Args:
            prior: the measure on the model space to draw from; any measure
                that can sample.
            rng: the generator to draw with.

        Returns:
            The ``(model, data)`` pair, one draw of :meth:`joint_measure`.
        """
        sample = self.joint_measure(prior).sample(rng=rng)
        return sample[0], sample[1]

    # ----------------------------------------------------------------- #
    #                        Transformed problems                       #
    # ----------------------------------------------------------------- #

    def parameterized(
        self, parameterization: LinearOperator, /
    ) -> LinearForwardProblem:
        """The same data, seen through a restricted model space.

        Args:
            parameterization: from the parameter space into the model space.

        Returns:
            The problem on the parameter space, with the same error.

        Raises:
            ValueError: if the parameterization does not map into the model
                space.
        """
        if parameterization.codomain != self.model_space:
            raise ValueError("The parameterization must map into the model space.")
        return LinearForwardProblem(
            self._forward_operator @ parameterization, error=self._error
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

        Returns:
            The problem on the reduced data space.

        Raises:
            ValueError: if the reduction does not map from the data space.
            TypeError: with no explicit *error*, if the current one is a
                set, which cannot be pushed forward the way a measure can.
        """
        if reduction.domain != self.data_space:
            raise ValueError("The reduction must map from the data space.")
        if error is None and self.has_error:
            error = self.error_measure.push_forward(reduction)
        return LinearForwardProblem(reduction @ self._forward_operator, error=error)
