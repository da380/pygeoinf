"""
Defines the mathematical structure of a forward problem.

This module provides classes that encapsulate the core components of an
inverse problem. A forward problem describes the physical or mathematical
process that maps a set of unknown model parameters `u` to a set of observable
data `d`.

The module handles both the deterministic relationship `d = A(u)` and the more
realistic statistical model `d = A(u) + e`, where `e` represents random noise.

Key Classes
-----------
- `ForwardProblem`: A general class representing the link between a model
  space and a data space via a forward operator, with an optional data error.
- `LinearForwardProblem`: A specialization for linear problems where the
  forward operator is a `LinearOperator`.
"""

from __future__ import annotations
from typing import Optional, List, Tuple, TYPE_CHECKING

from scipy.stats import chi2

from .gaussian_measure import GaussianMeasure
from .direct_sum import ColumnLinearOperator, BlockLinearOperator
from .linear_operators import LinearOperator


# This block only runs for type checkers, not at runtime, to prevent
# circular import errors while still allowing type hints.
if TYPE_CHECKING:
    from .hilbert_space import HilbertSpace, Vector
    from .nonlinear_operators import NonLinearOperator


class ForwardProblem:
    """
    Represents a general forward problem.

    An instance is defined by a forward operator that maps from a model space
    to a data space, and an optional Gaussian measure representing the
    statistical distribution of errors in the data.
    """

    def __init__(
        self,
        forward_operator: NonLinearOperator,
        /,
        *,
        data_error_measure: Optional[GaussianMeasure] = None,
    ) -> None:
        """Initializes the ForwardProblem.

        Args:
            forward_operator: The operator that maps from the model space to the
                data space.
            data_error_measure: A Gaussian measure representing the distribution
                from which data errors are assumed to be drawn. If None, the
                data is considered to be error-free.
        """
        self._forward_operator: NonLinearOperator = forward_operator
        self._data_error_measure: Optional[GaussianMeasure] = data_error_measure
        if self.data_error_measure_set:
            if self.data_space != data_error_measure.domain:
                raise ValueError(
                    "Data error measure must be defined on the data space (codomain of the forward operator)."
                )

    @property
    def forward_operator(self) -> LinearOperator:
        """The forward operator, mapping from model to data space."""
        return self._forward_operator

    @property
    def data_error_measure_set(self) -> bool:
        """True if a data error measure has been set."""
        return self._data_error_measure is not None

    @property
    def data_error_measure(self) -> GaussianMeasure:
        """The measure from which data errors are drawn."""
        if not self.data_error_measure_set:
            raise AttributeError("Data error measure has not been set.")
        return self._data_error_measure

    @property
    def model_space(self) -> "HilbertSpace":
        """The model space (domain of the forward operator)."""
        return self.forward_operator.domain

    @property
    def data_space(self) -> "HilbertSpace":
        """The data space (codomain of the forward operator)."""
        return self.forward_operator.codomain


class LinearForwardProblem(ForwardProblem):
    """
    Represents a linear forward problem of the form `d = A(u) + e`.

    Here, `d` is the data, `A` is the linear forward operator, `u` is the model,
    and `e` is a random error drawn from a Gaussian distribution.
    """

    def __init__(
        self,
        forward_operator: LinearOperator,
        /,
        *,
        data_error_measure: Optional[GaussianMeasure] = None,
    ) -> None:
        """
        Args:
            forward_operator: The operator that maps from the model space to the
                data space.
            data_error_measure: A Gaussian measure representing the distribution
                from which data errors are assumed to be drawn. If None, the
                data is considered to be error-free.
        """

        if not isinstance(forward_operator, LinearOperator):
            raise ValueError("Forward operator must be a linear operator.")

        super().__init__(forward_operator, data_error_measure=data_error_measure)

    @staticmethod
    def from_direct_sum(
        forward_problems: List[LinearForwardProblem],
    ) -> LinearForwardProblem:
        """
        Forms a joint forward problem from a list of separate problems.

        This is a powerful tool for joint inversions, where a single underlying
        model is observed through multiple, independent measurement systems
        (e.g., different types of geophysical surveys).

        Args:
            forward_problems: A list of `LinearForwardProblem` instances that
                share a common model space.

        Returns:
            A single `LinearForwardProblem` where the data space is the direct
            sum of the individual data spaces.
        """
        if not forward_problems:
            raise ValueError("Cannot form a direct sum from an empty list.")

        model_space = forward_problems[0].model_space
        if not all(fp.model_space == model_space for fp in forward_problems):
            raise ValueError("All forward problems must share a common model space.")

        # Create a joint operator that maps one model to a list of data vectors
        joint_forward_operator = ColumnLinearOperator(
            [fp.forward_operator for fp in forward_problems]
        )

        # Combine the data error measures if they all exist
        if all(fp.data_error_measure_set for fp in forward_problems):
            data_error_measure = GaussianMeasure.from_direct_sum(
                [fp.data_error_measure for fp in forward_problems]
            )
        else:
            data_error_measure = None

        return LinearForwardProblem(
            joint_forward_operator, data_error_measure=data_error_measure
        )

    def data_measure_from_model(self, model: Vector) -> GaussianMeasure:
        """
        Returns the Gaussian measure for the data, given a specific model.

        The resulting measure has a mean of `A(model)` and the covariance of
        the data error.

        Args:
            model: A vector from the model space.

        Returns:
            The Gaussian measure representing the distribution of possible data.
        """
        if not self.data_error_measure_set:
            raise AttributeError("Data error measure has not been set.")

        return self.data_error_measure.affine_mapping(
            translation=self.forward_operator(model)
        )

    def data_measure_from_model_measure(
        self, model_measure: GaussianMeasure
    ) -> GaussianMeasure:
        """
        Given a measure for the model space, returns the induced measure on the
        data space.
        """

        if model_measure.domain != self.model_space:
            raise ValueError("Input measure not defined on the model space")

        data_measure = model_measure.affine_mapping(operator=self.forward_operator)

        return (
            data_measure + self.data_error_measure
            if self.data_error_measure_set
            else data_measure
        )

    def joint_measure(self, model_measure: GaussianMeasure) -> GaussianMeasure:
        """
        Given a measure for the model space, returns the joint measure for the model
        and data.
        """

        if self.data_error_measure_set:
            op = BlockLinearOperator(
                [
                    [
                        self.model_space.identity_operator(),
                        self.data_space.zero_operator(self.model_space),
                    ],
                    [self.forward_operator, self.data_space.identity_operator()],
                ]
            )

            mu = GaussianMeasure.from_direct_sum(
                [model_measure, self.data_error_measure]
            )

            return mu.affine_mapping(operator=op)
        else:
            op = ColumnLinearOperator(
                [self.model_space.identity_operator(), self.forward_operator]
            )

            return model_measure.affine_mapping(operator=op)

    def synthetic_data(self, model: Vector) -> Vector:
        """
        Generates a synthetic data vector for a given model.

        The data is computed as `d = A(model) + e`, where `e` is a random
        sample from the data error measure.

        Args:
            model: A vector from the model space.

        Returns:
            A synthetic data vector.
        """
        return self.data_measure_from_model(model).sample()

    def synthetic_model_and_data(self, prior: GaussianMeasure) -> Tuple[Vector, Vector]:
        """
        Generates a random model and corresponding synthetic data.

        Args:
            prior: A Gaussian measure on the model space, from which the
                random model `u` will be drawn.

        Returns:
            A tuple `(u, d)`, where `u` is the random model and `d` is the
            corresponding synthetic data.
        """
        # Draw a single sample from the joint distribution and unpack it
        u, d = self.joint_measure(prior).sample()
        return u, d

    def critical_chi_squared(self, significance_level: float) -> float:
        """
        Returns the critical value of the chi-squared statistic.

        This value serves as the threshold for the chi-squared test at a given
        significance level.

        Args:
            significance_level: The desired significance level (e.g., 0.95).

        Returns:
            The critical chi-squared value.
        """
        return chi2.ppf(significance_level, self.data_space.dim)

    def chi_squared_from_residual(self, residual: Vector) -> float:
        """
        Calculates the chi-squared statistic from a residual vector.

        Args:
            residual: The residual vector.

        Returns:
            The chi-squared statistic.
        """
        if self.data_error_measure_set:
            residual = self.data_space.subtract(
                residual, self.data_error_measure.expectation
            )
            inverse_data_covariance = self.data_error_measure.inverse_covariance
            return self.data_space.inner_product(
                inverse_data_covariance(residual), residual
            )
        else:
            return self.data_space.squared_norm(residual)

    def chi_squared(self, model: Vector, data: Vector) -> float:
        """
        Calculates the chi-squared statistic for a given model and data.

        This measures the misfit between the predicted and observed data.

        - If a data error measure with an inverse covariance `C_e^-1` is defined,
          this is the weighted misfit: `(d - A(u))^T * C_e^-1 * (d - A(u))`.
        - Otherwise, it is the squared L2 norm of the data residual: `||d - A(u)||^2`.

        Args:
            model: A vector from the model space.
            data: An observed data vector from the data space.

        Returns:
            The chi-squared statistic.
        """

        residual = self.data_space.subtract(data, self.forward_operator(model))
        return self.chi_squared_from_residual(residual)

    def chi_squared_test(
        self, significance_level: float, model: Vector, data: Vector
    ) -> bool:
        """
        Performs a chi-squared test for goodness of fit.

        Args:
            significance_level: The significance level for the test (e.g., 0.95).
            model: A vector from the model space.
            data: An observed data vector from the data space.

        Returns:
            True if the model is statistically compatible with the data at the
            specified significance level, False otherwise.
        """
        return self.chi_squared(model, data) < self.critical_chi_squared(
            significance_level
        )

    def parameterized_problem(
        self,
        parameterization: LinearOperator,
        /,
        *,
        dense: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> "LinearForwardProblem":
        """
        Creates a new forward problem based on a model parameterization.
        """
        if parameterization.codomain != self.model_space:
            raise ValueError(
                "The codomain of the parameterization operator must match "
                "the model space of the forward problem."
            )

        new_op = self.forward_operator @ parameterization

        if dense:
            new_op = new_op.with_dense_matrix(parallel=parallel, n_jobs=n_jobs)

        new_error_measure = self._data_error_measure
        if new_error_measure is not None and dense:
            new_error_measure = new_error_measure.with_dense_covariance(
                parallel=parallel, n_jobs=n_jobs
            )

        return LinearForwardProblem(new_op, data_error_measure=new_error_measure)

    def data_reduced_problem(
        self,
        reduction_operator: LinearOperator,
        /,
        *,
        reduced_data_error_measure: Optional[GaussianMeasure] = None,
        dense: bool = False,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> "LinearForwardProblem":
        """
        Creates a new forward problem by applying a reduction (or sketching)
        operator to the data space.

        Args:
            reduction_operator: A LinearOperator mapping from the current
                data space to the new reduced data space.
            reduced_data_error_measure: An optional data error measure on the
                reduced data space. If not provided, the original data error
                measure is pushed forward automatically.
            dense: If True, computes and stores operators as dense matrices.
            parallel: If True, computes the dense matrices in parallel.
            n_jobs: Number of CPU cores to use. -1 means all available.

        Returns:
            A new LinearForwardProblem operating in the reduced data space.
        """
        if reduction_operator.domain != self.data_space:
            raise ValueError(
                "The domain of the reduction operator must match "
                "the data space of the forward problem."
            )

        new_op = reduction_operator @ self.forward_operator
        if dense:
            new_op = new_op.with_dense_matrix(parallel=parallel, n_jobs=n_jobs)

        if reduced_data_error_measure is not None:
            if reduced_data_error_measure.domain != reduction_operator.codomain:
                raise ValueError(
                    "The domain of the reduced data error measure must match "
                    "the codomain of the reduction operator."
                )
            new_error_measure = reduced_data_error_measure
        else:
            new_error_measure = self._data_error_measure
            if new_error_measure is not None:
                new_error_measure = new_error_measure.affine_mapping(
                    operator=reduction_operator
                )

        if new_error_measure is not None and dense:
            new_error_measure = new_error_measure.with_dense_covariance(
                parallel=parallel, n_jobs=n_jobs
            )

        return type(self)(new_op, data_error_measure=new_error_measure)
