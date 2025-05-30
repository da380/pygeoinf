"""
Module for Gaussian measures on Hilbert spaces. 
"""

import numpy as np
from pygeoinf.hilbert_space import (
    LinearOperator,
    DiagonalLinearOperator,
    EuclideanSpace,
)
from pygeoinf.direct_sum import (
    HilbertSpaceDirectSum,
    BlockDiagonalLinearOperator,
)


class GaussianMeasure:
    """
    Class for Gaussian measures on a Hilbert space.
    """

    def __init__(
        self,
        domain,
        covariance,
        /,
        *,
        expectation=None,
        sample=None,
        inverse_covariance=None,
    ):
        """
        Args:
            domain (HilbertSpace): The Hilbert space on which the measure is defined.
            covariance (LinearOperator): A self-adjoint and non-negative linear operator on the domain
            expectation (vector | zero): The expectation value of the measure.
            sample (callable | None): A functor that returns a random sample from the measure.
            inverse_covariance (LinearOperator | None): The inverse of the covariance.
        """
        self._domain = domain
        self._covariance = covariance

        if expectation is None:
            self._expectation = self.domain.zero
        else:
            self._expectation = expectation

        self._sample = sample

        self._inverse_covariance = inverse_covariance

    @staticmethod
    def from_factored_covariance(factor, /, *, inverse_factor=None, expectation=None):
        """
        For a Gaussian measure whos covariance, C, is approximated in the form C = LL*,
        with L a mapping into the domain from Euclidean space.

        Args:
            factor (LinearOperator): Linear operator from Euclidean space into the
                domain of the measure.
            expectation (vector | zero): expected value of the measure.
            inverse factor (LinearOperator): An analgous factorisation of the inverse covariance.

        Returns:
            GassianMeasure: The measure with the required covariance and expectation.
        """

        def sample():
            value = factor(np.random.randn(factor.domain.dim))
            if expectation is None:
                return value
            else:
                return factor.codomain.add(value, expectation)

        covariance = factor @ factor.adjoint

        _inverse_covariance = (
            inverse_factor.adjoint @ inverse_factor
            if inverse_factor is not None
            else None
        )

        return GaussianMeasure(
            factor.codomain,
            covariance,
            expectation=expectation,
            sample=sample,
            inverse_covariance=_inverse_covariance,
        )

    @staticmethod
    def from_standard_deviation(domain, standard_deviation):
        """
        Forms a Gaussian measure on a Hilbert space with zero
        expectation and whose covariance is proportional to the
        identity operator.

        Args:
            domain (HilbertSpace): The Hilbert space on which the measure
                is defined.
            standard_devitation (float): The standard deviation by which the
                identity is scaled to form the covariance.

        Notes:
            This measure is only well-defined for finite-dimensional spaces and not
            for finite-dimensional approximations to infinite-dimensional spaces.
        """
        factor = standard_deviation * domain.identity_operator()
        inverse_factor = (1 / standard_deviation) * domain.identity_operator()
        return GaussianMeasure.from_factored_covariance(
            factor, inverse_factor=inverse_factor
        )

    @staticmethod
    def from_standard_deviations(domain, standard_deviations):
        """
        Forms a Gaussian measure on a Hilbert space with zero
        expectation and whose covariance is diagonal within the
        Galerkin representation.

        Args: domain (HilbertSpace): The Hilbert space on which the measure
                is defined.
        standard_devitations (numpy vector): The diagonal values of the covariance
            within its Galerkin representation.

        Raises:
            ValueError: If the dimension of the standard deviation vector does not
                match that of the Hilbert space.

        """
        if standard_deviations.size != domain.dim:
            raise ValueError(
                "Standard deviation vector does not have the correct length"
            )
        euclidean = EuclideanSpace(domain.dim)
        factor = DiagonalLinearOperator(
            euclidean, domain, standard_deviations, galerkin=True
        )
        return GaussianMeasure.from_factored_covariance(
            factor, inverse_factor=factor.inverse
        )

    @staticmethod
    def from_samples(domain, samples):
        """
        Forms a Gaussian measure from a set of samples based on
        the sample expectation and sample variance.

        Args:
            domain (HilbertSpace): The space the measure is defined on.
            samples ([vectors]): A list of samples.

        Notes:
            :
        """
        assert all([domain.is_element(x) for x in samples])

        n = len(samples)
        expectation = domain.zero
        for x in samples:
            expectation = domain.axpy(1 / n, x, expectation)

        if n == 1:
            covariance = LinearOperator.self_adjoint(domain, lambda x: domain.zero)

            def sample():
                return expectation

        else:
            offsets = [domain.subtract(x, expectation) for x in samples]
            covariance = LinearOperator.self_adjoint_from_tensor_product(
                domain, offsets
            ) / (n - 1)

            def sample():
                x = expectation
                randoms = np.random.randn(len(offsets))
                for y, r in zip(offsets, randoms):
                    x = domain.axpy(r / np.sqrt(n - 1), y, x)
                return x

        return GaussianMeasure(
            domain, covariance, expectation=expectation, sample=sample
        )

    @staticmethod
    def from_direct_sum(measures):
        """
        Forms the direct sum of a list of Gaussian measures.
        """

        domain = HilbertSpaceDirectSum([measure.domain for measure in measures])

        expectation = [measure.expectation for measure in measures]

        covariance = BlockDiagonalLinearOperator(
            [measure.covariance for measure in measures]
        )

        inverse_covariance = (
            BlockDiagonalLinearOperator(
                [measure.inverse_covariance for measure in measures]
            )
            if all(measure.inverse_covariance_set for measure in measures)
            else None
        )

        def sample_impl():
            return [measure.sample() for measure in measures]

        sample = (
            sample_impl if all(measure.sample_set for measure in measures) else None
        )

        return GaussianMeasure(
            domain,
            covariance,
            expectation=expectation,
            sample=sample,
            inverse_covariance=inverse_covariance,
        )

    @property
    def domain(self):
        """The Hilbert space the measure is defined on."""
        return self._domain

    @property
    def covariance(self):
        """The covariance operator as an instance of LinearOperator."""
        return self._covariance

    @property
    def inverse_covariance_set(self):
        """Returns true if inverse covariance is available"""
        return self._inverse_covariance is not None

    @property
    def inverse_covariance(self):
        """The covariance operator as an instance of LinearOperator."""
        if self._inverse_covariance is None:
            raise NotImplementedError("Inverse covariance is not set")
        else:
            return self._inverse_covariance

    @property
    def expectation(self):
        """The expectation of the measure."""
        return self._expectation

    @property
    def sample_set(self):
        """
        Returns true if sample method is available.
        """
        return self._sample is not None

    def sample(self):
        """
        Returns a random sample drawn from the measure.
        """
        if self._sample is None:
            raise NotImplementedError("Sample method is not set.")
        else:
            return self._sample()

    def samples(self, n):
        """
        Returns a list of n samples form the measure.
        """
        assert n >= 1
        return [self.sample() for _ in range(n)]

    def sample_expectation(self, n):
        """
        Returns the sample expectation using n > 1 samples.
        """
        assert n >= 1
        expectation = self.domain.zero
        for _ in range(n):
            sample = self.sample()
            expectation = self.domain.axpy(1 / n, sample, expectation)
        return expectation

    def affine_mapping(self, /, *, operator=None, translation=None):
        """
        Returns the push forward of the measure under an affine mapping.

        Args:
            operator (LinearOperator): The operator part of the mapping.
            translation (vector): The translational part of the mapping.

        Returns:
            Gaussian Measure: The transformed measure defined on the
                codomain of the operator.

        Raises:
            ValueError: If the domain of the operator domain is not
                the domain of the measure.

        Notes:
            If operator is not set, it defaults to the identity.
            It translation is not set, it defaults to zero.
        """

        if operator is None:
            _operator = self.domain.identity_operator()
        else:
            _operator = operator

        if translation is None:
            _translation = _operator.codomain.zero
        else:
            _translation = translation

        covariance = _operator @ self.covariance @ _operator.adjoint
        expectation = _operator.codomain.add(_operator(self.expectation), _translation)

        def sample():
            return _operator.codomain.add(_operator(self.sample()), _translation)

        return GaussianMeasure(
            _operator.codomain, covariance, expectation=expectation, sample=sample
        )

    def low_rank_approximation(self, rank, /, *, power=0):
        """
        Returns an approximation to the measure with the covariance operator
        replaced by a low-rank Cholesky decomposition along with the associated
        sampling method.
        """
        F = self.covariance.random_cholesky(rank, power=power)
        return GaussianMeasure.from_factored_covariance(F, expectation=self.expectation)

    def __neg__(self):
        """Negative of the measure."""
        return GaussianMeasure(
            self.domain,
            -self.covariance,
            expectation=self.domain.negative(self.expectation),
            sample=lambda: self.domain.negative(self.sample()),
        )

    def __mul__(self, alpha):
        """Multiply the measure by a scalar."""
        return GaussianMeasure(
            self.domain,
            alpha * alpha * self.covariance,
            expectation=self.domain.multiply(alpha, self.expectation),
            sample=lambda: self.domain.multiply(alpha, self.sample()),
        )

    def __rmul__(self, alpha):
        """Multiply the measure by a scalar."""
        return self * alpha

    def __add__(self, other):
        """Add two measures on the same domain."""
        return GaussianMeasure(
            self.domain,
            self.covariance + other.covariance,
            expectation=self.domain.add(self.expectation, other.expectation),
            sample=lambda: self.domain.add(self.sample(), other.sample()),
        )

    def __sub__(self, other):
        """Subtract two measures on the same domain."""
        return GaussianMeasure(
            self.domain,
            self.covariance + other.covariance,
            expectation=self.domain.subtract(self.expectation, other.expectation),
            sample=lambda: self.domain.subtract(self.sample(), other.sample()),
        )


class FactoredGaussianMeasure(GaussianMeasure):
    """
    Class for Gaussian measures whose covariance is expressed in a Cholesky form C = LL*.
    """

    def __init__(
        self,
        domain,
        covariance_factor,
        /,
        *,
        expectation=None,
        sample=None,
        inverse_covariance_factor=None,
    ):
        """
        Args:
            domain (HilbertSpace): The Hilbert space on which the measure is defined.
            covariance_factor (LinearOperator): The operator L in the factorisation C = LL*.
                The codomain of L is the domain of the measure, while L's domain is Euclidean
                space of an appropriate dimension.
            expectation (vector): The expectation value of the measure. Default is zero.
            inverse_covariance_factor (LinearOperator): The inverse of the covariance factor. Used to
                implement the inverse covariance. Default is none.
        """
        self._domain = domain
        self._covariance_factor = covariance_factor
        self._inverse_covariance_factor = inverse_covariance_factor

        domain = covariance_factor.codomain
        covariance = covariance_factor @ c.adjoint
        inverse_covariance = (
            inverse_covariance_factor.adjoint @ inverse_covariance_factor
            if inverse_covariance_factor is not None
            else None
        )

        def _sample():
            value = covariance_factor(np.random.randn(covariance_factor.domain.dim))
            if expectation is None:
                return value
            else:
                return domain.add(value, expectation)

        super().__init__(
            domain,
            covariance,
            expectation=expectation,
            sample=_sample,
            inverse_covariance=inverse_covariance,
        )


def sample_variance(measure, n):
    """
    Returns a sample-based estimate for the the pointwise variance
    for Gaussian measures whose elements can be multiplied. The measure
    must have a sample method defined.
    """

    if not measure.sample_set:
        raise ValueError("Measure does not have a sample method")

    if n < 1:
        raise ValueError("Number of samples must be greater than 1")

    samples = measure.samples(n)
    expectation = measure.expectation
    variance = measure.domain.zero
    for sample in samples:
        diff = sample - expectation
        prod = diff * diff
        variance += (1 / n) * prod

    return variance
