if __name__ == "__main__":
    pass

from linear_inference.vector_space import LinearOperator

class GaussianMeasure:

    # Define the measure in terms of its covariance operator (as an instance of LinearOperator).
    # The mean can be provided, and if not the distribution is zero mean. 
    # A function returning a sample from the distribution can be provided.
    def __init__(self, covariance, / , *, mean = None, sample = None):        
        assert covariance.domain == covariance.codomain
        self._covariance = covariance
        if mean is None:
            self._mean = self.domain.zero
        else:
            self._mean = mean
        if sample is None:
            self._sample_defined = False
        else:
            self._sample = sample
            self._sample_defined = True

    # Return the covariance operator. 
    @property
    def covariance(self):
        return self._covariance

    # Return the space the measure is defined on. 
    @property
    def domain(self):
        return self.covariance.domain

    # Return the mean. 
    @property
    def mean(self):
        return self._mean

    # Return true is sample is defined. 
    @property
    def sample_defined(self):
        return self._sample_defined

    # Return samples from the distribution. 
    def sample(self):
        if self.sample_defined:        
            return self._sample()
        else:
            raise NotImplementedError

    # Transform the measure under an affine transformation. If an operator 
    # is not provided, it is taken to be the identity mapping. If a translation
    # is not provided, it is taken to be zero.
    def affine_transformation(self, /, *,  operator = None, translation = None):
        if operator is None:
            operator_ = self.domain.identity_operator
        else:
            operator_ =  operator
        if translation is None:
            translation_ = self.domain.zero
        else:
            translation_ = translation
        covariance = operator_ @ self.covariance @ operator_.adjoint
        mean = self.mean + translation_
        if self.sample_defined:
            sample = lambda  : operator_(self.sample()) + translation_
        else : 
            sample = None
        return GaussianMeasure(covariance, mean = mean, sample = sample)
        
    # Transform the measure by multiplication by a scalar. 
    def __mul__(self, alpha):
        covariance = LinearOperator.self_adjoint_operator(self.domain, lambda x : alpha * alpha * self.covariance(x))
        mean = alpha * self.mean
        if self.sample_defined:
            sample = lambda : alpha * self.sample()
        else:
            sample = None
        return GaussianMeasure(covariance, mean = mean, sample = sample)

    def __rmul__(self, alpha):
        return self * alpha

    # Add another measure. 
    def __add__(self, other):
        assert self.domain == other.domain
        covariance = self.covariance + mean.covariance
        mean = self.mean + other.mean
        if self.sample_defined and other.sample_defined:
            sample  = lambda : self.sample() + other.sample()
        else:
            sample = None
        return GaussianMeasure(covariance, mean = mean, sample = sample) 

    # Subtract another measure. 
    def __sub__(self, other):
        assert self.domain == other.domain
        covariance = self.covariance + other.covariance
        mean = self.mean - other.mean
        if self.sample_defined and other.sample_defined:
            sample  = lambda : self.sample() - other.sample()
        else:
            sample = None
        return GaussianMeasure(covariance, mean = mean, sample = sample)     


        