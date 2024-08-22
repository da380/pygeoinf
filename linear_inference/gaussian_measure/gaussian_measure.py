if __name__ == "__main__":
    pass

from linear_inference.vector_space import LinearOperator

class GaussianMeasure:

    def __init__(self, domain, covariance, / , *, mean = None, sample = None):

        self._domain = domain
        self._covariance = covariance
        self._mean = mean
        self._sample = sample

    # Return the domain. 
    @property
    def domain(self):
        return self._domain
    
    # Return the mean. 
    @property
    def mean(self):
        if self._mean is None:
            return self._domain.zero
        else:
            return self._mean()

    # Return the covariance as an instance of LinearOperator.
    @property
    def covariance(self):
        return LinearOperator.self_adjoint_operator(self.domain, self._covariance)


    # Return samples from the distribution. 
    def sample(self, n):
        if self._sample is None:
            raise NotImplementedError("sample method has not been set")
        else:
            return self._sample(n)