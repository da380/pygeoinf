if __name__ == "__main__":
    pass

from pygeoinf.linear_operator import LinearOperator
from pygeoinf.linear_form import LinearForm
from scipy.stats import norm, multivariate_normal

class GaussianMeasure:


    def __init__(self, domain, covariance, / , *, mean = None, sample = None):  
        self._domain = domain              
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


    @staticmethod
    # Form a gaussian measure using a factored covariance. The factor is an operator, L,  from 
    # \mathbb{R}^{n} to the domain of the measure and such that the covariance is LL^{*}. 
    def from_factored_covariance(factor, /, *,  mean = None):                
        assert factor.domain.dim == factor.codomain.dim    
        covariance  = factor @ factor.adjoint
        sample = lambda : factor(norm().rvs(size = factor.domain.dim))
        if mean is not None:
            sample = lambda : mean + sample()        
        return GaussianMeasure(factor.codomain, covariance, mean = mean, sample = sample)

    # Return the space the measure is defined on. 
    @property
    def domain(self):
        return self._domain

    # Return the covariance as a LinearOperator object. 
    @property
    def covariance(self):
        return LinearOperator.self_adjoint(self.domain, self._covariance)

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
        assert operator.domain.dim == self.domain.dim        
        if operator is None:
            operator_ = self.domain.identity_operator
        else:
            operator_ =  operator
        if translation is None:
            translation_ = operator.codomain.zero
        else:
            translation_ = translation
        covariance = operator_ @ self.covariance @ operator_.adjoint
        mean = operator(self.mean) + translation_
        if self.sample_defined:
            sample = lambda  : operator_(self.sample()) + translation_
        else : 
            sample = None
        return GaussianMeasure(operator.codomain, covariance, mean = mean, sample = sample)


    # Set the sampling method by forming the covariance matrix relative to the basis. 
    # If the method is already set, nothing is done. 
    def set_sample_method_using_dense_matrices(self):
        if self.sample_defined:
            pass
        else:            
            dist = multivariate_normal(mean = self.mean, cov= self.covariance.to_dense_matrix)
            self._sample = lambda : self.domain.from_components(dist.rvs())
            self._sample_defined = True

    # Return the expectation of the linear form v -> (u,v) with 
    # v a random vector distributed according to the measure.
    def expectation_of_linear_form(self, u):    
        return self.domain.inner_product(u, self.mean)

    # Return the covariance of the linear forms v -> (u1,v) and v -> (u2,v)
    # v a random vector distributed according to the measure.
    def covariance_of_linear_forms(self,u1,u2):
        inner_product = self.domain.inner_product
        return inner_product(self.covariance(u1),u2) + self.expectation_of_linear_form(u1) * self.expectation_of_linear_form(u2)

    # Return the variance of the linear form  v -> (u,v) with 
    # v a random vector distributed according to the measure.        
    def variance_of_linear_form(self,u):
        return self.covariance_of_linear_forms(u,u)
        
    # Transform the measure by multiplication by a scalar. 
    def __mul__(self, alpha):
        covariance = LinearOperator.self_adjoint(self.domain, lambda x : alpha * alpha * self.covariance(x))
        mean = alpha * self.mean
        if self.sample_defined:
            sample = lambda : alpha * self.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, mean = mean, sample = sample)

    def __rmul__(self, alpha):
        return self * alpha

    # Add another measure. 
    def __add__(self, other):
        assert self.domain == other.domain
        covariance = self.covariance + other.covariance
        mean = self.mean + other.mean
        if self.sample_defined and other.sample_defined:
            sample  = lambda : self.sample() + other.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, mean = mean, sample = sample) 

    # Subtract another measure. 
    def __sub__(self, other):
        assert self.domain == other.domain
        covariance = self.covariance + other.covariance
        mean = self.mean - other.mean
        if self.sample_defined and other.sample_defined:
            sample  = lambda : self.sample() - other.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, mean = mean, sample = sample)     


        