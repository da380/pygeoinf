"""
This module defines the Gaussian measure class. 
"""

from pygeoinf.linear_operator import LinearOperator
from pygeoinf.linear_form import LinearForm
from pygeoinf.hilbert_space import HilbertSpace
from scipy.stats import norm, multivariate_normal

if __name__ == "__main__":
    pass

class GaussianMeasure:
    """
    Class for Gaussian measures on a real Hilbert space. A Gaussian measure is represented
    by the following information: 

    (1) The Hilbert space on which it is defined, as represented by a HilbertSpace object.
    (2) A callable object that represents the action of the covariance operator. 
    (3) The mean, or expected, value of the measure. 
    (4) A function that returns a random sample drawn from the measure. 
    """

    def __init__(self, domain, covariance, / , *, mean = None,
                 sample = None, sample_using_dense_matrices = False):  
        """
        Args:
            domain (HilbertSpace): The Hilbert space on which the measure is defined. 
            covariance: A callable object representing the covariance operator. 
            mean: The mean value of the measure. If none is provided, set equal to zero.  
            sample: A callable function that returns a random sample from the measure.         
        """
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

        if sample is None and sample_using_dense_matrices:
            dist = multivariate_normal(mean = self.mean, cov= self.covariance.to_dense_matrix)
            self._sample = lambda : self.domain.from_components(dist.rvs())
            self._sample_defined = True


    @staticmethod
    def from_factored_covariance(factor, /, *,  mean = None):                
        """ Form a Gaussian measure using a covariance of factored form. 

        Args:
            factor (LinearOperator): A linear operator, L, from \mathbb{R}^{n} to 
                the domain of the measure and such that the measure's covariance
                operator, C, can be written C = LL*.
            mean: The mean value of the measure. If none is provided, set equal to zero.  
        """
        assert factor.domain.dim == factor.codomain.dim    
        covariance  = factor @ factor.adjoint
        sample = lambda : factor(norm().rvs(size = factor.domain.dim))
        if mean is not None:
            sample = lambda : mean + sample()        
        return GaussianMeasure(factor.codomain, covariance, mean = mean, sample = sample)    

    @property
    def domain(self):
        """The Hilbert space the measure is defined on."""
        return self._domain
    
    @property
    def covariance(self):
        """The covariance operator as an instance of LinearOperator."""
        return LinearOperator.self_adjoint(self.domain, self._covariance)
    
    @property
    def mean(self):
        """The mean of the measure."""
        return self._mean

    @property
    def sample_defined(self):
        """True if the sample method has been implemented."""
        return self._sample_defined
    
    def sample(self):
        """Returns a random sample drawn from the measure."""
        if self.sample_defined:        
            return self._sample()
        else:
            raise NotImplementedError

    def affine_mapping(self, /, *,  operator = None, translation = None):
        """
        Returns the push forward of the measure under an affine mapping.

        Args:
            operator (LinearOperator): The linear operator part of the mapping. If
                not provided, then set equal to the identity. 
            translation: The translational part of the mapping, being a vector in the 
                codomain of the operator. 
        """
        assert operator.domain.dim == self.domain.dim        
        if operator is None:
            operator = LinearOperator.identity(self.domain)        
        if translation is None:
            translation = operator.codomain.zero        
        covariance = operator @ self.covariance @ operator.adjoint
        mean = operator(self.mean) + translation
        if self.sample_defined:
            sample = lambda  : operator_(self.sample()) + translation_
        else:
            sample = None
        return GaussianMeasure(operator.codomain, covariance, mean = mean, sample = sample)            

    def expectation_of_linear_form(self, u):    
        """
        Returns the expected value of the linear form defined by a vector. 
        """
        return self.domain.inner_product(u, self.mean)

    def covariance_of_linear_forms(self,u1,u2):
        """Returns the covariance of the linear forms defined by two vectors."""
        inner_product = self.domain.inner_product
        return inner_product(self.covariance(u1),u2) + self.expectation_of_linear_form(u1) * self.expectation_of_linear_form(u2)
       
    def variance_of_linear_form(self,u):
        """ Returns the variance of the linera form defined by a vector."""
        return self.covariance_of_linear_forms(u,u)
        
    def __mul__(self, alpha):
        """Multiply the measure by a scalar."""
        covariance = LinearOperator.self_adjoint(self.domain, lambda x : alpha * alpha * self.covariance(x))
        mean = alpha * self.mean
        if self.sample_defined:
            sample = lambda : alpha * self.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, mean = mean, sample = sample)

    def __rmul__(self, alpha):
        """Multiply the measure by a scalar."""
        return self * alpha
    
    def __add__(self, other):
        """Add two measures on the same domain."""
        assert self.domain == other.domain
        covariance = self.covariance + other.covariance
        mean = self.mean + other.mean
        if self.sample_defined and other.sample_defined:
            sample  = lambda : self.sample() + other.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, mean = mean, sample = sample) 

    
    def __sub__(self, other):
        """Subtract two measures on the same domain."""
        assert self.domain == other.domain
        covariance = self.covariance + other.covariance
        mean = self.mean - other.mean
        if self.sample_defined and other.sample_defined:
            sample  = lambda : self.sample() - other.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, mean = mean, sample = sample)     


        