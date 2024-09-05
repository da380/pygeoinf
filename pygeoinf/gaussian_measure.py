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
    (2) A functor that represents the action of the covariance operator. 
    (3) The expectation of the measure. 
    (4) A function that returns a random sample drawn from the measure. 
    """

    def __init__(self, domain, covariance, / , *, expectation = None,
                 sample = None, sample_using_dense_matrix = False):  
        """
        Args:
            domain (HilbertSpace): The Hilbert space on which the measure is defined. 
            covariance: A functor representing the covariance operator. 
            expectation: The expectation value of the measure. If none is provided, set equal to zero.  
            sample: A callable function that returns a random sample from the measure.         
        """
        self._domain = domain              
        self._covariance = covariance
        if expectation is None:
            self._expectation = self.domain.zero
        else:
            self._expectation = expectation
        if sample is None:
            self._sample_defined = False
        else:
            self._sample = sample
            self._sample_defined = True
        if sample is None and sample_using_dense_matrix:
            dist = multivariate_normal(expectation = self.expectation, cov= self.covariance.to_dense_matrix)
            self._sample = lambda : self.domain.from_components(dist.rvs())
            self._sample_defined = True


    @staticmethod
    def from_factored_covariance(factor, /, *,  expectation = None):                
        """ 
        Form a Gaussian measure using a covariance of factored form. 

        Here the user provided a linear operator, L, that from Euclidean
        space to the domain of the measure. The covariance operator then 
        takes the form C = LL*. 

        The value of this factorisation is that samples from the measure 
        can be formed by acting L on a vector of samples from a standard
        Gaussian distribution on \mathbb{R}.

        Args:
            factor (LinearOperator): The linear operator, L. 
            expectation: The expectation value of the measure. 

        Returns:
            GaussianMeasure: The measure with given expectation and covariance, LL*.    
        """     
        covariance  = factor @ factor.adjoint
        sample = lambda : factor(norm().rvs(size = factor.domain.dim))
        if expectation is not None:
            sample = lambda : expectation + sample()        
        return GaussianMeasure(factor.codomain, covariance, expectation = expectation, sample = sample)    

    @property
    def domain(self):
        """The Hilbert space the measure is defined on."""
        return self._domain
    
    @property
    def covariance(self):
        """The covariance operator as an instance of LinearOperator."""
        return LinearOperator.self_adjoint(self.domain, self._covariance)
    
    @property
    def expectation(self):
        """The expectation of the measure."""
        return self._expectation

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
        assert operator.domain == self.domain
        axpy = operator.codomain.axpy
        if operator is None:
            covariance = self.covariance
        else:
            covariance = operator @ self.covariance @ operator.adjoint        
        expectation = operator(self.expectation)
        if translation is not None:
            expectation = axpy(1, expectation, translation)
        if self.sample_defined:
            if translation is None:
                sample = lambda : operator(self.sample())
            else: 
                sample = lambda  : axpy(1, operator(self.sample()),
                                        translation)
        else:
            sample = None
        return GaussianMeasure(operator.codomain, covariance, expectation = expectation, sample = sample)            

    def expectation_of_linear_form(self, u):    
        """
        Returns the expected value of the linear form defined by a vector. 
        """
        return self.domain.inner_product(u, self.expectation)

    def covariance_of_linear_forms(self,u1,u2):
        """Returns the covariance of the linear forms defined by two vectors."""
        inner_product = self.domain.inner_product
        return (inner_product(self.covariance(u1),u2) 
               + self.expectation_of_linear_form(u1) 
               * self.expectation_of_linear_form(u2))
       
    def variance_of_linear_form(self,u):
        """ Returns the variance of the linera form defined by a vector."""
        return self.covariance_of_linear_forms(u,u)
        
    def __mul__(self, alpha):
        """Multiply the measure by a scalar."""
        covariance = LinearOperator.self_adjoint(self.domain,lambda x : alpha*2 * self.covariance(x))
        expectation = alpha * self.expectation
        if self.sample_defined:
            sample = lambda : alpha * self.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, expectation = expectation, sample = sample)

    def __rmul__(self, alpha):
        """Multiply the measure by a scalar."""
        return self * alpha
    
    def __add__(self, other):
        """Add two measures on the same domain."""
        assert self.domain == other.domain
        covariance = self.covariance + other.covariance
        expectation = self.expectation + other.expectation
        if self.sample_defined and other.sample_defined:
            sample  = lambda : self.sample() + other.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, expectation = expectation, sample = sample) 

    def __sub__(self, other):
        """Subtract two measures on the same domain."""
        assert self.domain == other.domain
        covariance = self.covariance + other.covariance
        expectation = self.expectation - other.expectation
        if self.sample_defined and other.sample_defined:
            sample  = lambda : self.sample() - other.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, expectation = expectation, sample = sample)     


        