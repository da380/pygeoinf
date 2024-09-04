"""
This module contains the definition of the Euclidean class. 
"""


import numpy as np
from scipy.stats import norm,multivariate_normal
from scipy.linalg import cho_factor, cho_solve
from pygeoinf.linear_form import LinearForm
from pygeoinf.hilbert_space import HilbertSpace
from pygeoinf.gaussian_measure import GaussianMeasure

if __name__ == "__main__":
    pass

class Euclidean(HilbertSpace):
    """
    Class that implements Euclidean space as a HilbertSpace object.
    
    The implementation is based on numpy arrays. By default, the standard metric is used,
    but a general one can be provided as a symmetric numpy matrix. 
    
    With a non-standard metric, the mappings from the dual is implemented using a Cholesky
    factorisation, and hence this implemention is not efficient for high-dimensional spaces. 
    """
    
    def __init__(self, dim, /, *, metric = None):
        """
        Args:
            dim (int): The dimension of the space. 
            metric: The metric tensor for the space as a symmetric numpy matrix. 

        Note:
            The symmetry or positive-definiteness of the matrix is not checked. 
        """
        
        if metric is None:
            from_dual = lambda xp : self.dual.to_components(xp)
            to_dual = lambda x : LinearForm(self, components = x)
            super(Euclidean,self).__init__(dim, lambda x : x, lambda x : x, 
                                           (lambda x1, x2, : np.dot(x1,x2)), 
                                            from_dual=from_dual, to_dual=to_dual)
        else:            
            factor = cho_factor(metric)
            inner_product = lambda x1, x2 : np.dot(metric @ x1, x2)
            from_dual = lambda xp : cho_solve(factor, self.dual.to_components(xp))
            to_dual = lambda x : metric @ x
            super(Euclidean,self).__init__(dim, lambda x : x, lambda x : x, inner_product, 
                                           from_dual = from_dual, to_dual=to_dual)
        
        self._metric = metric


    @staticmethod
    def with_random_metric(dim):
        """Form a space requested dimension with a random metric tensor."""
        A = norm.rvs(size = (dim, dim))
        metric = A.T @ A + 0.1 * np.identity(dim)        
        return Euclidean(dim, metric = metric)

    @property
    def metric(self):
        """The metric tensor for the space."""
        if self._metric is None:
            return np.identity(self.dim)
        else:
            return self._metric

    def gaussian_measure(self, covariance, /, *, mean = None):
        """
        Return a gaussian measure on the space with specified mean and covariance. 

        Args:
            covariance: The covariance operator as a symmetric numpy matrix. 
            mean: The mean vector. If input is None, value set to zero. 

        Notes:
            Sampling method uses scipy.stats.multivariate_normal which is not 
                efficient in high-dimensional spaces. 
q        """
        if mean is None:
            dist = multivariate_normal(cov = covariance)
        else:
            dist = multivariate_normal(mean = mean, cov = covariance)
        sample = lambda : dist.rvs()
        return GaussianMeasure(self, lambda x : covariance @ x, mean = mean, sample = sample)
        
