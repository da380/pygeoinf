if __name__ == "__main__":
    pass

import numpy as np
from scipy.stats import norm,multivariate_normal
from scipy.linalg import cho_factor, cho_solve
from linear_inference.linear_form import LinearForm
from linear_inference.vector_space import HilbertSpace
from linear_inference.gaussian_measure import GaussianMeasure



# Implementation of Euclidean space using numpy arrays. By default, the standard metric is used, 
# but a user-defined one can be supplied as a symmetric numpy matrix. For high-dimensional 
# spaces with non-standard inner-products a more efficient implementation could be made
# (e.g., using sparse matrices and iterative methods instead of direct solvers).
class Euclidean(HilbertSpace):
    
    def __init__(self, dim, /, *, metric = None):
        
        if metric is None:
            from_dual = lambda xp : self.dual.to_components(xp)
            to_dual = lambda x : LinearForm(self, components = x)
            super(Euclidean,self).__init__(dim, lambda x : x, lambda x : x, (lambda x1, x2, : np.dot(x1,x2)), 
                                                from_dual = from_dual,  to_dual = to_dual)
        else:            
            factor = cho_factor(metric)            
            inner_product = lambda x1, x2 : np.dot(metric @ x1, x2)
            from_dual = lambda xp : cho_solve(factor, self.dual.to_components(xp))
            super(Euclidean,self).__init__(dim, lambda x : x, lambda x : x, inner_product, from_dual = from_dual)
        
        self._metric = metric


    @staticmethod
    def with_random_metric(dim):
        A = norm.rvs(size = (dim, dim))
        metric = A.T @ A + 0.1 * np.identity(dim)        
        return Euclidean(dim, metric = metric)

    @property
    def metric(self):
        if self._metric is None:
            return np.identity(self.dim)
        else:
            return self._metric

    # Return a gaussian measure on the space, with the covariance provided as a numpy dense matrix. 
    def gaussian_measure(self, covariance, /, *, mean = None):
        if mean is None:
            dist = multivariate_normal(cov = covariance)
        else:
            dist = multivariate_normal(mean = mean, cov = covariance)
        sample = lambda : dist.rvs()
        return GaussianMeasure(self, lambda x : covariance @ x, mean = mean, sample = sample)
        
