if __name__ == "__main__":
    pass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
from linear_inference.vector_space import HilbertSpace, LinearForm




# Implementation of Euclidean space. By default, the standard metric is used, 
# but a user-defined one can be supplied as a symmetric numpy matrix. For high-dimensional 
# spaces with non-standard inner-products a more efficient implementation could be made
# (e.g., using sparse matrices and iterative methods instead of direct solvers).
class EuclideanL2(HilbertSpace):
    
    def __init__(self, dim, /, *, metric = None):
        
        if metric is None:
            from_dual = lambda xp : self.dual.to_components(xp)
            to_dual = lambda x : LinearForm(self, components = x)
            super(EuclideanL2,self).__init__(dim, lambda x : x, lambda x : x, (lambda x1, x2, : np.dot(x1,x2)), 
                                                from_dual = from_dual,  to_dual = to_dual)
        else:            
            factor = cho_factor(metric)            
            inner_product = lambda x1, x2 : np.dot(metric @ x1, x2)
            from_dual = lambda xp : cho_solve(factor, self.dual.to_components(xp))
            super(EuclideanL2,self).__init__(dim, lambda x : x , lambda x : x, inner_product, from_dual = from_dual)
        
        self._metric = metric



    @property
    def metric(self):
        if self._metric is None:
            return np.identity(self.dim)
        else:
            return self._metric

    @staticmethod
    def with_random_metric(dim):
        A = norm.rvs(size = (dim, dim))
        metric = A.T @ A + 0.1 * np.identity(dim)        
        return EuclideanL2(dim, metric = metric)


    