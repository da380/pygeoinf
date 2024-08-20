import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
from linear_inference.hilbert_space import VectorSpace, HilbertSpace, LinearOperator



class EuclideanSpace(HilbertSpace):

    def __init__(self,dimension, /, *, metric = None):        
        space = VectorSpace(dimension)
        if metric is None:            
            to_dual_space = lambda x : x
            from_dual_space = lambda x : x
        else:
            factor = cho_factor(metric)
            to_dual_space = lambda x : metric @ x
            from_dual_space =  lambda xp : cho_solve(factor,xp)
        super(EuclideanSpace,self).__init__(space, space, to_dual_space, from_dual_space)



    @staticmethod
    def with_random_metric(dimension):
        A = norm.rvs(size = (dimension,dimension))
        metric = A.T @ A + 0.1 * np.identity(dimension)        
        return EuclideanSpace(dimension, metric = metric)


        