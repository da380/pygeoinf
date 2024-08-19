import numpy as np
from scipy.linalg import cho_factor, cho_solve
from linear_inference.hilbert_space import VectorSpace, HilbertSpace, LinearOperator



class EuclideanSpace(HilbertSpace):

    def __init__(self,dimension, /, *, metric = None):
        identity = lambda x : x
        space = VectorSpace(dimension, identity, identity)
        if metric is None:            
            to_dual_space = identity
            from_dual_space = identity
        else:
            factor = cho_factor(metric)
            to_dual_space = lambda x : metric @ x
            from_dual_space =  lambda xp : cho_solve(factor,xp)
        super(EuclideanSpace,self).__init__(space, space, to_dual_space, from_dual_space)
        