from LinearInference.Hilbert import Space
import numpy as np
from scipy.linalg import norm
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve

# A numpy-based implementation of real Euclidean Space of given dimension with the inner product specified by a metric. 
def EuclideanSpace(dimension, metric = None):
        identity  = lambda x : x
        if metric is not None:
            assert metric.shape == (dimension,dimension)
            assert np.all(np.abs(metric-metric.T) < 1.e-8 * norm(metric))
            factor = cho_factor(metric)                
            riesz = lambda x : metric @ x
            inverseRiesz = lambda x : cho_solve(factor,x)
            innerProduct = lambda x1, x2 : np.dot(riesz(x1),x2)
            return Space(dimension, identity, identity, innerProduct, riesz, inverseRiesz)
        else:
            innerProduct = lambda x1, x2 : np.dot(x1,x2)
            return Space(dimension, identity, identity, innerProduct, identity, identity)
                                                

    
