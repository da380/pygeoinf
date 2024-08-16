from LinearInference.Hilbert import Space
from LinearInference.Operator import Linear
from LinearInference.Gaussian import Measure



import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from scipy.linalg import norm
from scipy.stats import multivariate_normal
from scipy.sparse import csr_array, triu
from scipy.sparse.linalg import LinearOperator


# A numpy-based implementation of real Euclidean Space of given dimension with the inner product specified by a metric. 
class EuclideanSpace(Space):

    def __init__(self,dimension, metric = None):
        identity  = lambda x : x
        if metric is not None:
            assert metric.shape == (dimension,dimension)
            assert np.all(np.abs(metric-metric.T) < 1.e-8 * norm(metric))
            factor = cho_factor(metric)                
            riesz = lambda x : metric @ x
            inverseRiesz = lambda x : cho_solve(factor,x)
            innerProduct = lambda x1, x2 : np.dot(riesz(x1),x2)
            super(EuclideanSpace, self).__init__(dimension, identity, identity, innerProduct, riesz, inverseRiesz)
        else:
            innerProduct = lambda x1, x2 : np.dot(x1,x2)
            super(EuclideanSpace, self).__init__(dimension, identity, identity, innerProduct, identity, identity)
                                                

    def _Identity(self,v):        
        assert v.size == self.Dimension
        return v

    def _Riesz(self,v):    
        return np.matmul(self.metric,v)

    def _InverseRiesz(self,vp):        
        return cho_solve(self.factor,vp)
    
    def _InnerProduct(self,v1,v2):    
        return np.dot(self.Riesz(v1),v2)


# Return a random metric tensor. 
def RandomMetricTensor(dim):
    A = np.random.normal(loc = 0, scale= 1, size= (dim, dim))
    return np.matmul(A,A.T) + 0.1*np.identity(dim)


# Gaussian measure on a Euclidean space.
def EuclideanGaussianMeasure(space, C, xbar):
    mean = lambda: xbar
    covariance = lambda x : C @ x
    sample = lambda: multivariate_normal(mean = xbar, cov = C).rvs()
    return Measure(space, mean, covariance, sample)


# Set up the first vector space. 
m = 5
X = EuclideanSpace(m, metric = RandomMetricTensor(m)) 

# Set up the second vector space. 
n = 2
Y = EuclideanSpace(n)

# Define the projection from X onto Y.
P = Linear(domain = X, coDomain = Y, mapping =  lambda x : x[:Y.Dimension],  dualMapping = lambda yp : np.append(yp,np.zeros(X.Dimension-Y.Dimension)))


print(P.AsDense)

















































            

