import numpy as np
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator


from linear_inference.interfaces import AbstractVectorSpace, AbstractHilbertSpace
from linear_inference.linear_form import LinearForm


class DualVectorSpace(AbstractVectorSpace):

    def __init__(self, space):
        self._space = space

    @property
    def dimension(self):
        return self.dual.dimension

    @property
    def dual(self):
        return self._space
        
    def to_components(self,xp):
        n = self.dimension
        c = np.zeros(n)
        cp = np.zeros(n)
        for i in range(n):
            c[i] = 1
            cp[i] = xp(self.dual.from_components(c))
            c[i] = 0
        return cp

    def from_components(self,cp):
        return  LinearForm(self.dual, lambda x : np.dot(self.dual.to_components(x),cp))
    
    
class VectorSpace(AbstractVectorSpace):
    
    def __init__(self, dimension, to_components = lambda x : x, from_components = lambda x : x):
        self._dimension = dimension
        self._to_components = to_components
        self._from_components = from_components    

    # Return the dimension of the space. 
    @property
    def dimension(self):
        return self._dimension

    @property
    def dual(self):
        return DualVectorSpace(self)

    # Map a vector to its components. 
    def to_components(self,x):
        return self._to_components(x)

    # Map components to a vector. 
    def from_components(self,c):
        return self._from_components(c)    


class DualHilbertSpace(AbstractHilbertSpace, DualVectorSpace):

    def __init__(self, space):
        super(DualHilbertSpace,self).__init__(space)
        self._space = space

    @property
    def dual(self):
        return self._space

    def inner_product(self, xp1, xp2):
        return self.dual.inner_product(self.dual.from_dual(xp1),self.dual.from_dual(xp2))

    def from_dual(self, x):
        return lambda y : self.dual.inner_product(x,y)

        
class HilbertSpace(AbstractHilbertSpace, VectorSpace):
    
    def __init__(self, dimension, to_components = lambda x : x, from_components = lambda x: x, inner_product = (lambda x1, x2 : np.dot(x1,x2)), from_dual = None):
        super(HilbertSpace,self).__init__(dimension, to_components, from_components)
        self._inner_product = inner_product
        if from_dual == None:                        
            self._set_metric()
            self._from_dual = lambda xp :  self._from_dual_default(xp)
        else:
            self._from_dual = from_dual

    @property
    def dual(self):
        return DualHilbertSpace(self)

    # Return the inner product of two vectors. 
    def inner_product(self, x1, x2):
        return self._inner_product(x1, x2)

    # Construct the Cholesky factorisation of the metric.
    def _set_metric(self):        
        metric = np.zeros((self.dimension, self.dimension))
        c1 = np.zeros(self.dimension)
        c2 = np.zeros(self.dimension)
        for i in range(self.dimension):
            c1[i] = 1
            x1 = self.from_components(c1)
            metric[i,i] = self.inner_product(x1,x1)
            for j in range(i+1,self.dimension):
                c2[j] = 1
                x2 = self.from_components(c2)                
                metric[i,j] = self.inner_product(x1,x2)          
                metric[j,i] = metric[i,j]
                c2[j] = 0
            c1[i] = 0                      
        self._metric_factor = cho_factor(metric)        

    # Default implementation for the representation of a dual vector. 
    def _from_dual_default(self,xp):    
        cp = self.dual.to_components(xp)
        c = cho_solve(self._metric_factor,cp)        
        return self.from_components(c)

    # Return the representation of a dual vector in the space. 
    def from_dual(self, xp):        
        return self._from_dual(xp)



