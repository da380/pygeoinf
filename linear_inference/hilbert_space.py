import numpy as np
from scipy.linalg import cho_factor, cho_solve
from linear_inference.vector_space import VectorSpace, LinearForm, LinearOperator


class HilbertSpace(VectorSpace):

    def __init__(self, dimension, to_components, from_components, inner_product, from_dual = None):
        self._dimension = dimension
        self._to_components = to_components
        self._from_components = from_components
        self._inner_product = inner_product
        # If the mapping from the dual space is not set, use the default implementation. 
        if from_dual == None:            
            self._set_metric()
            self._from_dual = lambda xp :  self._from_dual_default(xp)
        else:
            self._from_dual = from_dual

    # Return the dual space. 
    @property
    def dual(self):
        sub_dual = super(HilbertSpace,self).dual
        inner_product = lambda xp1, xp2 : self.inner_product(self.from_dual(xp1), 
                                                             self.from_dual(xp2))
        from_dual = lambda x1 : (lambda x2 : self.inner_product(x1,x2))
        return HilbertSpace(self.dimension, sub_dual.to_components, 
                            sub_dual.from_components, inner_product, 
                            from_dual)

    # Return the inner product of two vectors. 
    def inner_product(self, x1, x2):
        return self._inner_product(x1, x2)

    # Return the norm of a vector. 
    def norm(self,x):
        return np.sqrt(self.inner_product(x,x))

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

    # Return the represenation of the vector as a linear form. 
    def to_dual(self,x):
        return LinearForm(self, lambda y : self.inner_product(x,y))
    


    

    
