import numpy as np
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator
from linear_inference.linear_form import LinearForm

# Class for vector spaces. 
class VectorSpace:

    def __init__(self, dimension, to_components, from_components, /, * , dual_base = None):
        self._dimension = dimension
        self._to_components = to_components
        self._from_components = from_components    
        self._dual_base = dual_base

    # Return the dimension of the space. 
    @property
    def dimension(self):
        return self._dimension

    # Return the dual space. If the dual of a space, the original is returned. 
    @property
    def dual(self):
        if self._dual_base is None:            
            return VectorSpace(self.dimension, self._dual_to_components, 
                               self._dual_from_components, dual_base = self)
        else:
            return self._dual_base

    # Return the zero vector.
    @property
    def zero(self):
        return self.from_components(np.zeros(self.dimension))

    # Map a vector to its components. 
    def to_components(self,x):
       if isinstance(x, LinearForm):
        if x.components_stored:
            return x.components
       else:                
        return self._to_components(x)

    # Map components to a vector. 
    def from_components(self,c):
        return self._from_components(c)    

    # Maps a dual vector to its components. 
    def _dual_to_components(self,xp):
        n = self.dimension
        c = np.zeros(n)
        cp = np.zeros(n)
        for i in range(n):
            c[i] = 1
            cp[i] = xp(self.from_components(c))
            c[i] = 0
        return cp

     # Maps dual components to the dual vector. 
    def _dual_from_components(self, cp):
        return LinearForm(self, components = cp)

    # Return a vector whose components samples from a given distribution. 
    def random(self, dist = norm()):
        return self.from_components(norm.rvs(size = self.dimension))


# Class for Hilbert spaces.         
class HilbertSpace(VectorSpace):
    
    def __init__(self, dimension,  to_components, from_components, inner_product, /, *,  from_dual = None, dual_base = None):
        super(HilbertSpace,self).__init__(dimension, to_components, from_components)
        self._inner_product = inner_product
        if from_dual == None:                        
            self._form_and_factor_metric()
            self._from_dual = lambda xp :  self._from_dual_default(xp)
        else:
            self._from_dual = from_dual
        self._dual_base = dual_base

    # Return the dual. If space is the dual of another, the original is returned. 
    @property
    def dual(self):
        if self._dual_base is None:            
            return HilbertSpace(self.dimension,
                                self._dual_to_components,
                                self._dual_from_components, 
                                self._dual_inner_product,
                                from_dual = self.to_dual,
                                dual_base = self)
        else:
            return self._dual_base        

    # Return the inner product of two vectors. 
    def inner_product(self, x1, x2):
        return self._inner_product(x1, x2)

    # Return the norm of a vector. 
    def norm(self, x):
        return np.sqrt(self.inner_product(x,x))

    # Construct the Cholesky factorisation of the metric.
    def _form_and_factor_metric(self):
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

    # Map a vector to the corresponding dual vector. 
    def to_dual(self, x):
        return LinearForm(self, mapping = lambda y : self.inner_product(x,y))    

    # Inner product on the dual space. 
    def _dual_inner_product(self, xp1, xp2):
        return self.inner_product(self.from_dual(xp1),self.from_dual(xp2))



