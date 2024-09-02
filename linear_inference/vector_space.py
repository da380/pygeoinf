if __name__ == "__main__":
    pass

import numpy as np
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from linear_inference.linear_form import LinearForm
from linear_inference.linear_operator import LinearOperator

# Class for vector spaces. 
class VectorSpace:

    def __init__(self, dim, to_components, from_components, /, * , dual_base = None):
        self._dim = dim
        self._to_components = to_components
        self._from_components = from_components    
        self._dual_base = dual_base

    # Return the dimension of the space. 
    @property
    def dim(self):
        return self._dim

    # Return the dual space. If the dual of a space, the original is returned. 
    @property
    def dual(self):
        if self._dual_base is None:            
            return VectorSpace(self.dim, self._dual_to_components, 
                               self._dual_from_components, dual_base = self)
        else:
            return self._dual_base

    # Return the zero vector.
    @property
    def zero(self):
        return self.from_components(np.zeros(self.dim))


    # Return the identity operator on the space. 
    @property
    def identity_operator(self):
        return LinearOperator(self, self, lambda x : x, dual_mapping = lambda xp : xp)

    # Map a vector to its components. 
    def to_components(self,x):
       if isinstance(x, LinearForm) and x.components_stored:
        return x.components        
       else:                
        return self._to_components(x)

    # Map components to a vector. 
    def from_components(self,c):
        return self._from_components(c)    

    # Maps a dual vector to its components. 
    def _dual_to_components(self,xp):
        n = self.dim
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
        return self.from_components(norm.rvs(size = self.dim))




# Class for Hilbert spaces.         
class HilbertSpace(VectorSpace):
    
    def __init__(self, dim,  to_components, from_components, inner_product, /, *,  from_dual = None, to_dual = None, dual_base = None):

        # Form the underlying vector space. 
        super(HilbertSpace,self).__init__(dim, to_components, from_components)

        # Set the inner
        self._inner_product = inner_product

        # Set the mapping from the dual space.         
        if from_dual is None:                        
            self._form_and_factor_metric()
            self._from_dual = lambda xp :  self._from_dual_default(xp)
        else:
            self._from_dual = from_dual

        # Set the mapping to the dual space. 
        if to_dual is None:
            self._to_dual = self._to_dual_default
        else:
            self._to_dual = to_dual

        # Store the base space (which may be none).
        self._dual_base = dual_base

    @staticmethod 
    def from_vector_space(space, inner_product, /, *,  from_dual = None, to_dual = None):
        return HilbertSpace(space.dim, space. to_components, space.from_components, from_dual = from_dual, to_dual = to_dual)

    # Return the dual. If space is the dual of another, the original is returned. 
    @property
    def dual(self):
        if self._dual_base is None:            
            return HilbertSpace(self.dim,
                                self._dual_to_components,
                                self._dual_from_components, 
                                self._dual_inner_product,
                                from_dual = self.to_dual,
                                to_dual = self.from_dual,
                                dual_base = self)
        else:
            return self._dual_base        


    # Return the identity operator on the space. 
    @property
    def identity_operator(self):
        return LinearOperator(self, self, lambda x : x, dual_mapping = lambda xp : xp, adjoint_mapping = lambda x : x)

    # Return the underlying vector space. 
    @property
    def to_vector_space(self):
        return VectorSpace(self.dim, self.to_components, self.from_components, dual_base = self._dual_base)

    # Return the inner product of two vectors. 
    def inner_product(self, x1, x2):
        return self._inner_product(x1, x2)

    # Return the norm of a vector. 
    def norm(self, x):
        return np.sqrt(self.inner_product(x,x))

    # Construct the Cholesky factorisation of the metric.
    def _form_and_factor_metric(self):
        metric = np.zeros((self.dim, self.dim))
        c1 = np.zeros(self.dim)
        c2 = np.zeros(self.dim)
        for i in range(self.dim):
            c1[i] = 1
            x1 = self.from_components(c1)
            metric[i,i] = self.inner_product(x1,x1)
            for j in range(i+1,self.dim):
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
        return self._to_dual(x)

    # Default implementation of the mapping to the dual space. 
    def _to_dual_default(self,x):
        return LinearForm(self, mapping = lambda y : self.inner_product(x,y))    

    # Inner product on the dual space. 
    def _dual_inner_product(self, xp1, xp2):
        return self.inner_product(self.from_dual(xp1),self.from_dual(xp2))

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
        return Euclidean(dim, metric = metric)


    