from abc import ABC
import numpy as np
from scipy.stats import norm
from scipy.sparse.linalg import LinearOperator
from linear_inference.linear_form import LinearForm

class AbstractVectorSpace(ABC):

    # Return dimension of the space. 
    @property
    def dimension(self):
        pass

    # Return the dual of the vector space. 
    @property
    def dual(self):
        pass

    # Return the zero vector.
    @property
    def zero(self):
        return self.from_components(np.zeros(self.dimension))

    # Map vector to its components in basis. 
    def to_components(self, x):
        pass

    # Map components to vector using basis. 
    def from_components(self, c):
        pass

    # Return a vector whose components samples from a given distribution. 
    def random(self, dist = norm()):
        return self.from_components(norm.rvs(size = self.dimension))



class AbstractHilbertSpace(AbstractVectorSpace):

    # Return the inner product of two vectors.
    def inner_product(self, x1, x2):
        pass

    # Return the norm of a vector. 
    def norm(self, x):
        return np.sqrt(self.inner_product(x,x))

    # Map a dual vector to its representation. 
    def from_dual(self, xp):
        pass

    # Map a vector to a dual vector. 
    def to_dual(self, x):
        return LinearForm(self, lambda y : self.inner_product(x,y))    

    

class AbstractLinearOperator(ABC):

    # Return domain of the operator.
    @property
    def domain(self):
        pass

    # Return the codomain of the operator. 
    @property
    def codomain(self):
        pass

    # Return the dual of the operator. 
    @property
    def dual(self):
        pass

    # Return the action of the operator on a vector. 
    def __call__(self, x):
        pass

    # Overloads to make operators into a vector space and algebra. 
    def __mul__(self, s):
        pass

    def __rmul__(self, s):
        return self * s

    def __div__(self, s):
        return self * (1 /s)

    def __add__(self, other):
        return 

    def __sub__(self, other): 
        return self + (-1 * other)

    def __matmul__(self, other):
        pass

    # Return the operator as a dense matrix. 
    @property
    def to_dense_matrix(self):
        A = np.zeros((self.codomain.dimension, self.domain.dimension))
        c = np.zeros(self.domain.dimension)        
        for i in range(self.domain.dimension):
            c[i] = 1            
            A[:,i] = self.codomain.to_components(self(self.domain.from_components(c)))
            c[i] = 0
        return A

    # Return the operator as a scipy.sparse LinearOperator object.
    @property
    def to_scipy_sparse_linear_operator(self):
        shape = (self.codomain.dimension, self.domain.dimension)    
        matvec = lambda x : self.codomain.to_components(self(self.domain.from_components(x))) 
        return LinearOperator(shape, matvec = matvec)

    # For writing operator, convert to dense matrix.
    def __str__(self):
        return self.to_dense_matrix.__str__()

