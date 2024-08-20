import numpy as np
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator



# Class for linear forms on a vector space. 
class LinearForm:

    def __init__(self, domain, mapping):
        self._domain = domain    
        self._mapping = mapping    

    # Return the domain of the linear form.
    @property
    def domain(self):
        return self._domain    

    # Return action of the form on a vector. 
    def __call__(self,x):
        return self._mapping(x)

    # Overloads to make LinearForm a vector space. 
    def __mul__(self, s):
        return LinearForm(self.domain, lambda x : s * self(x))

    def __rmul__(self,s):
        return self * s

    def __div__(self, s):
        return self * (1/s)

    def __add__(self, other):
        assert self.domain == other.domain        
        return LinearForm(self.domain, lambda x : self(x) + other(x))

    def __sub__(self, other):
        assert self.domain == other.domain        
        return LinearForm(self.domain, lambda x : self(x) - other(x))         

    def __matmul__(self, other):
        return self(other)

    def __str__(self):
        return self.domain.dual.to_components(self).__str__()


# Class for linear operators between two vector spaces. 
class LinearOperator:

    def __init__(self, domain, codomain, mapping):        
        self._domain = domain
        self._codomain = codomain        
        self._mapping = mapping  
    
    # Return the domain of the linear operator. 
    @property
    def domain(self):
        return self._domain

    # Return the codomain of the linear operator. 
    @property 
    def codomain(self):
        return self._codomain    

    # Return the action of the operator on a vector. 
    def __call__(self,x):        
        return self._mapping(x)

    # Overloads to make LinearOperators a vector space and algebra.
    def __mul__(self, s):
        return LinearOperator(self.domain, self.codomain, lambda x : s * self(x))

    def __rmul__(self,s):
        return self * s

    def __div__(self, s):
        return self * (1/s)

    def __add__(self, other):
        assert self.domain == other.domain
        assert self.codomain == other.codomain
        return LinearOperator(self.domain, self.codomain, lambda x : self(x) + other(x))

    def __sub__(self, other):
        assert self.domain == other.domain
        assert self.codomain == other.codomain
        return LinearOperator(self.domain, self.codomain, lambda x : self(x) - other(x))        

    def __matmul__(self,other):
        if isinstance(other, LinearOperator):
            assert self.domain == other.codomain
            return LinearOperator(other.domain, self.codomain, lambda x : self(other(x)))
        else:
            return self(other)
            
    # For writing operator, convert to dense matrix.
    def __str__(self):
        return self.to_dense_matrix.__str__()

    # Return the operator relative to the bases for the domain and co-domain
    @property
    def to_matrix(self):
        domain = VectorSpace(domain.size)
        codomain = VectorSpace(codomain.size)
        mapping = lambda c : codomain.to_components(self @ domain.from_components(c))
        return LinearOperator(domain, codomain, mapping)

    # Return the operator as a dense matrix. 
    @property
    def to_dense_matrix(self):
        A = np.zeros((self.codomain.dimension, self.domain.dimension))
        c = np.zeros(self.domain.dimension)        
        for i in range(self.domain.dimension):
            c[i] = 1            
            A[:,i] = self.codomain.to_components(self @ self.domain.from_components(c))
            c[i] = 0
        return A

    # Define the abstract dual mapping. 
    @property
    def _default_dual_mapping(self):
        return 

    # Return the dual of the linear mapping.
    @property
    def dual(self):                        
        mapping = lambda yp : (lambda x : yp(self @ x))
        return LinearOperator(self.codomain.dual, self.domain.dual, mapping)              

# Class for vector spaces.
class VectorSpace:

    def __init__(self, dimension, to_components = lambda x : x, from_components = lambda x : x):
        self._dimension = dimension
        self._to_components = to_components
        self._from_components = from_components    

    # Return the dimension of the space. 
    @property
    def dimension(self):
        return self._dimension

    # Map a vector to its components. 
    def to_components(self,x):
        return self._to_components(x)

    # Map components to a vector. 
    def from_components(self,c):
        return self._from_components(c)    
    
    # Map a linear for on the space to its components relative to the 
    # induced basis. 
    def _to_dual_components(self,xp):
        n = self.dimension
        c = np.zeros(n)
        cp = np.zeros(n)
        for i in range(n):
            c[i] = 1
            cp[i] = xp(self.from_components(c))
            c[i] = 0
        return cp

    # Return the zero vector. 
    @property
    def zero(self):
        return self.from_components(np.zeros(self.dimension))

    # Return a vector whose components samples from a given distribution. 
    def random(self, dist = norm()):
        return self.from_components(norm.rvs(size = self.dimension))


    # Return the dual space. 
    @property 
    def dual(self):
        to_components = self._to_dual_components
        from_components = lambda c : LinearForm(self, lambda x : np.dot(self.to_components(x),c))
        return VectorSpace(self.dimension, to_components, from_components)

