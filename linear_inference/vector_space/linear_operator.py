if __name__ == "__main__":
    pass


import numpy as np
from scipy.sparse.linalg import LinearOperator as SciPyLinearOperator
from linear_inference.vector_space.linear_form import LinearForm

# Class for linear operators between two vector spaces. 
class LinearOperator:

    def __init__(self, domain, codomain, mapping, /, *, dual_mapping = None, adjoint_mapping = None, dual_base = None, adjoint_base = None):        
        self._domain = domain
        self._codomain = codomain        
        self._mapping = mapping  
        self._dual_mapping = dual_mapping
        self._adjoint_mapping = adjoint_mapping
        self._dual_base = dual_base
        self._adjoint_base = adjoint_base

    # Return a self adjoint operator. 
    @staticmethod
    def self_adjoint_operator(domain, mapping):
        return LinearOperator(domain, domain, mapping, adjoint_mapping = mapping)

    # Return a self-dual operator. 
    @staticmethod
    def self_dual_operator(domain, mapping):        
        return LinearOperator(X, X.dual, mapping, dual_mapping = mapping)
    
    # Return the domain of the linear operator. 
    @property
    def domain(self):
        return self._domain

    # Return the codomain of the linear operator. 
    @property 
    def codomain(self):
        return self._codomain    

    # Return the dual operator.
    @property
    def dual(self):
        if self._dual_base is None:
            if self._dual_mapping is None:
                if self._adjoint_mapping is None:
                    dual_mapping = lambda yp : LinearForm(self.domain, mapping = lambda x : yp(self(x)))
                else:
                    dual_mapping = lambda yp : self.domain.to_dual(self.adjoint(self.codomain.from_dual(yp)))
            else:
                dual_mapping = self._dual_mapping
            return LinearOperator(self.codomain.dual, self.domain.dual, dual_mapping, dual_mapping = self._mapping, dual_base = self)            
        else:            
            return self._dual_base
        
    # Return the adjoint. 
    @property 
    def adjoint(self):
        if self._adjoint_base is None:
            if self._adjoint_mapping is None:
                adjoint_mapping = lambda y : self.domain.from_dual(self.dual(self.codomain.to_dual(y)))
            else:
                adjoint_mapping = self._adjoint_mapping
            return LinearOperator(self.codomain, self.domain, adjoint_mapping, adjoint_mapping = self._mapping, adjoint_base = self)
        else:
            return self._adjoint_base
        
    # Return the action of the operator on a vector. 
    def __call__(self,x):        
        return self._mapping(x)

    # Overloads to make LinearOperators a vector space and algebra.
    def __mul__(self, s):
        return LinearOperator(self.domain, self.codomain, lambda x : s * self(x))

    def __rmul__(self, s):
        return self * s

    def __div__(self, s):
        return self * (1 /s)        

    def __add__(self, other):
        assert self.domain == other.domain
        assert self.codomain == other.codomain
        return LinearOperator(self.domain, self.codomain, lambda x : self(x) + other(x))   

    def __sub__(self, other): 
        return self + (-1 * other)
 
    def __matmul__(self,other):        
        assert self.domain == other.codomain
        return LinearOperator(other.domain, self.codomain, lambda x : self(other(x)))

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
        if self._adjoint_mapping is None:
            return SciPyLinearOperator(shape, matvec = matvec)
        else:
            rmatvec = lambda y : self.domain.to_components(self.adjoint(self.codomain.from_components(y)))
            return SciPyLinearOperator(shape, matvec = matvec, rmatvec = rmatvec)

    # Convert to dense matrix for printing values. 
    def __str__(self):
        return self.to_dense_matrix.__str__()

    
    
