"""
This module contains the definition of the LinearOperator class. 
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator as SciPyLinearOperator
from pygeoinf.linear_form import LinearForm


if __name__ == "__main__":
    pass


class LinearOperator:
    """
    Class for linear operators between real vector spaces. A linear operator is represented in 
    terms of the following information:

    (1) The domain of the operator, this being an instance of VectorSpace (or a derived class).
    (2) The codomain of the operator, this being an instance of VectorSpace (or a derived class).
    (3) A functor that performs the action of the operator on an element of its domain. 

    The dual of the linear operator can be deduced automatically, but the method is inefficient. 
    If the action of the dual is known this can be specified. 

    For operators between Hilbert spaces, the adjoint of the linear operator can be deduced 
    automatically, but the method is inefficient. If the action of the dual is known this can
    be specified. The adjoint can also be efficiently determined from a specified dual mapping, 
    and conversely. 
    """

    def __init__(self, domain, codomain, mapping, /, *, dual_mapping = None,
                 adjoint_mapping = None, dual_base = None, adjoint_base = None):        
        """
        Args:
            domain (VectorSpace): The domain of the operator.
            codomain (VectorSpace): The codomain of the operator.
            mapping: A functor representing the action of the operator. 
            dual_mapping: A callable oject representing the action of the dual operator. 
            adjoint_mapping: A functor representing the action of the adjoint operator. 
            dual_base (bool): Used internally to record that an operator is the dual of another. 
            adjoint_base (bool): Used internally to record that an operator is the adjoint of another. 
        """
        self._domain = domain
        self._codomain = codomain        
        self._mapping = mapping  
        self._dual_mapping = dual_mapping
        self._adjoint_mapping = adjoint_mapping
        self._dual_base = dual_base
        self._adjoint_base = adjoint_base        
                    
    @staticmethod
    def identity(domain):
        return LinearOperator(domain, domain,lambda x:x, 
                              dual_mapping=lambda x:x, 
                              adjoint_mapping=lambda x:x)


    @staticmethod
    def self_adjoint(domain, mapping):
        """
        Return a self-adjoint operator on a Hilbert space.

        Args:
            domain (HilbertSpace): The domain and codomain of the operator. 
            mapping: A functor that implements the action of the operator. 

        Return:
            LinearOperator: The self-adjoint operator on the domain. 

        Note:
            The function does not check that the mapping provided does actually
            define a self-adjoint operator.
        """
        return LinearOperator(domain, domain, mapping, adjoint_mapping = mapping)
    
    @staticmethod  
    def self_dual(domain, mapping):        
        """
        Return a self-dual operator from a vector space to its dual. 

        Args:
            domain (VectorSpace): The domain of the operator. 
            mapping: A functor that implements the action of the operator. 

        Return:
            LinearOperator: The self-dual operator on the domain. 

        Note:
            The function does not check that the mapping provided does actually
            define a self-dual operator.
        """
        return LinearOperator(domain, domain.dual, mapping, dual_mapping = mapping)

    @staticmethod
    def from_diagonal_values(domain, codomain, diags):
        """
        Returns a linear operator between spaces of the same dimension
        whose matrix representation is diagonal. 

        Args:
            domain (VectorSpace): The domain of the operator. 
            codomain (VectorSpace): The codomain of the operator. 
            diags (ArrayLike): The diagonal values within the matrix representation. 

        Returns:
            LinearOperator: The linear operator so formed. 

        Raises:
            ValueError: If dimensions of domain and codomain are different. 
        """
        if domain.dim != codomain.dim:
            raise ValueError("Domain and codomain must have the same dimensions")        
        if domain.dim != diags.size:
            raise ValueError("Diagonal values have the wrong size.")
        mapping = lambda x : codomain.from_components(diags * domain.to_components(x))        
        dual_mapping = lambda yp :domain.dual.from_components(diags * codomain.dual.to_components(yp))
        return LinearOperator(domain, codomain, mapping, dual_mapping=dual_mapping)
    
    @property
    def domain(self):
        """The domain of the operator."""
        return self._domain

    @property 
    def codomain(self):
        """The codomain of the operator."""
        return self._codomain    

    @property
    def hilbert_operator(self):
        """True if the operator is between Hilbert spaces."""
        return self._hilbert_operator

    @property
    def dual(self):
        """The dual of the operator."""
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
        
    @property 
    def adjoint(self):
        """The adjoint of an operator between Hilbert spaces."""
        if self._adjoint_base is None:
            if self._adjoint_mapping is None:
                adjoint_mapping = lambda y : self.domain.from_dual(self.dual(self.codomain.to_dual(y)))
            else:
                adjoint_mapping = self._adjoint_mapping
            return LinearOperator(self.codomain, self.domain, adjoint_mapping, adjoint_mapping = self._mapping, adjoint_base = self)
        else:
            return self._adjoint_base

    @property
    def to_dense_matrix(self):
        """The operator as a dense matrix."""
        A = np.zeros((self.codomain.dim, self.domain.dim))
        c = np.zeros(self.domain.dim)        
        for i in range(self.domain.dim):
            c[i] = 1            
            A[:,i] = self.codomain.to_components(self(self.domain.from_components(c)))
            c[i] = 0
        return A
    
    @property
    def to_scipy_sparse_linear_operator(self):
        """The operator converted to a scipy.sparse.LinearOperator object."""
        shape = (self.codomain.dim, self.domain.dim)    
        matvec = lambda x : self.codomain.to_components(self(self.domain.from_components(x))) 
        if self._adjoint_mapping is None:
            return SciPyLinearOperator(shape, matvec = matvec)
        else:
            rmatvec = lambda y : self.domain.to_components(self.adjoint(self.codomain.from_components(y)))
            return SciPyLinearOperator(shape, matvec = matvec, rmatvec = rmatvec)

        
    def __call__(self,x):        
        """The action of the operator on an element of its domain."""
        return self._mapping(x)
    
    def __mul__(self, s):
        """Multiplication of an operator by a scalar."""
        return LinearOperator(self.domain, self.codomain, lambda x : s * self(x))

    def __rmul__(self, s):
        """Multiplication of an operator by a scalar."""        
        return self * s

    def __div__(self, s):
        """Division of an operator by a scalar."""
        return self * (1 /s)        

    def __add__(self, other):
        """Addition of two operators with equal domains and codomains."""
        assert self.domain == other.domain
        assert self.codomain == other.codomain
        return LinearOperator(self.domain, self.codomain, lambda x : self(x) + other(x))   

    def __sub__(self, other): 
        """Subtraction of two operators with equal domains and codomains."""
        return self + (-1 * other)
 
    def __matmul__(self,other):        
        """Composition of linear operators with compatible domains and codomains."""
        assert self.domain == other.codomain
        return LinearOperator(other.domain, self.codomain, lambda x : self(other(x)))
     
    def __str__(self):
        """Print the operator as a dense matrix."""
        return self.to_dense_matrix.__str__()

    
    
