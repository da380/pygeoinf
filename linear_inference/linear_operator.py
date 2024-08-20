import numpy as np
from linear_inference.interfaces import AbstractLinearOperator
from linear_inference.linear_form import LinearForm

class DualLinearOperator(AbstractLinearOperator):

    def __init__(self, base, mapping):
        self._base = base        
        self._mapping = mapping

    @property
    def domain(self):
        return self._base.codomain.dual

    @property
    def codomain(self):
        return self._base.domain.dual

    @property
    def dual(self):
        return self._base

    def __call__(self, yp):
        return self._mapping(yp)

    # Overloads to make LinearOperators a vector space and algebra.
    def __mul__(self, s):
        return DualLinearOperator(s * self.dual, lambda xp : s * self(xp))

    def __add__(self, other):
        assert self.domain == other.domain
        assert self.codomain == other.codomain
        return DualLinearOperator(self.dual + other.dual, lambda xp : self(xp) + other(xp))

    def __matmul__(self,other):        
        assert self.domain == other.codomain
        return DualLinearOperator(other.dual @ self.dual, lambda xp : self(other(xp)))

'''
# Class for adjoint linear operators between two Hilbert spaces. 
class AdjointLinearOperator(AbstractLinearOperator):

    def __init__(self, base, domain, codomain, mapping):
        self._base = base
        self._domain = domain
        self._codomain = codomain
        self._mapping = mapping


    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dual(self):
        return self._base

    def __call__(self, yp):
        return self._mapping(yp)

    # Overloads to make LinearOperators a vector space and algebra.
    def __mul__(self, s):
        return DualLinearOperator(s * self.dual, self.domain, self.codomain, lambda xp : s * self(xp))

    def __add__(self, other):
        assert self.domain == other.domain
        assert self.codomain == other.codomain
        return DualLinearOperator(self.dual + other.dual, self.domain, self.codomain, lambda xp : self(xp) + other(xp))

    def __matmul__(self,other):        
        assert self.domain == other.codomain
        return DualLinearOperator(other.dual @ self.dual, other.domain, self.codomain, lambda xp : self(other(xp)))
'''

# Class for linear operators between two vector spaces. 
class LinearOperator(AbstractLinearOperator):

    def __init__(self, domain, codomain, mapping, /, *, dual_mapping = None, adjoint_mapping = None):        
        self._domain = domain
        self._codomain = codomain        
        self._mapping = mapping  
        self._dual_mapping = dual_mapping
        self._adjoint_mapping = adjoint_mapping
    
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
        if self._dual_mapping is None:
            dual_mapping = lambda yp : LinearForm(self.domain, lambda x : yp(self(x)))
        else:
            dual_mapping = self._dual_mapping
        return DualLinearOperator(self, dual_mapping)
        
    # Return the action of the operator on a vector. 
    def __call__(self,x):        
        return self._mapping(x)

    # Overloads to make LinearOperators a vector space and algebra.
    def __mul__(self, s):
        return LinearOperator(self.domain, self.codomain, lambda x : s * self(x))

    def __add__(self, other):
        assert self.domain == other.domain
        assert self.codomain == other.codomain
        return LinearOperator(self.domain, self.codomain, lambda x : self(x) + other(x))    

    def __matmul__(self,other):        
        assert self.domain == other.codomain
        return LinearOperator(other.domain, self.codomain, lambda x : self(other(x)))
    
    
