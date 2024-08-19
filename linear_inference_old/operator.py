import numpy as np


# Class for a linear operator.
class LinearOperator:

    def __init__(self, domain, codomain , mapping):
        self._domain = domain
        self._codomain = codomain        
        self._mapping = mapping


    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    def __call__(self,x):
        return self._mapping(x)

    def __mul__(self,scalar):
        return LinearOperator(self.domain, self.codomain, lambda x : scalar * self(x))

    def __rmul__(self,scalar):
        return self * scalar

    def __div(self,scalar):
        return self * (1/scalar)

    def __add__(self,other):
        assert self.domain == other.domain
        assert self.codomain == other.codomain
        return LinearOperator(self.domain, self.codomain, lambda x : self(x) + other(x))

    def __sub__(self,other):
        assert self.domain == other.domain
        assert self.codomain == other.codomain
        return LinearOperator(self.domain, self.codomain, lambda x : self(x) - other(x))        
    
    
    def __matmul__(self,other):
        assert self.domain == other.codomain
        return LinearOperator(other.domain, self.codomain, lambda x : self(other(x)))



    