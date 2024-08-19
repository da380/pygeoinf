import numpy as np
from scipy.stats import norm as gaussian

# Class for a vector space. 
class Space:

    def __init__(self, dimension, to_components, from_components):
        self._dimension = dimension
        self._to_components = to_components
        self._from_components = from_components

    @property
    def dimension(self):
        return self._dimension

    def to_components(self, x):
        return self._to_components(x)

    def from_components(self, c):
        return self._from_components(c)


    @property 
    def zero(self):
        return self.from_components(np.zeros(self.dimension))

    def random(self,dist = gaussian()):
        return self.from_components(dist.rvs(size = self.dimension))



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



    # Class for a linear form. 
    class LinearForm:
        
        def __init__(self, domain, mapping):
            self._domain = domain    
            self._mapping = mapping

        @property
        def domain(self):
            return self._domain

        def __call__(self,x):
            return self._mapping(x)    

        def __mul__(self,scalar):
            return LinearForm(self.domain, lambda x : scalar * self(x))

        def __rmul__(self,scalar):
            return self * scalar

        def __div__(self, scalar):
            return self * (1 / scalar)

        def __add__(self, other):
            assert self.domain == other.domain
            return LinearForm(self.domain, lambda x : self(x) + other(x))

        def __sub__(self, other):
            assert self.domain == other.domain
            return LinearForm(self.domain, lambda x : self(x) - other(x))            

        def __matmul__(self, operator):
            assert self.domain == operator.codomain
            return LinearForm(operator.domain, lambda x : self(operator(x)))





        

    
