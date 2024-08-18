
import numpy as np
from scipy.stats import norm

# Declare the reals as a vector space structure. 
class Real:

    def __init__(self):
        pass

    @property
    def Dimension(self):
        return 1

    def ToComponents(self,x):
        return x

    def FromComponents(self,c):
        return c


# Define the linear operator class.
class LinearOperator:

    def __init__(self, domain, coDomain, mapping):
        self._domain = domain
        self._coDomain = coDomain
        self._mapping = mapping

    @property
    def Domain(self):
        return _self._domain

    @property
    def CoDomain(self):
        return self._coDomain

    # Return action of operator on vector.
    def __call__(self,x):
        return self._mapping(x)

    # Define scalar multiplication.
    def __mul__(self,scalar):
        return Operator(self.Domain, self.CoDomain, lambda x : scalar * self(x))

    def __rmul__(self,scalar):
        return self * scalar

    # Define scalar division.
    def __div(self,scalar):
        return self * (1/scalar)

    # Define addition. 
    def __add__(self,other):
        assert self.Domain == other.Domain
        assert self.CoDomain == other.CoDomain
        return Linear(self.Domain, self.CoDomain, lambda x : self(x) + other(x))

    # Define subtraction. 
    def __sub__(self,other):
        assert self.Domain == other.Domain
        assert self.CoDomain == other.CoDomain
        return Linear(self.Domain, self.CoDomain, lambda x : self(x) - other(x))
    
    # Define composition. 
    def __matmul__(self,other):
        assert self.Domain == other.CoDomain
        return Linear(other.Domain, self.CoDomain, lambda x : self(other(x)))


# Define linear form class. 
class LinearForm(LinearOperator):

    def __init__(self, domain, mapping):
        super(LinearForm,self).__init__(domain, Real, mapping)


# Define a general vector space.
class Space:

    def __init__(self, dimension, toComponents, fromComponents):
        self._dimension = dimension
        self._toComponents = toComponents
        self._fromComponents = fromComponents

    @property
    def Dimension(self):
        return self._dimension

    def ToComponents(self,x):
        return self._toComponents(x)

    def FromComponents(self,c):
        return self._fromComponents(c)

    @property
    def Zero(self):
        return self.FromComponents(np.zeros(self.Dimension))

    def Random(self,dist = norm()):
        return self.FromComponents(dist.rvs(size = self.Dimension))


# Define a dual vector space. 
class DualSpace(Space):

    def __init__(self, space):       
        self._base = space  
        super(DualSpace,self).__init__(self.Base.Dimension, lambda xp : self._ToComponents(xp), 
                                                            lambda cp : self._FromComponents(cp))        

    @property
    def Base(self):
        return self._base

    def _ToComponents(self, xp):
        n = self.Base.Dimension
        c = np.zeros(n)
        cp = np.zeros(n)
        for i in range(n):
            c[i] = 1
            cp[i] = xp(self.Base.FromComponents(c))
            c[i] = 0
            return cp

    def _FromComponents(self, cp):
        mapping = lambda x : np.dot(cp,self.Base.ToComponents(x))
        return LinearForm(self.Base,mapping)

    @property
    def Zero(self):
        return LinearForm(self._space,lambda x : 0)
            
# Map a vector within a space to a linear form on its double dual
def AsLinearForm(space, x):
    return LinearForm(DualSpace(X), lambda xp : xp(x))


# Define the dual linear operator class. 
class DualLinearOperator(LinearOperator):

    def __init__(self, A):
        domain = DualSpace(A.CoDomain)
        coDomain = DualSpace(A.Domain)
        mapping = lambda yp : LinearForm(operator.Domain,lambda x : (yp @ A)(x))
        super(DualLinearOperator,self).__init__(domain, coDomain, mapping)

        





    
        






    

