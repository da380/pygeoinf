import numpy as np
from scipy.stats import norm


class VectorSpace:

    def __init__(self, dim, to_components, from_components):
        self._dim = dim
        self._to_components = to_components
        self._from_components = from_components
    
    @property
    def dim(self):
        return self._dim

    def to_components(self,x):
        return self._to_components(x)

    def from_components(self,c):
        return self._from_components(c)

    def random(self):
        return self.from_components(norm().rvs(size = self.dim))


class Real(VectorSpace):
    def __init__(self):
        super().__init__(1, self._to_components_local, self._from_components_local)

    def _to_components_local(self, x):
        return np.array([x])

    def _from_components_local(self,c):
        if isinstance(c, np.ndarray):
            return c[0]
        else:
            return c
    

class LinearOperator:

    def __init__(self, domain, codomain, /, *, mapping = None,
                 dual_mapping = None, matrix = None):
        self._domain = domain
        self._codomain = codomain        
        self._matrix = matrix        
        if matrix is not None:
            self._mapping = self._mapping_from_matrix                    
        else:
            self._mapping = mapping
            self._dual_mapping = dual_mapping

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def matrix(self):
        if self._matrix is None:            
            return self._compute_matrix()            
        else:            
            return self._matrix

    def store_matrix(self):
        if self._matrix is None:
            self._matrix = self._compute_matrix()

    def _mapping_from_matrix(self,x):        
        cx = self.domain.to_components(x)
        cy = self.matrix @ cx
        return self.codomain.from_components(cy)
        
    def _compute_matrix(self):        
        matrix = np.zeros((self.codomain.dim, self.domain.dim))        
        cx = np.zeros(self.domain.dim)                        
        for i in range(self.domain.dim):
            cx[i] = 1
            x = self.domain.from_components(cx)
            y = self(x)                
            matrix[:,i] = self.codomain.to_components(y)
            cx[i] = 0
        return matrix            

    def __call__(self, x):
        return self._mapping(x)

    def __mul__(self, a):
        return LinearOperator(self.domain, self.codomain, mapping = lambda x : a * self(x))
        
    def __rmul__(self, s):
        return self * s

    def __add__(self, other):        
        return LinearOperator(self.domain, self.codomain, mapping = lambda x : self(x) + other(x))


    def __sub__(self, other):       
        return LinearOperator(self.domain, self.codomain, mapping = lambda x : self(x) - other(x))  

    def __str__(self):
        return self.matrix.__str__()


class DualVector(LinearOperator):

    def __init__(self, domain, /, *, mapping = None, matrix = None):
        super().__init__(domain, Real(), mapping = mapping, matrix=matrix )


class DualVectorSpace(VectorSpace):

    def __init__(self, space):
        
        if isinstance(space, DualVectorSpace):                        
            super().__init__(space.formed_from.dim, space.formed_from.to_components, space.formed_from.from_components)
        else:            
            self._space = space
            super().__init__(space.dim, self._dual_to_components, self._dual_from_components)
            
    @property
    def formed_from(self):
        return self._space

    def _dual_to_components(self, xp):
        return xp.matrix

    def _dual_from_components(self,cp):
        return DualVector(self._space, matrix = cp)


class DualOperator(LinearOperator):

    def __init__(self, operator):
        
        if isinstance(operator, DualOperator):
            super().__init__(operator.formed_from.domain, operator.formed_from.codomain, 
                             mapping = operator.formed_from, matrix = operator.formed_from._matrix)
        else:            
            self._operator = operator
            domain = DualVectorSpace(operator.codomain)
            codomain = DualVectorSpace(operator.domain)            
            if operator._matrix is None:
                if operator._dual_mapping is None:                
                    print("Dual symbolically")
                    mapping = lambda yp : DualVector(operator.domain, mapping = lambda x: yp(operator(x)))                                    
                else:
                    print("Dual from mapping")
                    mapping = operator._dual_mapping
                super().__init__(domain, codomain, mapping=mapping)
            else:
                print("Dual from matrix")
                super().__init__(domain, codomain, matrix = operator._matrix.T)

            
    @property
    def formed_from(self):
        return self._operator

    




