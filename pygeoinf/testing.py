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

    def _random_components(self):
        return norm().rvs(size = (self.dim,1))

    def random(self):
        return self.from_components(self._random_components())

    def check(self, /, *,trials = 1, rtol = 1e-9):
        okay = True
        for i in range(trials):
            okay = okay and self._check(rtol)
        return okay            
        
    def _check(self, rtol):
        c1 = self._random_components()
        c2 = self.to_components(self.from_components(c1))        
        return np.linalg.norm(c1-c2) < rtol * np.linalg.norm(c1)
0


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

_Real = Real()


def RealN(dim):
    return VectorSpace(dim, lambda x : x.reshape(dim,1), lambda x : x.reshape(dim,))


class LinearOperator:

    def __init__(self, domain, codomain, /, *, mapping = None,
                 dual_mapping = None, adjoint_mapping = None, matrix = None):
        self._domain = domain
        self._codomain = codomain        
        self._matrix = matrix        
        if matrix is not None:
            self._mapping = self._mapping_from_matrix                    
        else:
            self._mapping = mapping
            self._dual_mapping = dual_mapping
            self._adjoint_mapping = adjoint_mapping

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
            matrix[:,i] = self.codomain.to_components(y)[:,0]
            cx[i] = 0
        return matrix            

    def __call__(self, x):
        return self._mapping(x)

    def __mul__(self, a):
        return LinearOperator(self.domain, self.codomain, mapping = lambda x : a * self(x))
        
    def __rmul__(self, s):
        return self * s

    def __add__(self, other):        
        if self.domain != other.domain:
            raise ValueError("Domains must be equal")
        if self.codomain != other.codomain:
            raise ValueError("Codomains must be equal")
        return LinearOperator(self.domain, self.codomain, mapping = lambda x : self(x) + other(x))

    def __sub__(self, other):       
        if self.domain != other.domain:
            raise ValueError("Domains must be equal")
        if self.codomain != other.codomain:
            raise ValueError("Codomains must be equal")
        return LinearOperator(self.domain, self.codomain, mapping = lambda x : self(x) - other(x))  

    def __str__(self):
        return self.matrix.__str__()


class DualVector(LinearOperator):

    def __init__(self, domain, /, *, mapping = None, matrix = None):
        super().__init__(domain, _Real, mapping = mapping, matrix=matrix )


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
        return xp.matrix.T

    def _dual_from_components(self,cp):
        return DualVector(self._space, matrix = cp.reshape(1,self.dim))


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
                    mapping = lambda yp : DualVector(operator.domain, mapping = lambda x: yp(operator(x)))                                    
                else:                    
                    mapping = operator._dual_mapping
                super().__init__(domain, codomain, mapping=mapping)
            else:                
                super().__init__(domain, codomain, matrix = operator._matrix.T)

            
    @property
    def formed_from(self):
        return self._operator

    
class HilbertSpace(VectorSpace):

    def __init__(self, dim, to_components, from_components, inner_product, /, *, to_dual = None, from_dual = None):

        super().__init__(dim, to_components, from_components)

        self._inner_product = inner_product

        if to_dual is None:
            self._to_dual = lambda x : DualVector(self, mapping = lambda y : self.inner_product(x,y))
        else:
            self._to_dual = to_dual

        if from_dual is None:
            raise NotImplementedError("To be done!")
        else:
            self._from_dual = from_dual

    @staticmethod
    def from_vector_space(space, inner_product, /, *, to_dual = None, from_dual = None):
        return HilbertSpace(space.dim, space.to_components, space.from_components, 
                            inner_product, to_dual=to_dual, from_dual=from_dual)

    @property
    def vector_space(self):
        return VectorSpace(self.dim, self.to_components, self.from_components)
        
    def inner_product(self, x1, x2):
        return self._inner_product(x1,x2)

    def norm(self,x):
        return np.sqrt(self.inner_product(x,x))
        

    def to_dual(self, x):
        return self._to_dual(x)

    def from_dual(self, xp):        
        return self._from_dual(xp)


def Euclidean(dim):
    RN = RealN(dim)
    inner_product = lambda x1, x2 : np.dot(x1,x2)
    to_dual = lambda x : DualVector(RN, matrix = x)
    from_dual = lambda xp : xp.matrix[0,:]
    return HilbertSpace.from_vector_space(RN, inner_product, to_dual=to_dual, from_dual=from_dual)



def is_hilbert_space_operator(operator):
    return isinstance(operator.domain, HilbertSpace) and isinstance(operator.codomain, HilbertSpace)

class DualHilbertSpace(HilbertSpace):

    def __init__(self, space):

        if isinstance(space, DualHilbertSpace):
            original = space.formed_from
            super().__init__(original.dim, original.to_components, original.from_components, 
                             original.inner_product, to_dual=original.to_dual, from_dual=original.from_dual)
        else:
            self._space = space
            dual_space = DualVectorSpace(space)
            inner_product = lambda xp1, xp2 : space.inner_product(space.from_dual(xp1), space.from_dual(xp2))            
            super().__init__(space.dim, dual_space.to_components, dual_space.from_components,
                             inner_product, to_dual=space.from_dual, from_dual=space.to_dual)


    @property
    def formed_from(self):
        return self._space


class AdjointOperator(LinearOperator):

    def __init__(self, operator):

        if not is_hilbert_space_operator(operator):
            raise ValueError("Adjoint defined only for operators between Hilbert Spaces.")

        if isinstance(operator, AdjointOperator):
            super().__init__(operator.formed_from.domain, operator.formed_from.codomain, 
                             mapping = operator.formed_from.mapping)
        else:
            self._operator = operator
            domain = DualHilbertSpace(operator.codomain)
            codomain = DualHilbertSpace(operator.domain)
            if operator._adjoint_mapping is None:
                dual_operator = DualOperator(operator)
                print(dual_operator)
                mapping = lambda y : codomain.to_dual(dual_operator(domain.from_dual(y)))
            else:
                mapping = operator._adjoint_mapping
            super().__init__(domain, codomain, mapping=mapping)


    @property
    def formed_from(self):
        return self._operator

    

        

