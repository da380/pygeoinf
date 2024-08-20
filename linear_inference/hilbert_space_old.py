import numpy as np
from abc import ABC
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator

class AbstractVectorSpace(ABC):    

    @property
    def dimension(self):
        pass

    def to_components(self,x):
        pass

    def from_components(self,x):
        pass

    @property
    def dual(self):
        pass

    @property
    def zero(self):
        return self.from_components(np.zeros(self.dimension))

    # Return a vectors whose components are uncorrelated samples from a distribution. 
    def random(self, dist = norm()):
        return self.from_components(norm.rvs(size = self.dimension))


class AbstractHilbertSpace(AbstractVectorSpace):
    pass




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

    def __init__(self, domain, codomain, mapping, /, *, dual_mapping = None, adjoint_mapping = None):        
        self._domain = domain
        self._codomain = codomain
        if not self.hilbert_space_operator:
            self._domain = domain.base
            self._codomain = codomain.base    
        self._mapping = mapping  
        self._dual_mapping = dual_mapping
        self._adjoint_mapping = adjoint_mapping
        if self.adjoint_set:
            assert self.hilbert_space_operator, "adjoint mapping only defined for Hilbert space operators"

    # Return the domain of the linear operator. 
    @property
    def domain(self):
        return self._domain

    # Return the codomain of the linear operator. 
    @property 
    def codomain(self):
        return self._codomain

    # Returns true for mappings between Hilbert spaces.
    @property
    def hilbert_space_operator(self):
        return isinstance(self.domain,AbstractHilbertSpace) and isinstance(self.codomain,AbstractHilbertSpace)

    # Returns true if the dual mapping has been set directly. 
    @property
    def dual_set(self):
        return self._dual_mapping is not None

    # Return true if the adjoint mapping has been set directly. 
    @property
    def adjoint_set(self):
        return self._adjoint_mapping is not None

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
    def _abstract_dual_mapping(self):
        return lambda yp : (lambda x : yp(self @ x))

    # Return the dual of the linear mapping.
    @property
    def dual(self):                
        if self.dual_set:
            return LinearOperator(self.codomain.dual, self.domain.dual, self._dual_mapping, dual_mapping= self._mapping)
        elif self.adjoint_set:
            mapping = lambda yp : self.domain.to_dual(self.adjoint(self.codomain.from_dual(yp)))
            adjoint_mapping = lambda xp : self.codomain.to_dual(self(self.domain.from_dual(xp)))
            return LinearOperator(self.codomain.dual, self.domain.dual, mapping, adjoint_mapping= adjoint_mapping)
        else:
            if self.hilbert_space_operator:
                mapping = lambda yp : self.domain.dual.from_linear_form(self._abstract_dual_mapping(self.codomain.dual.to_linear_form(yp)))                
            else:                
                mapping = self._abstract_dual_mapping        
        return LinearOperator(self.codomain.dual, self.domain.dual, mapping)          

    # Return the adjoint for Hilbert space operators. 
    @property
    def adjoint(self):
        assert self.hilbert_space_operator
        if self.adjoint_set:            
            return LinearOperator(self.codomain.dual, self.domain.dual, self._adjoint_mapping, adjoint_mapping = self._mapping)
        else:
            mapping = lambda y : self.domain.dual.from_dual(self.dual(self.codomain.dual.to_dual(y)))
            return LinearOperator(self.codomain.dual, self.domain.dual, mapping, adjoint_mapping = self._mapping)



# Class for vector spaces. 
class VectorSpace(AbstractVectorSpace):

    def __init__(self, dimension, to_components = lambda x : x, from_components = lambda x : x):
        self._dimension = dimension
        self._to_components = to_components
        self._from_components = from_components

    # Return the space. 
    @property
    def base(self):
        return self

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

    # Return the action of a dual vector on a vector. 
    def duality_product(self,xp,x):
        return xp(x)

    # Return the zero vector. 
    @property
    def zero(self):
        return self.from_components(np.zeros(self.dimension))

    # Return a vectors whose components are uncorrelated samples from a distribution. 
    def random(self, dist = norm()):
        return self.from_components(norm.rvs(size = self.dimension))

    
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

    # Return the dual space. 
    @property 
    def dual(self):
        to_components = self._to_dual_components
        from_components = lambda c : LinearForm(self, lambda x : np.dot(self.to_components(x),c))
        return VectorSpace(self.dimension, to_components, from_components)



# Class for a Hilbert space with a concrete representation of its dual. 
class HilbertSpace(AbstractHilbertSpace, VectorSpace):

    def __init__(self, space, dual, to_dual, from_dual):
        self._space = space
        self._dual = dual
        self._to_dual = to_dual
        self._from_dual = from_dual    
        super(HilbertSpace,self).__init__(space.dimension, space.to_components, space.from_components)
        

    # Return the underlying vector space. 
    @property
    def base(self):
        return self._space

    # Map a vector to its representation in the dual space. 
    def to_dual(self,x):
        return self._to_dual(x)

    # Return a vector from its dual representation. 
    def from_dual(self,xp):
        return self._from_dual(xp)

    # Return the action of a dual vector on a vector. 
    def duality_product(self, xp, x):
        return np.dot(self.dual.to_components(xp), self.to_components(x))

    # Return the inner product of two vectors. 
    def inner_product(self, x1, x2):        
        return self.duality_product(self.to_dual(x1),x2)

    # Return the norm of a vector. 
    def norm(self, x):
        return np.sqrt(self.inner_product(x,x))

    # Map a vector to a linear form on the space. 
    def to_linear_form(self,x):
        return LinearForm(self, lambda y : self.inner_product(x,y))

    # Determine the vector that represents a linear form.
    def from_linear_form(self, f):
        cp = self._dual.to_components(f)


    @property
    def dual(self):        
        return HilbertSpace(self._dual, self._space, self.from_dual, self.to_dual)


    @staticmethod
    def from_inner_product(space, inner_product):
        n = space.dimension
        Rn = VectorSpace(n, lambda x : x, lambda x : x)
        metric = np.zeros((n,n))
        c1 = np.zeros(n)
        c2 = np.zeros(n)
        for i in range(n):
            c1[i] = 1
            x1 = space.from_components(c1)
            metric[i,i] = inner_product(x1,x1)
            for j in range(i+1,n):
                c2[j] = 1
                x2 = space.from_components(c2)
                tmp = inner_product(x1,x2)
                metric[i,j] = tmp
                metric[j,i] = tmp
                c2[j] = 0
            c1[i] = 0
        factor = cho_factor(metric)        
        to_dual = lambda x : metric @ space.to_components(x)
        from_dual = lambda xp : space.from_components( cho_solve(factor, xp))
        return HilbertSpace(space, Rn , to_dual, from_dual)

            


