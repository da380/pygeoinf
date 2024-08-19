import numpy as np
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve

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

    # Return the dual of the linear mapping.
    @property
    def dual(self):
        domain = self.codomain.dual
        codomain = self.domain.dual
        mapping = lambda yp : (lambda x : yp(self @ x))
        return LinearOperator(domain, codomain, mapping)  

    # Return the "covariant-dual" for Hilbert space operators
    @property
    def covariant_dual(self):
        domain = self.domain.dual_representation
        codomain = self.codomain.dual_representation
        mapping = lambda x : codomain.to_dual_representation( self(domain.from_dual_representation(x)))
        return LinearOperator(domain, codomain, mapping)

    # Return the adjoint for Hilbert space operators. 
    @property
    def adjoint(self):
        domain = self.codomain
        codomain = self.domain
        def mapping(x):            
            xp = domain.dual_representation.to_linear_form(domain._to_dual_representation(x))
            yp = self.dual @ xp
            return codomain.from_dual_representation(codomain.dual_representation.from_linear_form(yp))
        return LinearOperator(domain, codomain, mapping)

    

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

    # Return the abstract dual space. 
    @property 
    def dual(self):
        to_components = self._to_dual_components
        from_components = lambda c : (lambda x : np.dot(self.to_components(x),c))
        return VectorSpace(self.dimension, to_components, from_components)


    # Return the vector as a linear form on the double dual. 
    def to_linear_form(self, x):
        return LinearForm(self.dual, lambda xp : xp(x))



# Class for a Hilbert space with a concrete representation of its dual. 
class HilbertSpace(VectorSpace):

    def __init__(self, space, dual_representation, to_dual_representation, from_dual_representation):
        self._space = space
        self._dual_representation = dual_representation
        self._to_dual_representation = to_dual_representation
        self._from_dual_representation = from_dual_representation    
        super(HilbertSpace,self).__init__(space.dimension, space.to_components, space.from_components)
        
    def to_dual_representation(self,x):
        return self._to_dual_representation(x)

    def from_dual_representation(self,xp):
        return self._from_dual_representation(xp)

    def duality_product(self, xp, x):
        return np.dot(self._dual_representation.to_components(xp), self.to_components(x))

    def inner_product(self, x1, x2):        
        return self.duality_product(self.to_dual_representation(x1),x2)

    def norm(self, x):
        return np.sqrt(self.inner_product(x,x))

    def to_linear_form(self,x):
        return LinearForm(self, lambda y : self.inner_product(x,y))

    def from_linear_form(self, f):
        c = np.zeros(self.dimension)
        cp = np.zeros(self.dimension)
        for i in range(self.dimension):
            c[i] = 1
            x = self.from_components(c)
            cp[i] = f(x)
            c[i] = 0
        xp = self._dual_representation.from_components(cp)
        return self.from_dual_representation(xp)

    @property
    def dual_representation(self):        
        return HilbertSpace(self._dual_representation, self._space, self.from_dual_representation, self.to_dual_representation)


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
        to_dual_representation = lambda x : metric @ space.to_components(x)
        from_dual_representation = lambda xp : space.from_components( cho_solve(factor, xp))
        return HilbertSpace(space, Rn , to_dual_representation, from_dual_representation)

            


