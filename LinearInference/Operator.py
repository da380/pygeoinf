import numpy as np
from scipy.sparse.linalg import LinearOperator as SciPyOp
from LinearInference.Euclidean import EuclideanSpace


'''
Wrapper class for a linear operator, A, between two Hilbert spaces X and Y. The domain 
and co-domain of the operator must be instances of the class HilbertSpace or of a class 
that is derived from HilbertSpace. 

To form an instance of this class, the user must provide:

domain: The domain, X, of the linear operator. 

coDomain: The co-domain, Y, of the linear operator. 

mapping: A function that implements the action of the linear operator. It maps X -> Y.

dualMapping: A function that implements the action of the dual linear operator. It maps Y' -> X'.
        
'''

class Linear:

    def __init__(self,domain, coDomain, mapping, dualMapping):

        self._domain = domain
        self._coDomain = coDomain
        self._mapping = mapping
        self._dualMapping = dualMapping    

    # Return the domain of the operator.
    @property
    def Domain(self):
        return self._domain

    # Return the co-domain of the operator. 
    @property
    def CoDomain(self):
        return self._coDomain

    # Action of the operator on a vector. 
    def __call__(self,v):
        return self._mapping(v)


    # Return the dual operator.
    @property
    def Dual(self):
        return Linear(self.CoDomain.Dual, 
                              self.Domain.Dual,
                              self._dualMapping, 
                              self._mapping)                                

    # Return the adjoint operator.
    @property
    def Adjoint(self):
        mapping = lambda v : self.Domain.InverseRiesz(self._dualMapping(self.CoDomain.Riesz(v)))
        dualMapping = lambda v : self.CoDomain.InverseRiesz(self._mapping(self.Domain.Riesz(v)))
        return Linear(self.CoDomain, self.Domain,mapping, dualMapping)

    # Return the matrix representation of the operator as an instance of the Linear class. 
    @property
    def AsMatrix(self):
        domain = EuclideanSpace(self.Domain.Dimension)
        coDomain = EuclideanSpace(self.CoDomain.Dimension)
        mapping = lambda x : self.CoDomain.ToComponents(self( self.Domain.FromComponents(x)))
        dualMapping = lambda x : self.Domain.Dual.ToComponents(self.Dual(self.CoDomain.Dual.FromComponents(x)))
        return Linear(domain, coDomain, mapping, dualMapping)

    # Return the representation of the operator with respect to the bases for X and Y. The matrix is not formed, 
    # but returned using the SciPy LinearOperator class which is suitable for matrix-free linear algebra.
    @property
    def AsSciPyLinearOperator(self):
        shape = (self.CoDomain.Dimension, self.Domain.Dimension)    
        matvec = lambda x : self.AsMatrix(x)
        rmatvec = lambda x : self.AsMatrix.Dual(x)
        return SciPyOp(shape, matvec = matvec, rmatvec = rmatvec)

    # Return the representation of the operator with  to the bases for X and Y as a numpy array. 
    @property
    def AsDense(self):        
        m = self.Domain.Dimension
        n = self.CoDomain.Dimension
        A = np.zeros((n,m))        
        x = np.zeros(m)
        for i in range(m):
            x[i] = 1
            y = self.AsMatrix(x)
            A[:,i] = y
            x[i] = 0
        return A
    
    # Define scalar multiplication.
    def __mul__(self,scalar):
        return Operator(self.Domain, self.CoDomain, lambda x : scalar * self(x), lambda x : scalar * self.Dual(x))

    def __rmul__(self,scalar):
        return self * scalar

    # Define scalar division.
    def __div(self,scalar):
        return self * (1/scalar)

    # Define addition. 
    def __add__(self,other):
        assert self.Domain == other.Domain
        assert self.CoDomain == other.CoDomain
        return Linear(self.Domain, self.CoDomain, lambda x : self(x) + other(x), lambda x : self.Dual(x) + other.Dual(x))

    # Define subtraction. 
    def __sub__(self,other):
        assert self.Domain == other.Domain
        assert self.CoDomain == other.CoDomain
        return Linear(self.Domain, self.CoDomain, lambda x : self(x) - other(x), lambda x : self.Dual(x) - other.Dual(x))
    
    # Define composition. 
    def __matmul__(self,other):
        assert self.Domain == other.CoDomain
        return Linear(other.Domain, self.CoDomain, lambda x : self(other(x)), lambda x: other.Dual(self.Dual(x)))


    # Return dense matrix to be printed
    def __str__(self):
        return self.AsDense.__str__()


    # Return direct sum with a second operator. 
    def DirectSum(self, other):
        domain = self.Domain.DirectSum(other.Domain)
        coDomain = self.CoDomain.DirectSum(other.CoDomain)
        mapping = lambda x : (self(x[0]),other(x[1]))
        dualMapping = lambda x : (self.Dual(x[0]), other.Dual(x[1]))
        return Linear(domain, coDomain, mapping, dualMapping)



# Class for self-dual lear operators.
class SelfDual(Linear):
    def __init__(self, Space, mapping):        
        super(SelfDual,self).__init__(Space,Space.Dual,mapping,mapping)


# Class for self-adjoint linea operators. 
class SelfAdjoint(Linear):

    def __init__(self, Space, mapping):
        dualMapping = lambda x : Space.Riesz(mapping(Space.InverseRiesz(x)))
        super(SelfAdjoint,self).__init__(Space,Space,mapping,dualMapping)
                                                            
             
