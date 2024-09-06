"""
This module defines the LinearOperator class along with the 
class DualVector as a special case. 

The definition of the later class requires an implementation
of the real numbers as a VectorSpace instance. This is stored
as a module variable, _REAL. There is no need for this variable
 to be imported or used by others!
"""


import numpy as np
from pygeoinf.vector_space import VectorSpace


class LinearOperator:
    """
    Class for linear operators between two vector spaces. To define an 
    instance, the user must provide the following:

        (1) The domain of the operator as an instance of VectorSpace. 
        (2) The codomain of the operator as an instance of VectorSpace.     

    To define the action of the operator they can provide either:

        (a) A functor that represents the action of the operator. 
        (b) The matrix representation for the operator relative to the 
            basis for the domain and codomain. This matrix can be a dense
            numpy matrix, a scipy sparse matrix, or another object that 
            behaves in the same way. 

    In addition, the user can supply functors that implement the action of
    the operators dual and its adjoint, with the latter defined only in the 
    case of operators between Hilbert Spaces. 

    Linear operators form an algebra over the reals in the usual way. Overloads
    for the relevant operators are provided. In all cases, these operations are
    lazily implemented. 
    """

    def __init__(self, domain, codomain, /, *, mapping = None,
                 dual_mapping = None, adjoint_mapping = None, matrix = None):
        """
        Args:
            domain (VectorSpace): The domain of the operator. 
            codomain (VectorSpace): The codomain of the operator. 
            mapping (callable | None): A functor that implements the 
                action of the operator. 
            dual_mapping (callable | None): A functor that implements 
                the action of the dual operator. 
            adjoint_mapping (callable | None): A functor that implements
                the action of the adjoint operator. 
            matrix (MatrixLike | None): The matrix representation of the 
                operator relative to the bases for its domain and codomain.        
        """
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
        """Domain of the operator."""
        return self._domain

    @property
    def codomain(self):
        """Codomain of the operator."""
        return self._codomain

    @property
    def matrix(self):
        """Matrix representation of the operator."""
        if self._matrix is None:            
            return self._compute_matrix()            
        else:            
            return self._matrix

    @property
    def shape(self):
        """Shape of the operator's matrix representation."""
        return (self.codomain.dim, self.domain.dim)

    def store_matrix(self):
        """Call to compute and store the operators matrix representation."""
        if self._matrix is None:
            self._matrix = self._compute_matrix()

    def _mapping_from_matrix(self,x):
        # Sets the mapping from the assigned matrix.        
        cx = self.domain.to_components(x)
        cy = self.matrix @ cx
        return self.codomain.from_components(cy)
        
    def _compute_matrix(self):        
        # Compute the matrix representation through.
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
        """Action of the operator on a vector."""
        return self._mapping(x)

    def __mul__(self, a):
        """Multiply operator by a scalar."""
        return LinearOperator(self.domain, self.codomain, mapping = lambda x : a * self(x))
        
    def __rmul__(self, a):
        """Multiply operator by a scalar."""
        return self * a

    def __div__(self, a):
        """Divide operator by a scalar."""
        return self * (1/a)

    def __add__(self, other):
        """Sum of two operators."""        
        if self.domain != other.domain:
            raise ValueError("Domains must be equal")
        if self.codomain != other.codomain:
            raise ValueError("Codomains must be equal")
        return LinearOperator(self.domain, self.codomain, mapping = lambda x : self(x) + other(x))

    def __sub__(self, other):     
        """Difference of two operators."""  
        if self.domain != other.domain:
            raise ValueError("Domains must be equal")
        if self.codomain != other.codomain:
            raise ValueError("Codomains must be equal")
        return LinearOperator(self.domain, self.codomain, mapping = lambda x : self(x) - other(x))  

    def __matmul__(self, other):
        """Composition of two operators."""
        if self.domain != other.codomain:
            raise ValueError("Operators cannot be composed")        
        return LinearOperator(other.domain, self.codomain, mapping = lambda x : self(other(x)))  

    def __str__(self):
        """Print the operator as its matrix representation."""
        return self.matrix.__str__()


# Implementation of the real numbers as a VectorSpace. This should not be 
# imported, used, or changed!
_REAL = VectorSpace(1, lambda x : np.array([[x,]]), lambda c : c[0,0])


class LinearForm(LinearOperator):
    """
    Class for linear forms on a vector space. 

    The vector space is represented by a VectorSpace object. Linear forms
    are (continuous) linear mappings from the to the real numbers. These
    mappings form a vector space known that is *dual* to the original.

    A linear form can be specified either:

        (1) In terms of a functor that performs its action on a vector. 
        (2) Its matrix-representation relative to the basis for the space.

    Specification of a form in terms of its matrix representation offers
    computational advatages in many cases. A form specified in terms 
    of a mapping can (as with a general linear operator) compute and 
    store its matrix representation.
    """

    def __init__(self, domain, /, *, mapping = None, matrix = None):
        """
        Args:
            domain (VectorSpace): Domain of the linear form. 
            mapping (callable | None): A functor that performs the action
                of the linear form on a vector. 
            matrix (MatrixLike | None): The matrix representation of the 
                form, this having shape (1,dim) with dim the dimension of
                the domain. 
        """
        super().__init__(domain, _REAL, mapping = mapping, matrix=matrix )
