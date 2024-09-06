"""
This module defined the HilbertSpace and DualHilbertSpace class. 
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from pygeoinf.vector_space import VectorSpace, standard_vector_space
from pygeoinf.linear_operator import LinearForm, LinearOperator
from pygeoinf.dual import DualVectorSpace, DualOperator

class HilbertSpace(VectorSpace):
    """
    Class for Hilbert spaces. To define an instance, the user provides:

        (1) The dimension of the space, or the dimension of the 
            finite-dimensional approximating space. 
        (2) A mapping from elements of the space to their components. 
            These components must be expressed as numpy arrays with
            shape (dim,1) with dim the spaces dimension. 
        (3) A mapping from components back to the vectors. This
            needs to be the inverse of the mapping in (2), but 
            this requirement is not automatically checked. 
        (4) The inner product on the space. 

    Note that this space inherits from VectorSpace. 

    The user can also provide either of the following:
        
        (a) The mapping from the space to its dual. 
        (b) The mapping from a dual vector to its representation 
            within the space. 

    The existence of such mappings follows from the Riesz representation
    theorem. If (a) is not provided, then then cannonical form of this
    mapping is used, with a vector, x, being mapped to the dual vector 
    that acts on a vector, y, by y -> (x,y).

    If (b) is not provided, the metric tensor for the space relative 
    to its basis is computed along with its Cholesky factor. In this 
    case the mapping in (a) is replaced by one using dual components
    that is more efficient. 

    Within high-dimensional spaces, custom mappings to and from the 
    dual space should be provided for the user for better efficiency.  

    A HilbertSpace can be constructed directly, or from an already 
    formed VectorSpace using the static method "from_vector_space". 
    The latter option is often preferable as a custom definition of 
    the functor "to_dual" requires that the dual space be available.
    """

    def __init__(self, dim, to_components, from_components, inner_product,
                 /, *, to_dual = None, from_dual = None):
        """
        Args:
            dim (int): The dimension of the space, or of the 
                finite-dimensional approximating space. 
            to_components (callable):  A functor that maps vectors
                to their components. 
            from_components (callable): A functor that maps components
                to vectors.         
            inner_product (callable): A functor the implements the inner 
                product on the space. 
            to_dual (callable | None): A funcator that maps a vector 
                to the cannonically associated dual vector. 
            from_dual (callable | None): A functor that maps a dual vector
                to its representation on the space. 
        """
        super().__init__(dim, to_components, from_components)
        self._inner_product = inner_product
        if to_dual is None:
            self._to_dual = lambda x : LinearForm(self, mapping = lambda y : self.inner_product(x,y))
        else:
            self._to_dual = to_dual

        if from_dual is None:
            raise NotImplementedError("To be done!")
        else:
            self._from_dual = from_dual

    @staticmethod
    def from_vector_space(space, inner_product, /, *, to_dual = None, from_dual = None):
        """
        Form a HilbertSpace from a VectorSpace by providing additional structures. 

        Args:
            space (VectorSpace): The vector space to form as base for the Hilbert space. 
            inner_product (callable): A functor the implements the inner 
                product on the space. 
            to_dual (callable | None): A funcator that maps a vector 
                to the cannonically associated dual vector. 
            from_dual (callable | None): A functor that maps a dual vector
                to its representation on the space.             
        """
        return HilbertSpace(space.dim, space.to_components, space.from_components, 
                            inner_product, to_dual=to_dual, from_dual=from_dual)

    @property
    def vector_space(self):
        """The underlying vector space."""
        return VectorSpace(self.dim, self.to_components, self.from_components)
        
    def inner_product(self, x1, x2):
        """Return the inner product of two vectors."""
        return self._inner_product(x1,x2)

    def norm(self,x):
        """Return the norm of a vector."""
        return np.sqrt(self.inner_product(x,x))
        
    def to_dual(self, x):
        """Map a vector to cannonically associated dual vector."""
        return self._to_dual(x)

    def from_dual(self, xp):
        """Map a dual vector to its representation in the space."""
        return self._from_dual(xp)


def standard_euclidean_space(dim): 
    """ Returns Euclidean space its stanard basis and inner product."""
    space = standard_vector_space(dim)
    dual_space = DualVectorSpace(space)
    inner_product = lambda x1, x2 : np.dot(x1, x2)
    to_dual = lambda x :  dual_space.from_components(space.to_components(x))
    from_dual = lambda xp : space.from_components(dual_space.to_components(xp))
    return HilbertSpace.from_vector_space(space, inner_product, to_dual=to_dual,
                                          from_dual=from_dual)


def _is_hilbert_space_operator(operator):
    # True is an operator maps between Hilbert spaces. 
    return isinstance(operator.domain, HilbertSpace) and isinstance(operator.codomain, HilbertSpace)

class DualHilbertSpace(HilbertSpace):
    """
    Class for dual Hilbert spaces. To form an instance, a HilbertSpace object 
    is provided. Elements of the dual space are linear forms on the 
    original space. The basis for the dual space is induced from that 
    on the original space in a cannonical manner. Similarly, the 
    inner product and mappings to and from the its dual (i.e., the original space)    
    follow naturually. 

    Note that if the dual of a dual space it requested, the double dual is 
    not formed. Rather, a copy of the original space is constructed. This 
    is consistent with the reflexivity of all finite-dimensional spaces 
    and of infinite-dimensional Hilbert spaces that might be approximated. 
    """
    def __init__(self, space):
        """
        Args:
            space (HilbertSpace): The Hilbert space whose dual is to be formed. 

        Notes:
            If the dual of a dual space is requested, the double dual is not formed, 
            and instead a copy of the original is generated.
        """
        if isinstance(space, DualHilbertSpace):
            original = space._original
            super().__init__(original.dim, original.to_components, original.from_components, 
                             original.inner_product, to_dual=original.to_dual, from_dual=original.from_dual)
        else:
            self._original = space
            dual_space = DualVectorSpace(space)            
            super().__init__(space.dim, dual_space.to_components, dual_space.from_components,
                             self._dual_inner_product, to_dual=space.from_dual, from_dual=space.to_dual)    

    def _dual_inner_product(self, xp1, xp2):
        return self._original.inner_product(self._original.from_dual(xp1), self._original.from_dual(xp2))            


class AdjointOperator(LinearOperator):
    """
    Class for the adjoint of a linear operator. To form an instance, a 
    LinearOperator object is provided. If the original operator is
    denoted by A, then its adjoint, A*, is defined through 

    ( y , A x ) = ( A y, x )

    for all x in the domain of the operator and y in the operator's codomain, 
    where ( , ) are appropriate inner products. 

    The adjoint mapping can be provided or determined in one of two ways:

        (1) A functor that implements the adjoint mapping has been provided. 
        (2) The operators dual mapping is used to compute the adjoint. 

    If (2) is applied, the dual mapping can itself be generated in  
    a range of ways as documented in DualOperator. 

    Efficient implementation depend on either the adjoint or dual mappings
    being provided by the user. 

    If the adjoint of a adjoint operator is requested, the double dual is not formed, 
    and instead a copy of the original operator is generated. This is consistent
    with the treatment of dual spaces and dual operators. 
    """
    def __init__(self, operator):
        """
        Args:
            operator (LinearOperator): The operator whose adjoint is to be formed.         

        Raises:
            ValueError: If the operator does not map between Hilbert spaces. 
        """
        if not _is_hilbert_space_operator(operator):
            raise ValueError("Adjoint defined only for operators between Hilbert Spaces.")

        if isinstance(operator, AdjointOperator):
            original = operator._original
            super().__init__(original.domain, original.codomain, mapping = original.mapping)
        else:
            self._original = operator            
            if operator._adjoint_mapping is None:                
                mapping = self._default_adjoint_mapping
            else:
                mapping = operator._adjoint_mapping
            super().__init__(operator.codomain, operator.domain, mapping=mapping)


    def _default_adjoint_mapping(self, y):
        operator = self._original
        dual_operator = DualOperator(operator)                
        return operator.domain.from_dual(dual_operator(operator.codomain.to_dual(y)))

