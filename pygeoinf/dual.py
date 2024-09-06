"""
This module defines the DualVectorSpace class and the DualOperator class.
"""

from pygeoinf.vector_space import VectorSpace
from pygeoinf.linear_operator import LinearOperator, LinearForm


class DualVectorSpace(VectorSpace):
    """
    Class for dual vector spaces. To form an instance, a VectorSpace object 
    is provided. Elements of the dual vector space are linear forms on the 
    original space. The basis for the dual space is induced from that 
    on the original space in a cannonical manner.     

    Note that if the dual of a dual space it requested, the double dual is 
    not formed. Rather, a copy of the original space is constructed. This 
    is consistent with the reflexivity of all finite-dimensional spaces 
    and of infinite-dimensional Hilbert spaces that might be approximated. 
    """
    def __init__(self, space):        
        """
        Args:
            space (VectorSpace): The vector space whose dual is to be formed. 

        Notes:
            If the dual of a dual space is requested, the double dual is not formed, 
            and instead a copy of the original is generated. 
        """
        if isinstance(space, DualVectorSpace):                        
            original = space._original
            super().__init__(original.dim, original.to_components, original.from_components)
        else:            
            self._original = space
            super().__init__(space.dim, self._dual_to_components, self._dual_from_components)                

    def _dual_to_components(self, xp):
        # Mapping to components for a dual space. Note that components are 
        # always column vectors, but a linear forms matrix representation is
        # a row vector. 
        return xp.matrix.reshape(self.dim,1)

    def _dual_from_components(self,cp):
        # Mapping from components for a dual space. Note that components are 
        # always column vectors, but a linear forms matrix representation is
        # a row vector. 
        return LinearForm(self._original, matrix = cp.reshape(1,self.dim))



class DualOperator(LinearOperator):
    """
    Class for the dual of a linear operator. To form an instance, a 
    LinearOperator object is provided. If the original operator is
    denoted by A, then its dual, A', is defined through 

    < yp , A x > = < A' yp, x >

    for all x in the domain of the operator and yp in the dual of the 
    operator's codomain, and where < , > is the dual pairing. 

    The dual mapping can be provided or determined in one of three ways:

        (1) A functor that implements the mapping is provided by the user. 
        (2) The matrix representation of the operator is available and 
            whose transpose is used to define the mapping. 
        (3) The dual operator's mapping is defined directly via the above 
            definition. 

    While the third approach is simplest from a user perspective, it 
    is inefficient in many cases. For example, let yp = A' xp for 
    xp a linear form on the domain of A. Determine the action of 
    yp on a vector in the codomain of the operator requires a single
    evaluation of A. And hence, determining the components of the 
    dual vector would require the action of A n-times, with n the 
    dimension of the operators codomain. 



    If the dual of a dual operator is requested, the double dual is not formed, 
    and instead a copy of the original operator is generated. This is consistent
    with the treatment of dual spaces. 
    """

    def __init__(self, operator):
        """
        Args:
            operator (LinearOperator): The operator whose dual is to be formed.         
        """
        if isinstance(operator, DualOperator):
            original = operator._original
            super().__init__(original.domain, original.codomain, mapping = original)
        else:            
            self._orginal = operator
            domain = DualVectorSpace(operator.codomain)
            codomain = DualVectorSpace(operator.domain)            
            if operator._matrix is None:
                if operator._dual_mapping is None:                                    
                    mapping = self._default_dual_mapping
                else:                    
                    mapping = operator._dual_mapping
                super().__init__(domain, codomain, mapping=mapping)
            else:                
                super().__init__(domain, codomain, matrix = operator._matrix.T)    
                

    def _default_dual_mapping(self,yp):
        # Default implementation of the dual operator via its definition. 
        return LinearForm(self._orginal.domain, mapping = lambda x: yp(self._orginal(x)))                                    


