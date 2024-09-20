"""
This module defined the VectorSpace class along with a function 
that returns n-dimensional real vector space with its standard
basis as an instance of this class. 
"""

from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve
from scipy.stats import norm, multivariate_normal
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator as ScipyLinOp
from scipy.sparse.linalg import gmres, bicg, bicgstab, cg



class VectorSpace:
    """
    A class for real vector spaces. To define an instance, the
    user needs to provide the following:

        (1) The dimension of the space, or the dimension of the 
            finite-dimensional approximating space. 
        (2) A mapping from elements of the space to their components. 
            These components must be expressed as numpy arrays with
            shape (dim,1) with dim the spaces dimension. 
        (3) A mapping from components back to the vectors. This
            needs to be the inverse of the mapping in (2), but 
            this requirement is not automatically checked. 

    Note that this class does *not* define elements of the 
    vector space. These must be pre-defined separately. It 
    is also assumed that the usual vector operations are 
    available for this latter space. 
    """

    def __init__(self, dim, to_components, from_components, /, *, base = None):
        """
        Args:
            dim (int): The dimension of the space, or of the 
                finite-dimensional approximating space. 
            to_components (callable):  A functor that maps vectors
                to their components. 
            from_components (callable): A functor that maps components
                to vectors. 
            base (VectorSpace | None): Set to none for an original space, 
                and to the base space when forming the dual. 
        """
        self._dim = dim
        self._to_components = to_components
        self._from_components = from_components
        self._base = base
    
    @property
    def dim(self):
        """The dimension of the space."""
        return self._dim    

    @property
    def dual(self):
        """The dual of the vector space."""
        if self._base is None:
            return VectorSpace(self.dim, self._dual_to_components, self._dual_from_components, base = self)
        else:            
            return self._base

    @property
    def zero(self):
        return self.from_components(np.zeros((self.dim)))

    def to_components(self,x):
        """Maps vectors to components."""        
        return self._to_components(x)

    def from_components(self,c):
        """Maps components to vectors."""
        return self._from_components(c)


    def basis_vector(self,i):
        """Return the ith basis vector."""
        c = np.zeros(self.dim)
        c[i] = 1
        return self.from_components(c)

    def random(self):
        """Returns a random vector with components drwn from a standard Gaussian distribution."""
        return self.from_components(norm().rvs(size = self.dim))

    def identity(self):
        """Returns identity operator on the space."""
        if isinstance(self, HilbertSpace):
            return LinearOperator(self, self, lambda x: x, dual_mapping = lambda yp : yp, 
                                  adjoint_mapping=lambda y : y)            
        else:
            return LinearOperator(self, self, lambda x: x, dual_mapping = lambda yp : yp)            

    def _dual_to_components(self, xp):    
        if xp._matrix is None:             
            matrix = xp.matrix(dense=True)
        else:                        
            matrix = xp._matrix
        return matrix.reshape(xp.domain.dim)
        #return xp.components

    def _dual_from_components(self,cp):        
        return LinearForm(self, components=cp)


class LinearOperator:
    """
    Class for linear operators between two vector spaces. To define an 
    instance, the user must provide the following:

        (1) The domain of the operator as an instance of VectorSpace. 
        (2) The codomain of the operator as an instance of VectorSpace.     

    To define the action of the operator they can provide either:

        (a) A functor that represents the action of the operator.         

    For option (a) the dual and adjoint can be deduced internally 
    but in an inefficient manner, or they can be supplied directly. 

    Linear operators form an algebra over the reals in the usual way. Overloads
    for the relevant operators are provided. In all cases, these operations are
    lazily implemented. 
    """

    def __init__(self, domain, codomain, mapping = None, /, *, 
                 dual_mapping = None, adjoint_mapping = None,                   
                 dual_base = None, adjoint_base = None):
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
        """
        
        self._domain = domain
        self._codomain = codomain        
        self._dual_base = dual_base
        self._adjoint_base = adjoint_base                
        self._mapping = mapping
        self._matrix = None
        if dual_mapping is None:                            
            if self.hilbert_operator:
                if adjoint_mapping is None:
                    self._dual_mapping = self._dual_mapping_default
                    self._adjoint_mapping = self._adjoint_mapping_from_dual
                else:
                    self._adjoint_mapping = adjoint_mapping
                    self._dual_mapping = self._dual_mapping_from_adjoint
            else:                    
                self._dual_mapping = self._dual_mapping_default
                self._adjoint_mapping = None
        else:
            self._dual_mapping = dual_mapping
            if self.hilbert_operator:
                if adjoint_mapping is None:
                    self._adjoint_mapping = self._adjoint_mapping_from_dual
                else:
                    self._adjoint_mapping = adjoint_mapping
            else:
                self._adjoint_mapping = None
            
    @staticmethod
    def self_dual(domain, mapping):
        """Returns a self-dual operator in terms of its domain and mapping."""
        return LinearOperator(domain, domain.dual, mapping, dual_mapping=mapping)

    @staticmethod
    def self_adjoint(domain, mapping):
        """Returns a self-adjoint operator in terms of its domain and mapping."""
        return LinearOperator(domain, domain, mapping, adjoint_mapping=mapping)

    @property
    def domain(self):
        """Domain of the operator."""
        return self._domain

    @property
    def codomain(self):
        """Codomain of the operator."""
        return self._codomain

    @property
    def hilbert_operator(self):
        """True is operator maps between Hilbert spaces."""
        return isinstance(self.domain,HilbertSpace) and isinstance(self.codomain,HilbertSpace)            

    @property
    def dual(self):
        """The dual of the operator."""
        if self._dual_base is None:            
            return LinearOperator(self.codomain.dual, self.domain.dual,
                                  self._dual_mapping, dual_mapping=self,
                                  dual_base = self)                      
        else:
            return self._dual_base

    @property
    def adjoint(self):
        """The adjoint of the operator."""
        if not self.hilbert_operator:
            raise NotImplementedError("Adjoint not defined for the operator.")
        if self._adjoint_base is None:
            return LinearOperator(self.codomain, self.domain, self._adjoint_mapping,
                                  adjoint_mapping=self._mapping, adjoint_base=self)
        else:
            return self._adjoint_base


    def matrix(self, /, *, dense = False, galerkin=False):
        """Return matrix representation of the operator."""
        if dense:
            return self._compute_dense_matrix(galerkin)
        else:
            if galerkin:
                if not self.hilbert_operator:
                    raise NotImplementedError("Defined only for operators between Hilbert spaces.")
                matvec = lambda c : self.codomain.dual.to_components(self.codomain.to_dual(self(self.domain.from_components(c))))
                rmatvec = lambda c : self.domain.dual.to_components(self.domain.to_dual(self._adjoint_mapping(self.codomain.from_components(c))))                                            
            else:
                matvec = lambda c : self.codomain.to_components(self(self.domain.from_components(c)))
                rmatvec = lambda c : self.domain.dual.to_components(self.dual(self.codomain.dual.from_components(c)))
            return ScipyLinOp((self.codomain.dim, self.domain.dim), matvec=matvec, rmatvec=rmatvec)
    

    def _dual_mapping_default(self, yp):
        # Default implementation of the dual mapping. 
        return LinearForm(self.domain, mapping=lambda x : yp(self(x)))       

    def _dual_mapping_from_adjoint(self, yp):
        # Dual mapping in terms of the adjoint. 
        y = self.codomain.from_dual(yp)
        x = self._adjoint_mapping(y)
        return self.domain.to_dual(x)
    
    def _adjoint_mapping_from_dual(self, y):
        # Adjoing mapping in terms of the dual.
        yp = self.codomain.to_dual(y)
        xp = self._dual_mapping(yp)
        return self.domain.from_dual(xp)

    def _compute_dense_matrix(self, galerkin=False):                
        # Compute the matrix representation in dense form. 
        matrix = np.zeros((self.codomain.dim, self.domain.dim))              
        cx = np.zeros(self.domain.dim)                        
        for i in range(self.domain.dim):
            cx[i] = 1            
            matrix[:,i] = (self.matrix(galerkin=galerkin) @ cx)[:]
            cx[i] = 0
        return matrix            

    
    def __call__(self, x):
        """Action of the operator on a vector."""
        if self._mapping is None:
            raise NotImplementedError("Mapping has not been set.")
        return self._mapping(x)

    def __neg__(self):
        """Negative of the operator."""
        domain = self.domain
        codomain = self.codomain
        mapping = lambda x : -self(x)
        dual_mapping = lambda yp : -self.dual(yp)
        if self.hilbert_operator:
            adjoint_mapping = lambda y : -self.adjoint(y)            
        return LinearOperator(domain, codomain, mapping, dual_mapping=dual_mapping,
                              adjoint_mapping=adjoint_mapping)

    def __mul__(self, a):
        """Multiply by a scalar."""
        domain = self.domain
        codomain = self.codomain
        mapping = lambda x : a * self(x)
        dual_mapping = lambda yp : 2 * self.dual(yp)
        if self.hilbert_operator:
            adjoint_mapping = lambda y : a * self.adjoint(y)
        else:
            adjoint_mapping = None            
        return LinearOperator(domain, codomain, mapping, dual_mapping=dual_mapping,
                              adjoint_mapping=adjoint_mapping)        
        
    def __rmul__(self, a):
        """Multiply by a scalar."""
        return self * a

    def __truediv__(self, a):
        """Divide by scalar."""
        return self * (1/a)

    def __add__(self, other):
        """Add another operator."""        
        domain = self.domain
        codomain = self.codomain
        mapping = lambda x :  self(x) + other(x)
        dual_mapping = lambda yp : self.dual(yp) + other.dual(yp)
        if self.hilbert_operator:        
            adjoint_mapping = lambda y : self.adjoint(y) + other.adjoint(y)
        else:
            adjoint_mapping = None                        
        return LinearOperator(domain, codomain, mapping, dual_mapping=dual_mapping,
                              adjoint_mapping=adjoint_mapping)        
        
    def __sub__(self, other):
        """Subtract another operator."""        
        domain = self.domain
        codomain = self.codomain
        mapping = lambda x :  self(x) - other(x)
        dual_mapping = lambda yp : self.dual(yp) - other.dual(yp)
        if self.hilbert_operator:
            adjoint_mapping = lambda y : self.adjoint(y) - other.adjoint(y)
        else:
            adjoint_mapping = None                                    
        return LinearOperator(domain, codomain, mapping, dual_mapping=dual_mapping,
                              adjoint_mapping=adjoint_mapping)                           

    def __matmul__(self, other):
        """Compose with another operator."""        
        domain = other.domain
        codomain = self.codomain
        mapping = lambda x :  self(other(x))
        dual_mapping = lambda yp : other.dual(self.dual(yp))
        if self.hilbert_operator:
            adjoint_mapping = lambda y : other.adjoint(self.adjoint(y))
        else:
            adjoint_mapping = None                        
        return LinearOperator(domain, codomain, mapping, dual_mapping=dual_mapping,
                              adjoint_mapping=adjoint_mapping)                           

        
    def __str__(self):
        """Print the operator as its dense matrix representation."""
        if self._matrix is None:
            return self.matrix(dense=True).__str__()
        else:
            return self._matrix.__str__()


# Global definition of the real numbers as a VectorSpace. 
_REAL = VectorSpace(1, lambda x : np.array([x]), lambda c : c[0])


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

    def __init__(self, domain, /, *, mapping = None, components = None):
        """
        Args:
            domain (VectorSpace): Domain of the linear form. 
            mapping (callable | None): A functor that performs the action
                of the linear form on a vector. 
            matrix (MatrixLike | None): The matrix representation of the 
                form, this having shape (1,dim) with dim the dimension of
                the domain. 
        """

        if components is None:
            super().__init__(domain, _REAL, mapping)
            self._matrix = None
        else:
            super().__init__(domain, _REAL, self._mapping_from_components)
            self._matrix = components.reshape(1,self.domain.dim)        
                    

    def _mapping_from_components(self, x):
        return self.codomain.from_components(np.dot(self._matrix,self.domain.to_components(x)))

    @property
    def components(self):        
        if self._matrix is None:             
            matrix = self.matrix(dense=True)
        else:                        
            matrix = self._matrix
        return matrix.reshape(self.domain.dim)


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
                 /, *, to_dual = None, from_dual = None, base = None):
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
            base (VectorSpace | None): Set to none for an original space, 
                and to the base space when forming the dual.                 
        """
        super().__init__(dim, to_components, from_components)
        self._inner_product = inner_product

        if from_dual is None:
            self.__metric_tensor = self.calculate__metric_tensor()
            self.__metric_tensor_factor = cho_factor(self.metric_tensor)
            self._from_dual = self._from_dual_default
        else:
            self.__metric_tensor = None
            self._from_dual = from_dual            

        if to_dual is None:
            if self.__metric_tensor is None:
                self._to_dual = self._to_dual_default
            else:                                
                self._to_dual = self._to_dual_default_with_metric
        else:
            self._to_dual = to_dual

        self._base = base


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
    
    @property
    def dual(self):
        """The dual of the Hilbert space."""
        if self._base is None:
            return HilbertSpace(self.dim, self._dual_to_components, self._dual_from_components, 
                                self._dual_inner_product, to_dual=self.from_dual, 
                                from_dual=self.to_dual, base = self)
        else:            
            return self._base

    @property
    def metric_tensor(self):
        """The metric tensor for the space."""
        if self.__metric_tensor is None:
            return self.calculate_metric_tensor()
        else:
            return self.__metric_tensor
        
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

    def calculate_metric_tensor(self):
        """Return the space's metric tensor as a numpy matrix."""
        metric_tensor = np.zeros((self.dim, self.dim))
        c1 = np.zeros(self.dim)
        c2 = np.zeros(self.dim)
        for i in range(self.dim):
            c1[i] = 1
            x1 = self.from_components(c1)            
            metric_tensor[i,i] = self.inner_product(x1,x1)
            for j in range(i+1,self.dim):
                c2[j] = 1
                x2 = self.from_components(c2)
                metric_tensor[i,j] = self.inner_product(x1,x2)
                metric_tensor[j,i] = metric_tensor[i,j]
                c2[j] = 0
            c1[i] = 0
        return metric_tensor
    
    def _to_dual_default(self,x):
        return LinearForm(self, mapping = lambda y : self.inner_product(x,y))

    def _to_dual_default_with_metric(self, x):
        cp = self.metric_tensor @ self.to_components(x)
        return self.dual.from_components(cp)

    def _from_dual_default(self, xp):
        cp = self.dual.to_components(xp)
        c = cho_solve(self._metric_tensor_factor, cp)
        return self.from_components(c)

    def _dual_inner_product(self, xp1, xp2):
        return self.inner_product(self.from_dual(xp1), self.from_dual(xp2))            


def euclidean_space(dim, /, *, metric_tensor = None):
    """
    Returns Euclidean space implemented as a HilbertSpace instance. 

    Args:
        dim (int): Dimension of the space. 
        metric_tensor (ArrayLike | None): The metric tensor that defines
            the inner product on the space. If none is provided, this is 
            taken to the identity matrix, and hence the standard Euclidean
            inner product is used. 

    Notes:
        To work with a non-standard inner product, the Cholesky factor of the 
        metric tensor is formed. This implementation uses dense matrices, and 
        will not be efficient in high-dimensional spaces. An alternative 
        implementation that uses pre-conditioned iterative solvers would be 
        preferable in such cases. 
    """

    to_components = lambda x : x
    from_components = lambda c : c
    space = VectorSpace(dim, to_components, from_components)

    if metric_tensor is None:        
        inner_product = lambda x1, x2 : np.dot(x1, x2)
        to_dual = lambda x:  space.dual.from_components(space.to_components(x))
        from_dual = lambda xp : space.from_components(space.dual.to_components(xp))        
    else:
        factor = cho_factor(metric_tensor)
        inner_product = lambda x1, x2 : np.dot(metric_tensor @ x1, x2)
        to_dual = lambda x : space.dual.from_components(metric_tensor @space.to_components(x))
        from_dual = lambda xp : space.from_components(cho_solve(factor, space.dual.to_components(xp)))

    return HilbertSpace.from_vector_space(space, inner_product,to_dual=to_dual, from_dual=from_dual)

            
class GaussianMeasure:
    """
    Class for Gaussian measures on a Hilbert space.  
    """

    def __init__(self, domain, covariance, / , *, expectation = None,
                 sample = None, sample_using_matrix = False):  
        """
        Args:
            domain (HilbertSpace): The Hilbert space on which the measure is defined. 
            covariance (callable): A functor representing the covariance operator. 
            expectation (Vector): The expectation value of the measure. If none is provided, set equal to zero.  
            sample (callable): A functor that returns a random sample from the measure.         
        """
        self._domain = domain              
        self._covariance = covariance
        if expectation is None:
            self._expectation = self.domain.zero
        else:
            self._expectation = expectation
        if sample is None:
            self._sample_defined = False
        else:
            self._sample = sample
            self._sample_defined = True
        if sample is None and sample_using_matrix:
            dist = multivariate_normal(mean = self.expectation, cov= self.covariance.matrix)
            self._sample = lambda : self.domain.from_components(dist.rvs())
            self._sample_defined = True


    @staticmethod
    def from_factored_covariance(factor, /, *,  expectation = None):
            
        covariance  = factor @ factor.adjoint
        sample = lambda : factor(norm().rvs(size = factor.domain.dim))
        if expectation is not None:
            sample = lambda : expectation + sample()        
        return GaussianMeasure(factor.codomain, covariance, expectation = expectation, sample = sample)    

    @property
    def domain(self):
        """The Hilbert space the measure is defined on."""
        return self._domain
    
    @property
    def covariance(self):
        """The covariance operator as an instance of LinearOperator."""
        return LinearOperator.self_adjoint(self.domain, self._covariance)
    
    @property
    def expectation(self):
        """The expectation of the measure."""
        return self._expectation

    @property
    def sample_defined(self):
        """True if the sample method has been implemented."""
        return self._sample_defined
    
    def sample(self):
        """Returns a random sample drawn from the measure."""
        if self.sample_defined:        
            return self._sample()
        else:
            raise NotImplementedError

    def affine_mapping(self, /, *,  operator = None, translation = None):
        """
        Returns the push forward of the measure under an affine mapping.

        Args:
            operator (LinearOperator): The operator part of the mapping.
            translation (vector): The translational part of the mapping.

        Returns:
            Gaussian Measure: The transformed measure defined on the 
                codomain of the operator.

        Raises:
            ValueError: If the domain of the operator domain is not 
                the domain of the measure.

        Notes:
            If operator is not set, it defaults to the identity.
            It translation is not set, it defaults to zero. 
        """
        assert operator.domain == self.domain        
        if operator is None:
            covariance = self.covariance
        else:
            covariance = operator @ self.covariance @ operator.adjoint        
        expectation = operator(self.expectation)
        if translation is not None:
            expectation = expectation + translation
        if self.sample_defined:
            if translation is None:
                sample = lambda : operator(self.sample())
            else: 
                sample = lambda  : operator(self.sample()) + translation                                        
        else:
            sample = None
        return GaussianMeasure(operator.codomain, covariance, expectation = expectation, sample = sample)            

    def __neg__(self):
        """Negative of the measure."""
        if self.sample_defined:
            sample = lambda : -self.sample()
        else:
            sample = None    
        return GaussianMeasure(self.domain, self.covariance, expectation=-self.expectation, sample=sample)
    
        
    def __mul__(self, alpha):
        """Multiply the measure by a scalar."""
        covariance = LinearOperator.self_adjoint(self.domain,lambda x : alpha*2 * self.covariance(x))
        expectation = alpha * self.expectation
        if self.sample_defined:
            sample = lambda : alpha * self.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, expectation = expectation, sample = sample)

    def __rmul__(self, alpha):
        """Multiply the measure by a scalar."""
        return self * alpha
    
    def __add__(self, other):
        """Add two measures on the same domain."""
        assert self.domain == other.domain
        covariance = self.covariance + other.covariance
        expectation = self.expectation + other.expectation
        if self.sample_defined and other.sample_defined:
            sample  = lambda : self.sample() + other.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, expectation = expectation, sample = sample) 

    def __sub__(self, other):
        """Subtract two measures on the same domain."""
        assert self.domain == other.domain
        covariance = self.covariance + other.covariance
        expectation = self.expectation - other.expectation
        if self.sample_defined and other.sample_defined:
            sample  = lambda : self.sample() - other.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, expectation = expectation, sample = sample)     


class LinearSolver(ABC):


    def __init__(self):
        self._operator = None
        self._initial_guess = None

    @abstractmethod
    def _set_up(self):
        pass

    @property    
    def operator(self):
        return self._operator

    @operator.setter
    def operator(self, operator):
        self._operator = operator
        self._set_up()

    @property
    def domain(self):
        return self.operator.domain

    @property
    def codomain(self):
        return self.operator.codomain

    @property
    def initial_guess(self):
        return self._initial_guess
    
    @initial_guess.setter
    def initial_guess(self, initial_guess):
        self._initial_guess = initial_guess

    
    @property
    @abstractmethod    
    def inverse_operator(self):
        """Returns a LinearOperator that is the inverse of the given operator."""
        pass

class MatrixSolver(LinearSolver):

    def __init__(self, galerkin):
        super().__init__()
        self._galerkin = galerkin

    @property
    def galerkin(self):
        return self._galerkin

    @abstractmethod
    def _solver(self, c, trans):
        pass

    def _mapping(self, y):
        if self.galerkin:
            yp = self.codomain.to_dual(y)
            cyp = self.codomain.dual.to_components(yp)
            cx = self._solver(cyp,0)
            return self.domain.from_components(cx)
        else:
            cy = self.codomain.to_components(y)
            cx = self._solver(cy,0)
            return self.domain.from_components(cx)

    def _dual_mapping(self, xp):
        if self.galerkin:            
            cxp = self.domain.dual.to_components(xp)
            cy = self._solver(cxp,1)
            y = self.codomain.from_components(cy)
            return self.codomain.to_dual(y)                        
        else:
            
            cxp = self.domain.dual.to_components(xp)
            cyp = self._solver(cxp,1)
            return self.codomain.dual.from_components(cyp)

    def _adjoint_mapping(self,x):
        if self.galerkin:
            xp = self.domain.to_dual(x)
            cxp = self.domain.dual.to_components(xp)
            cy = self._solver(cxp,1)
            return self.codomain.from_components(cy)
        else:
            xp = self.domain.to_dual(x)
            cxp = self.domain.dual.to_components(xp)
            cyp = self._solver(cxp,1)
            yp = self.codomain.dual.from_components(cyp)
            return self.codomain.from_dual(yp)

    @property
    def inverse_operator(self):
        return LinearOperator(self.codomain, self.domain, self._mapping, 
                              dual_mapping=self._dual_mapping, 
                              adjoint_mapping=self._adjoint_mapping)
            


class MatrixSolverLU(MatrixSolver):

    def __init__(self, /, *, galerkin=False):     
        super().__init__(galerkin)   
        
    def _set_up(self):
        self._factor = lu_factor(self.operator.matrix(dense=True, galerkin=self.galerkin))

    def _solver(self, c, trans):
        return lu_solve(self._factor, c, trans=trans)


class MatrixSolverCholesky(MatrixSolver):

    def __init__(self, /, *, galerkin=False):     
        super().__init__(galerkin)   
        
    def _set_up(self):
        self._factor = cho_factor(self.operator.matrix(dense=True, galerkin=self.galerkin))

    def _solver(self, c, trans):
        return cho_solve(self._factor, c)


    
class MatrixSolverCG(MatrixSolver):
    
    def __init__(self, /, *, preconditioner=None, galerkin=False,
                 rtol=1e-05, atol=0.0, maxiter=None, callback=None):  
        super().__init__(galerkin)                          
        self._preconditioner = preconditioner
        self._galerkin = galerkin        
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._callback = callback
        
    def _set_up(self):
        self._matrix = self.operator.matrix(galerkin=self.galerkin)

    def _solver(self, c, trans):
        if self.initial_guess is None:
            x0 = None
        else:
            x0 = self.domain.to_components(self.initial_guess)        
        return cg(self._matrix, c, x0 = x0, rtol = self._rtol, atol = self._atol, 
                 maxiter=self._maxiter, callback=self._callback)[0]



class MatrixSolverBICSTAB(MatrixSolver):
    
    def __init__(self, /, *, preconditioner=None, galerkin=False,
                 rtol=1e-05, atol=0.0, maxiter=None, callback=None):  
        super().__init__(galerkin)                          
        self._preconditioner = preconditioner
        self._galerkin = galerkin        
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._callback = callback
        
    def _set_up(self):
        self._matrix = self.operator.matrix(galerkin=self.galerkin)

    def _solver(self, c, trans):
        if self.initial_guess is None:
            x0 = None
        else:
            x0 = self.domain.to_components(self.initial_guess)        
        return bicgstab(self._matrix, c, x0 = x0, rtol = self._rtol, atol = self._atol, 
                 maxiter=self._maxiter, callback=self._callback)[0]

    



class MatrixSolverGMRES(MatrixSolver):
    
    def __init__(self, /, *, preconditioner=None, galerkin=False,
                 rtol=1e-05, atol=0.0, maxiter=None, callback=None):  
        super().__init__(galerkin)                          
        self._preconditioner = preconditioner
        self._galerkin = galerkin        
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._callback = callback
        
    def _set_up(self):
        self._matrix = self.operator.matrix(galerkin=self.galerkin)

    def _solver(self, c, trans):
        if self.initial_guess is None:
            x0 = None
        else:
            x0 = self.domain.to_components(self.initial_guess)        
        return gmres(self._matrix, c, x0 = x0, rtol = self._rtol, atol = self._atol, 
                 maxiter=self._maxiter, callback=self._callback)[0]

    

