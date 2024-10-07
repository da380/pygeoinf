import numpy as np
import pyshtools as sh
from pygeoinf.linalg import VectorSpace, HilbertSpace, \
                            LinearOperator, GaussianMeasure, \
                            EuclideanSpace, LinearForm
from scipy.sparse import diags



class SHToolsHelper:
    """Helper class for working with pyshtool grid functions and coefficients."""


    def __init__(self, lmax, /, *, radius=1, grid = "DH"):
        """
        Args:
            lmax (int): Maximum degree for spherical harmonic expansions. 
            radius (float): radius of the sphere. 
            grid (str): Grid type for spatial functions.               
        """
        self._lmax = lmax
        self._radius = radius
        self._grid = grid
        if self.grid == "DH2":
            self._sampling = 2
        else:
            self._sampling = 1
        self._extend = True
        self._normalization = "ortho"
        self._csphase = 1
        

    @property
    def lmax(self):
        """Returns truncation degree."""
        return self._lmax

    @property
    def dim(self):
        """Returns dimension of vector space."""
        return (self.lmax+1)**2

    @property
    def radius(self):
        """Returns radius of the sphere."""
        return self._radius

    @property
    def grid(self):
        """Returns spatial grid option."""
        return self._grid

    @property
    def extend(self):
        """True is spatial grid contains longitudes 0 and 360."""
        return self._extend

    @property
    def normalization(self):
        """Spherical harmonic normalisation option."""
        return self._normalization

    @property
    def csphase(self):
        """Condon Shortley phase option."""
        return self._csphase    

    
    def spherical_harmonic_index(self, l, m):
        """Return the component index for given spherical harmonic degree and order."""
        if m >= 0:
            return int(l*(l+1)/2) + m
        else:
            offset = int((self.lmax + 1)*(self.lmax + 2) / 2)
            return offset + int((l - 1) * l / 2) - m - 1

    def _to_components_from_coeffs(self, coeffs):
        """Return component vector from coefficient array."""
        c = np.empty(self.dim)        
        i = 0
        for l in range(self.lmax+1):
            j = i + l + 1
            c[i:j] = coeffs[0,l,:l+1] 
            i = j
        for l in range(1,self.lmax+1):
            j = i + l
            c[i:j] = coeffs[1,l,1:l+1] 
            i = j    
        return c
    
    def _to_components_from_SHCoeffs(self, ulm):        
        """Return component vector from SHCoeffs object."""
        return self._to_components_from_coeffs(ulm.coeffs)

    
    def _to_components_from_SHGrid(self, u):      
        """Return component vector from SHGrid object."""
        ulm = u.expand(normalization=self.normalization, csphase=self.csphase)  
        return self._to_components_from_SHCoeffs(ulm)


    def _from_components_to_SHCoeffs(self,c):
        """Return SHCoeffs object from its component vector."""
        coeffs = np.zeros((2,self.lmax+1, self.lmax+1))
        i = 0
        for l in range(self.lmax+1):
            j = i + l + 1
            coeffs[0,l,:l+1] = c[i:j] 
            i = j
        for l in range(1,self.lmax+1): 
            j = i + l
            coeffs[1,l,1:l+1] = c[i:j]
            i = j    
        ulm = sh.SHCoeffs.from_array(coeffs, normalization = self.normalization, csphase = self.csphase)
        return ulm
    
    def _from_components_to_SHGrid(self,c):        
        """Return SHGrid object from its component vector."""
        ulm = self._from_components_to_SHCoeffs(c)
        return ulm.expand(grid=self.grid, extend=self.extend)

    def _degree_dependent_scaling_to_diagonal_matrix(self, f):        
        values = np.zeros(self.dim)
        i = 0
        for l in range(self.lmax+1):
            j = i + l + 1
            values[i:j] = f(l)
            i = j
        for l in range(1,self.lmax+1):
            j = i + l
            values[i:j] = f(l)
            i = j        
        return diags([values], [0])


class SphericalHarmonicExpansion(SHToolsHelper,VectorSpace):

    def __init__(self, lmax, /, *,  vector_as_SHGrid=True, radius=1,   grid = "DH"):
        SHToolsHelper.__init__(self, lmax, radius=radius, grid=grid)        
        if vector_as_SHGrid:
            VectorSpace.__init__(self, self.dim, self._to_components_from_SHGrid, self._from_components_to_SHGrid)
        else:
            VectorSpace.__init__(self, self.dim, self._to_components_from_SHCoeffs, self._from_components_to_SHCoeffs)



class Sobolev(SHToolsHelper, HilbertSpace):

    def __init__(self, lmax, order, scale , /, *,  vector_as_SHGrid=True, radius=1,  grid = "DH"):
        self._order = order
        self._scale = scale
        SHToolsHelper.__init__(self, lmax, radius=radius, grid=grid)                        
        self._metric_tensor = self._degree_dependent_scaling_to_diagonal_matrix(self._sobolev_function)
        self._inverse_metric_tensor = self._degree_dependent_scaling_to_diagonal_matrix(lambda l : 1 / self._sobolev_function(l))
        if vector_as_SHGrid:
            HilbertSpace.__init__(self, self.dim, self._to_components_from_SHGrid, self._from_components_to_SHGrid, 
                                  self._inner_product_impl, to_dual= self._to_dual_impl, from_dual=self._from_dual_impl)
        else:
            HilbertSpace.__init__(self, self.dim, self._to_components_from_SHCoeffs, self._from_components_to_SHCoeffs, 
                                  self._inner_product_impl, to_dual= self._to_dual_impl, from_dual=self._from_dual_impl)        


    #=============================================#
    #                   Properties                #
    #=============================================#

    @property
    def order(self):
        return self._order

    @property
    def scale(self):
        return self._scale


    #==============================================#
    #                 Public methods               #
    #==============================================#

    def dirac(self, latitude, longitude, /, *, degrees=True):       
        if degrees:
            colatitude = 90 - latitude
        else:
            colatitude = np.pi/2 - latitude
        coeffs = sh.expand.spharm(self.lmax, colatitude, longitude, normalization="ortho", degrees=degrees)
        c = self._to_components_from_coeffs(coeffs)
        return self.dual.from_components(c)

    def dirac_representation(self, latitude, longitude, /, *, degrees=True):
        up = self.dirac(latitude, longitude, degrees=degrees)
        return self.from_dual(up)


    def point_evaluation_operator(self, lats, lons, /, *, degrees=True):
        
        assert lats.size == lons.size    
        codomain = EuclideanSpace(lats.size)

        def mapping(u):
            ulm = u.expand(normalization=self.normalization, csphase=self.csphase)
            return ulm.expand(lat=lats, lon=lons, degrees=degrees)

        def dual_mapping(vp):
            cvp = codomain.dual.to_components(vp)
            cup = np.zeros(self.dim)
            for (c, lat,lon) in zip(cvp, lats, lons):                
                dirac = self.dirac(lat, lon, degrees=degrees)
                cup += c * self.dual.to_components(dirac)
            return LinearForm(self, components=cup)

        return LinearOperator(self, codomain, mapping, dual_mapping=dual_mapping)


    def invariant_operator(self, codomain, f):
        if not isinstance(codomain, Sobolev):
            raise ValueError("Codomain must be another Sobolev space on a sphere.")        
        matrix = self._degree_dependent_scaling_to_diagonal_matrix(f)
        if codomain == self:
            mapping = lambda x : self.from_components(matrix @ self.to_components(x))
            return LinearOperator.self_adjoint(self, mapping)
        else:
            mapping = lambda x : codomain.from_components(matrix @ self.to_components(x))
            dual_mapping = lambda yp : self.dual.from_components(matrix @ codomain.dual.to_components(yp))            
            return LinearOperator(self,  codomain, mapping, dual_mapping=dual_mapping)


    def invariant_gaussian_measure(self, f, /, *, expectation = None):
        g = lambda l : np.sqrt(f(l) / (self.radius**2 * self._sobolev_function(l)))                
        h = lambda l : np.sqrt( self.radius**2 * self._sobolev_function(l) * f(l))
        matrix = self._degree_dependent_scaling_to_diagonal_matrix(g)
        adjoint_matrix = self._degree_dependent_scaling_to_diagonal_matrix(h)
        domain = EuclideanSpace(self.dim)
        mapping = lambda c : self.from_components(matrix @ c)        
        adjoint_mapping = lambda u : adjoint_matrix@ self.to_components(u)        
        factor = LinearOperator(domain, self, mapping, adjoint_mapping= adjoint_mapping)
        return GaussianMeasure.from_factored_covariance(factor, expectation=expectation)


    def sobolev_gaussian_measure(self, order, scale, amplitude, /, *,  expectation=None):               
        f = lambda l : (1 + scale**2 * l *(l+1))**(-order) 
        return self.invariant_gaussian_measure(self._normalise_covariance_function(f,amplitude),
                                               expectation=expectation)    
    
    def heat_kernel_gaussian_measure(self, scale, amplitude, /, *, expectation=None):
        f = lambda l : np.exp(-0.5 * l*(l+1) * scale**2)
        return self.invariant_gaussian_measure(self._normalise_covariance_function(f,amplitude),
                                               expectation=expectation)

    #==============================================#
    #                Private methods               #
    #==============================================#

    def _sobolev_function(self, l):
        return (1 + self.scale**2 * l *(l+1))**self.order

    def _inner_product_impl(self, u, v):
        return self.radius**2 * np.dot(self._metric_tensor @ self.to_components(u), self.to_components(v))        

    def _to_dual_impl(self, u):
        c = self._metric_tensor @ self.to_components(u) * self.radius**2
        return self.dual.from_components(c)

    def _from_dual_impl(self, up):
        c = self._inverse_metric_tensor@ self.dual.to_components(up) / self.radius**2
        return self.from_components(c)

    def _normalise_covariance_function(self, f, amplitude):        
        sum = 0
        for l in range(self.lmax+1):
            sum +=  f(l) * (2*l+1) / (4*np.pi * self.radius**2 * self._sobolev_function(l))
        return lambda l : amplitude**2 * f(l) / sum


class Lebesgue(Sobolev):

    def __init__(self, lmax, /, *, vector_as_SHGrid=True, radius=1,  grid = "DH"):
        super().__init__(lmax, 0, 0, vector_as_SHGrid=vector_as_SHGrid, radius=radius, grid=grid)    


    