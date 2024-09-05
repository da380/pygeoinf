if __name__ == "__main__":
    pass


import numpy as np
import matplotlib.pyplot as plt
from pyshtools import SHCoeffs, SHGrid
from pyshtools.expand import spharm
from scipy.sparse import diags 
from pygeoinf.vector_space import HilbertSpace
from pygeoinf.euclidean import Euclidean
from pygeoinf.linear_operator import LinearOperator
from pygeoinf.linear_form import LinearForm
from pygeoinf.gaussian_measure import GaussianMeasure






# Implements the Sobolev space H^s on a sphere using a spherical harmonic basis. The inner product
# is defined in terms of the Laplace-Beltrami operator, \Delta, by:
#
# (u,v)_{H^s} = ( \Lambda^{s/2} u, \Lambda^{s/2} v)_{L^2}
#
# where \Lambda = 1 + \lambda^{2} \Delta and \lambda is the chosen length-scale. 
class Sobolev(HilbertSpace):
    
    def __init__(self, order, scale,  /, *, lmax = None, rtol = 1.e-6, power_of_two = False,
                 radius = 1, elements_as_SHGrid = True, csphase = 1, grid = "DH", extend = True):    

        # Store the basic information. 
        self._order = order
        self._radius = radius  
        self._scale = scale         
        
        # Deduce lmax is not given. 
        if lmax is None:
            self._lmax = self._estimate_lmax(rtol,power_of_two)
        else:
            self._lmax = lmax

        # Store SHTools options. 
        self._elements_as_SHGrid = elements_as_SHGrid   
        self._normalization = "ortho"
        if csphase in [-1,1]:
            self._csphase = csphase
        else:
            raise ValueError("invalid csphase choice")
        if grid in ["DH", "DH2", "GLQ"]:            
            self._grid = grid
            if grid == "DH2":
                self._sampling = 2
            else:
                self._sampling = 1            
        else:
            raise ValueError("invalid grid choice")        
        self._extend = extend

        # Construct the metric and its inverse.         
        metric_values = self._scaling_to_diagonal_values(self._sobolev_function,self.lmax)
        self._metric =  self._diagonal_values_to_matrix(metric_values)
        inverse_metric_values = np.reciprocal(metric_values)
        self._inverse_metric = self._diagonal_values_to_matrix(inverse_metric_values)

        # Set the mappings to and from components.                         
        if elements_as_SHGrid:
            to_components = self._to_components_from_SHGrid
            from_components = self._from_components_to_SHGrid
        else:
            to_components = self._to_components_from_SHCoeffs
            from_components = self._from_components_to_SHCoeffs

        # Construct the base class.         
        dim = (self.lmax+1)**2
        super(Sobolev, self).__init__(dim, to_components, from_components, 
                                       self._inner_product_local, from_dual = self._from_dual_local, 
                                       to_dual = self._to_dual_local)


    
    # Estimate lmax based on order and scale. 
    def _estimate_lmax(self, rtol, power_of_two):
        err = 1
        l = 0 
        sum = 0
        while err > rtol:
            term = (2*l+1) * (1 + self.scale**2 * l*(l+1))**-self.order
            sum += term
            err = term/sum
            l += 1
        
        if power_of_two:
            p = np.log2(l)
            l = 2**int(p) if p -int(p) == 0 else 2**(int(p)+1)

        return l

        



    # Return maximum degree. 
    @property
    def lmax(self):
        return self._lmax            

    # Return the sobolev order. fraction
    @property
    def order(self):
        return self._order

    # Return the radius. 
    @property
    def radius(self):
        return self._radius

    # Return the relative length scale. 
    @property 
    def scale(self):
        return self._scale

    # Return the scaling function within the Sobolev inner product. 
    def _sobolev_function(self,l):
        return  (1 + self.scale**2 * l * (l + 1))**self.order

    # Expand a SHGrid or SHCoeffs object using the stored conventions. 
    def expand(self, u):
        if isinstance(u,SHGrid):
            return u.expand(normalization = self._normalization, csphase = self._csphase)
        elif isinstance(u,SHCoeffs):
            return u.expand(grid = self._grid, extend = self._extend)

    # Return the component index for the (l,m)th spherical harmonic coefficient
    def spherical_harmonic_index(self, l, m):
        if m >= 0:
            return int(l*(l+1)/2) + m
        else:
            offset = int((self.lmax + 1)*(self.lmax + 2) / 2)
            return offset + int((l - 1) * l / 2) - m - 1


    # Flatten shtools coefficient format into contiguous vector. 
    def _to_components_from_coeffs(self, coeffs):
        c = np.empty(self.dim)        
        i = 0
        for l in range(self.lmax+1):
            j = i + l + 1
            c[i:j] = coeffs[0,l,:l+1] * self.radius   
            i = j
        for l in range(1,self.lmax+1):
            j = i + l
            c[i:j] = coeffs[1,l,1:l+1] * self.radius
            i = j    
        return c

    # Map a SHCoeffs object to its components as a contiguous vector. 
    def _to_components_from_SHCoeffs(self, ulm):        
        return self._to_components_from_coeffs(ulm.coeffs)

    # Map a SHGrid object to its components.
    def _to_components_from_SHGrid(self, u):        
        return self._to_components_from_SHCoeffs(self.expand(u))

    # Map components to a SHCoeffs object.
    def _from_components_to_SHCoeffs(self,c):
        coeffs = np.zeros((2,self.lmax+1, self.lmax+1))
        i = 0
        for l in range(self.lmax+1):
            j = i + l + 1
            coeffs[0,l,:l+1] = c[i:j] / self.radius
            i = j
        for l in range(1,self.lmax+1): 
            j = i + l
            coeffs[1,l,1:l+1] = c[i:j] / self.radius
            i = j    
        ulm = SHCoeffs.from_array(coeffs, normalization = self._normalization, csphase = self._csphase)
        return ulm

    # Map components to a SHGrid object. 
    def _from_components_to_SHGrid(self,c):        
        return self.expand(self._from_components_to_SHCoeffs(c))

    # Convert a degree dependent scaling function to vector of diagonal values.
    @staticmethod
    def _scaling_to_diagonal_values(f, lmax):
        dim = (lmax+1)**2
        values = np.zeros(dim)
        i = 0
        for l in range(lmax+1):
            j = i + l + 1
            values[i:j] = f(l)
            i = j
        for l in range(1,lmax+1):
            j = i + l
            values[i:j] = f(l)
            i = j        
        return values

    # Convert vector of diagonal values to sparse diagonal matrix. 
    @staticmethod
    def _diagonal_values_to_matrix(values):
        return diags([values], [0])

    # Convert a degree dependent scaling function to sparse diagonal matrix.    
    def _scaling_to_diagonal_matrix(self, f):
        return self._diagonal_values_to_matrix(self._scaling_to_diagonal_values(f,self.lmax))    

    # Local definition of the inner product. 
    def _inner_product_local(self, u1, u2):
        c1 = self.to_components(u1)
        c2 = self.to_components(u2)
        return np.dot(self._metric @ c1, c2)

    # Local definition of mapping from the dual. 
    def _from_dual_local(self, up):        
        cp = self.dual.to_components(up)
        c = self._inverse_metric @ cp 
        return self.from_components(c)

    # Local definition of mapping to the dual.
    def _to_dual_local(self, u):
        c = self.to_components(u)
        cp = self._metric @ c
        return LinearForm(self, components = cp)

    # Return the Dirac measure at a given point as a linear form.
    def dirac_form(self, latitude, longitude, /, *, degrees = True):
        if degrees:
            colatitude = 90 - latitude
        else:
            colatitude = np.pi/2 - latitude
        coeffs = spharm(self.lmax, colatitude, longitude, normalization = self._normalization, csphase = self._csphase, degrees = degrees) / self.radius**2
        return LinearForm(self, components = self._to_components_from_coeffs(coeffs))

    # Return the Dirac measure at a given point as an element of the space. 
    def dirac(self, latitude, longitude, /, *, degrees = True):
        #if self.order <= 1:
        #    raise ValueError("Order of the space must be > 1")
        return self.from_dual(self.dirac_form(latitude, longitude, degrees=degrees))

    # Return a rotationally invarient linear operator on the space given the 
    # degree-dependent scaling function. 
    def invariant_linear_operator(self, codomain, f):
        domain = self
        if codomain is None:
            codomain = self
        else:
            assert isinstance(codomain, Sobolev)
        matrix = self._scaling_to_diagonal_matrix(f)
        mapping = lambda u : codomain.from_components(matrix @ domain.to_components(u))
        dual_mapping = lambda up : domain.dual.from_components(matrix @ codomain.dual.to_components(up))
        return LinearOperator(domain, codomain, mapping, dual_mapping=dual_mapping)        

    # Return a rotationally invarient Gaussian measure on th space given the degree-dependent scaling function. 
    def invariant_gaussian_measure(self, f, /, *, mean = None):
        g = lambda l : np.sqrt(f(l) / self._sobolev_function(l))
        matrix = self._scaling_to_diagonal_matrix(g)                    
        mapping = lambda c : self.from_components(matrix @ c)        
        adjoint_mapping = lambda u : self._metric @ matrix @ self.to_components(u) 
        factor = LinearOperator(Euclidean(self.dim), self, mapping, adjoint_mapping= adjoint_mapping)
        return GaussianMeasure.from_factored_covariance(factor, mean = mean)

    # Normalises a the covariance function for an invariant measure such that 
    # the pointwise standard deviation is equal to amplitude. 
    def _normalise_covariance_function(self, f, amplitude):        
        sum = 0
        for l in range(self.lmax+1):
            sum +=  f(l) * (2*l+1) / (4*np.pi * self.radius**2 * self._sobolev_function(l))
        return lambda l : amplitude**2 * f(l) / sum


    # Return a rotationally invariant Gaussian measure with covariance of the Sobolev form.
    def sobolev_gaussian_measure(self, order, scale, amplitude, /, *,  mean = None):               
        f = lambda l : (1 + scale**2 * l *(l+1))**(-order) 
        return self.invariant_gaussian_measure(self._normalise_covariance_function(f,amplitude), mean = mean)        
                        
    # Return a rotationally invariant Gaussian measure iwth covariance of the heat kernel form.
    def heat_kernel_gaussian_measure(self, scale, amplitude, /, *, mean = None):
        f = lambda l : np.exp(-0.5 * l*(l+1) * (scale/self.radius)**2)
        return self.invariant_gaussian_measure(self._normalise_covariance_function(f,amplitude),mean = mean)

    # Make a simple plot of an element of the space. 
    def plot(self,u, /, *, cmap = "RdBu",  show = True, colorbar = True, symmetric = True):        
        if isinstance(u, SHGrid):
            plt.pcolor(u.lons(), u.lats(), u.data,cmap = cmap)
            if colorbar:
                plt.colorbar(orientation = "horizontal")
            if symmetric:
                max = np.max(np.abs(u.data))
                plt.clim([-max,max])
            if show:
                plt.show()            
        else:            
            self.plot(self.expand(u))

            
# Implementation of the Lebesgue space L^{2} on a two-sphere. Obtained as a special case of H^{s} with order set to zero. 
# Note that with this value of s, the value of the length-scale does not matter. 
class Lebesgue(Sobolev):

    def __init__(self, lmax, /, *, radius = 1, scale = 1,
                 elements_as_SHGrid = True, csphase = 1, grid = "DH", extend = True):    
        super(Lebesgue,self).__init__(lmax, 0, 0, radius = radius, elements_as_SHGrid = elements_as_SHGrid,
                                      csphase = csphase, grid = grid, extend = extend)










    


        



            


    


