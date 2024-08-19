import numpy as np
from scipy.stats import norm

if __name__ == "__main__":
    pass

'''
 Wrapper class for a real Hilbert Space. To form an instance, the user provides the following:

 dimension: The dimension of the vector space. 

 fromComponents: A function that maps an array of components to the corresponding vector.

 toComponents: A function that maps a vector to an array of its components.

 innerProduct: A function that implements the inner product on the space. 

 riesz : A function that maps a dual vector to its representation within the space. 

 inverseRiesz: A function that maps a vector to its representation within the dual space.  
'''
class Space:


    def __init__(self, dimension, fromComponents, toComponents, 
                 innerProduct, riesz, inverseRiesz):
        self._dimension = dimension
        self._fromComponents = fromComponents
        self._toComponents  = toComponents
        self._innerProduct = innerProduct
        self._riesz = riesz
        self._inverseRiesz = inverseRiesz
        
    # Return the dimension. 
    @property
    def Dimension(self):
        return self._dimension

    # Return a vector given its components.
    def FromComponents(self,x):
        return self._fromComponents(x)

    # Return the components of a vector.
    def ToComponents(self,x):
        return self._toComponents(x)

    # Return the inner product of two vectors.
    def InnerProduct(self,x1,x2):
        return self._innerProduct(x1, x2)

    # Return the norm of a vector.
    def Norm(self,x):
        return np.sqrt(self.InnerProduct(x,x))

    # Return the vector corresponding to a dual vector. 
    def Riesz(self,x):
        return self._riesz(x)

    # Return the dual vector corresponding to a vector. 
    def InverseRiesz(self,x):
        return self._inverseRiesz(x)
    

    # Return the dual Space. 
    @property
    def Dual(self):
        fromComponents = lambda x : self.InverseRiesz(self.FromComponents(x))
        toComponents   = lambda x : self.ToComponents(self.Riesz(x))
        innerProduct   = lambda x1, x2 : self.innerProduct(self.InverseRiesz(x1), self.InverseRiesz(x2))
        return Space(self.Dimension, fromComponents, toComponents, innerProduct, self.InverseRiesz, self.Riesz)


    # Return the direct sum with another space. 
    def DirectSum(self,other):
        dimension = self.Dimension + other.Dimension
        fromComponets = lambda x : (self.FromComponents(x[:self.Dimension]), other.FromComponents(x[other.Dimension+1:]))
        toComponents  = lambda x : np.append(self.ToComponents(x[0]),other.ToComponents(x[1])) 
        innerProduct = lambda x0, x1 : self.InnerProduct(x0[0],x1[0]) + other.InnerProduct(x0[1],x1[1])
        riesz = lambda x : (self.Riesz(x[0]),other.Riesz(x[1]))
        inverseRiesz = lambda x : (self.InverseRiesz(x[0]),other.InverseRiesz(x[1]))
        return Space(dimension, fromComponets, toComponents, innerProduct, riesz, inverseRiesz)


    # Return the zero vector.
    @property
    def Zero(self):
        return self.FromComponents(np.zeros(self.Dimension))

    # Return a random vector.
    def Random(self, dist = norm(loc = 0, scale = 1)):
        return self.FromComponents(dist.rvs(self.Dimension))





    




    

    
