from LinearInference.Operator import SelfAdjoint


import numpy as np


if __name__ == "__main__":
    pass

class Measure:

    def __init__(self, space, mean, covariance, sample):
        self._space = space
        self._mean = mean
        self._covariance = covariance
        self._sample = sample

    # Return the Hilbert space.
    @property
    def Space(self):
        return self._space

    # Return the *value* of the mean
    @property
    def Mean(self):
        return self._mean()

    # Return the covariance as a SelfAdjointOperator. 
    @property
    def Covariance(self):                
        return SelfAdjoint(self.Space,self._covariance)

    # Return a sample from the measure.
    def Sample(self):
        return self._sample()

    # Return the push-forward of the measure under an affine mapping. 
    # By deafult, the translational part of the mapping is zero. 
    def Affine(self, A, a = None):
        assert A.Domain == self.Space
        space = A.CoDomain
        covariance = A @ self.Covariance @ A.Adjoint
        if a is None:                    
            mean = lambda : self.Mean() 
            sample = lambda: A(self._sample())
        else:
            mean = lambda : self.Mean() + a
            sample = lambda: A(self._sample()) + a        
        return Measure(space,mean,covariance, sample)

    # Return sum of the measure with a second one.
    def __add__(self,other):
        assert self.Space == other.Space
        mean = lambda : self.Mean + other.Mean
        covariance = lambda x : self.Covariance(x) + other.Covariance(x)
        sample = lambda : self.Sample() + other.Sample()
        return Measure(self.Space, mean, covariance, sample)

    # Return difference of the measure with a second one.
    def __sub__(self,other):
        assert self.Space == other.Space
        mean = lambda : self.Mean - other.Mean
        covariance = lambda x : self.Covariance(x) + other.Covariance(x)
        sample = lambda : self.Sample() - other.Sample()
        return Measure(self.Space, mean, covariance, sample)   

    # Return the product of the distribution with a scalar.
    def __mul__(self,scalar):
        mean = lambda : scalar * self.Mean
        covariance = lambda x : scalar * scalar * self.Covariance(x)
        sample = lambda : scalar * self.Sample()
        return Measure(self.Space, mean, covariance, sample)

    def __rmul__(self,scalar):
        return self * scalar

    # Return the quotient of the distribution with a scalar.
    def __div__(self,scalar):
        mean = lambda :  self.Mean / scalar
        covariance = lambda x :  self.Covariance(x) / (scalar * scalar)
        sample = lambda : self.Sample() / scalar
        return Measure(self.Space, mean, covariance, sample)


    # Return the direct sum of two measures.
    def DirectSum(self,other):
        space = self.Space.DirectSum(other.Space)
        mean = lambda : (self.Mean, other.Mean)
        covariance = lambda x : (self.Covariance(x[0]), other.Covariance(x[1]))
        sample = lambda : (self.Sample(), other.Sample()) 
        return Measure(space, mean, covariance, sample)

        