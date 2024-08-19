import numpy as np

# Interface class for a vector space. 
class Vector:

    def __init__(self, dimension, toComponents, fromComponents):
        self._dimension = dimension
        self._toComponents = toComponents
        self._fromComponents = fromComponents

    @property
    def Dimension(self):
        return self._dimension

    def ToComponents(self, x):
        return self._toComponents(x)

    def FromComponents(self, c):
        return self._fromComponents(c)


    @property Zero(self):
    


# Interface class for a Banach space. 
class Banach(Vector):

    def __init__(self, dimension, toComponents, fromComponents, norm):
        super(Banach,self).__init__(dimension, toComponents, fromComponents)
        self._norm = norm

    def Norm(self, x):
        return self._norm(x)

