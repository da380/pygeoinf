from linear_inference import banach

# Class for a Hilbert space.
class Space(banach.Space):

    def __init__(self, dimension, to_components, from_components, inner_product):
        super(Space,self).__init__(dimension, to_components, from_components, 
                                     lambda x : np.sqrt(inner_product(x,x)))
        self._inner_product = inner_product

    @staticmethod 
    def from_space(space, inner_product):
        return Hilbert(space.dimension, space.to_components, space.from_components, inner_product)

    def inner_product(self, x1, x2):
        return self._inner_product(x1,x2)
    
    

# Class for a Hilbert space and a realisation of its dual. 
class PrimalDual(Space):

    def __init__(self, primal, dual, to_dual, from_dual):
        assert primal.dimension == dual.dimension
        self._primal = primal
        self._dual = dual
        self._to_dual = to_dual
        self._from_dual = from_dual
        super(PrimalDual,self).__init__(primal.dimension, primal.to_components,
                                    primal.from_components, primal.inner_product)


    @property
    def dual(self):
        return PrimalDual(self._dual, self._primal, self._from_dual, self._to_dual)

    
    