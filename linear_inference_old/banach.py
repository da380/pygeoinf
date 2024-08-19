from linear_inference import vector

# Class for a Banach space. 
class Space(vector.Space):

    def __init__(self, dimension, to_components, from_components, norm):
        super(Space,self).__init__(dimension, to_components, from_components)
        self._norm = norm

    def norm(self, x):
        return self._norm(x)

    @staticmethod
    def from_space(space, norm):
        return Space(space.dimension, space.to_components, space.from_components, norm)

