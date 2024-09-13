import pytest
import numpy as np
from scipy.stats import norm
from pygeoinf.linalg import VectorSpace


@pytest.mark.parametrize("dim", [(0), (10), (24)])
class TestVectorSpace:

    
    def space(self, dim):                
        to_components = lambda x : x.reshape(dim,1)
        from_components = lambda c : c.reshape(dim,)
        return VectorSpace(dim, to_components, from_components)

    def test_zeros(self, dim):
        space = self.space(dim)
        assert np.all(space.zero == np.zeros(space.dim))


    def test_mutual_inverse(self, dim):
        space = self.space(dim)
        c1 = norm().rvs((dim,1))
        c2 = space.to_components(space.from_components(c1))
        assert np.all(c1 == c2)
        