
from linear_inference.linear_operator import LinearOperator


# Class for rotationally invariant linear operators acting on functions defined on a two-sphere. 
# Given the domain, the operator is defined in terms of a degree-dependent scaling function 
# that is applied within the spherical harmonic domain. 
class InvariantLinearOperator(LinearOperator):

    def __init__(self, domain, function):
        pass