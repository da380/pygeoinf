import matplotlib.pyplot as plt

import numpy as np
from numpy import pi


import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev


class TestClass:

    def __init__(self, lmax: int, radius: float, grid: str):
        self.lmax = lmax
        self.radius = radius
        self.grid = grid

    def lebesgue_load_space(self) -> Lebesgue:
        return Lebesgue(
            self.lmax,
            radius=self.radius,
            grid=self.grid,
        )

    def lebesgue_response_space(self) -> inf.HilbertSpaceDirectSum:
        field_space = self.lebesgue_load_space()
        return inf.HilbertSpaceDirectSum(
            [field_space, field_space, field_space, inf.EuclideanSpace(2)]
        )

    def sobolev_load_space(self, order: float, scale: float) -> Sobolev:
        return Sobolev(self.lmax, order, scale, radius=self.radius, grid=self.grid)

    def sobolev_response_space(
        self, order: float, scale: float
    ) -> inf.HilbertSpaceDirectSum:
        field_space = Sobolev(
            self.lmax, order + 1, scale, radius=self.radius, grid=self.grid
        )
        return inf.HilbertSpaceDirectSum(
            [field_space, field_space, field_space, inf.EuclideanSpace(2)]
        )

    def as_lebesgue_linear_operator(self) -> inf.LinearOperator:

        domain = self.lebesgue_load_space()
        codomain = self.lebesgue_response_space()

        def mapping(u):
            return [u, u, u, np.zeros(2)]

        def adjoint_mapping(response):
            return sum(response[:3])

        return inf.LinearOperator(
            domain, codomain, mapping, adjoint_mapping=adjoint_mapping
        )

    def as_sobolev_linear_operator(self, order: float, scale: float):

        domain = self.sobolev_load_space(order, scale)
        codomain = self.sobolev_response_space(order, scale)

        lebesgue_operator = self.as_lebesgue_linear_operator()

        for l1, s2 in zip(
            lebesgue_operator.codomain.subspaces[:3], codomain.subspaces[:3]
        ):
            l2 = s2.underlying_space
            # print(l1.lmax, l1.radius, l1.grid)
            # print(l2.lmax, l2.radius, l2.grid)
            # print(l1 == l2)

        return inf.LinearOperator.from_formal_adjoint(
            domain, codomain, lebesgue_operator
        )


test = TestClass(32, 1, "DH")

A = test.as_lebesgue_linear_operator()

X = A.domain
Y = A.codomain

u = X.random()
v = Y.random()

lhs = Y.inner_product(A(u), v)
rhs = X.inner_product(u, A.adjoint(v))

print(lhs, rhs)

B = test.as_sobolev_linear_operator(1, 1)

X = B.domain
Y = B.codomain

u = X.random()
v = Y.random()

lhs = Y.inner_product(B(u), v)
rhs = X.inner_product(u, B.adjoint(v))

print(lhs, rhs)
