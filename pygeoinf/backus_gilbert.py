"""
Module for Backus-Gilbert like methods for solving inference problems. To be done...
"""

from __future__ import annotations

from .hilbert_space import HilbertSpace, Vector
from .linear_operators import LinearOperator
from .nonlinear_forms import NonLinearForm



class HyperEllipsoid:
    """
    A class for hyper-ellipsoids in a Hilbert Space. Such sets occur within
    the context of Backus-Gilbert methods, both in terms of prior constraints
    and posterior bounds on the property space.

    The hyper-ellipsoid is defined through the inequality

    (A(x-x_0), x-x_0)_{X} <= r**2,

    where A is a self-adjoint linear operator on the space, X, x is an arbitrary vector, x_0 is the
    centre, and r the radius.
    """

    def __init__(
        self,
        space: HilbertSpace,
        radius: float,
        /,
        *,
        centre: Vector = None,
        operator: LinearOperator = None,
    ) -> None:
        """
        Args:
            space (HilbertSpace): The Hilbert space in which the hyper-ellipsoid is defined.
            radius (float): The radius of the hyper-ellipsoid.
            centre (Vector); The centre of the hyper-ellipsoid. The default is None which corresponds to
                the zero-vector.
            operator (LinearOperator): A self-adjoint operator on the space defining the hyper-ellipsoid.
                The default is None which corresponds to the identity operator.
        """

        if not isinstance(space, HilbertSpace):
            raise ValueError("Input space must be a HilbertSpace")
        self._space = space

        if not radius > 0:
            raise ValueError("Input radius must be positive.")
        self._radius = radius

        if operator is None:
            self._operator = space.identity_operator()
        else:
            if not (operator.domain == space and operator.is_automorphism):
                raise ValueError("Operator is not of the appropriate form.")
            self._operator = operator

        if centre is None:
            self._centre = space.zero
        else:
            if not space.is_element(centre):
                raise ValueError("The input centre does not lie in the space.")
            self._centre = centre

    @property
    def space(self) -> HilbertSpace:
        """
        Returns the HilbertSpace the hyper-ellipsoid is defined on.
        """
        return self._space

    @property
    def radius(self) -> float:
        """
        Returns the radius of the hyper-ellipsoid.
        """
        return self._radius

    @property
    def operator(self) -> LinearOperator:
        """
        Returns the operator for the hyper-ellipsoid.
        """
        return self._operator

    @property
    def centre(self) -> Vector:
        """
        Returns the centre of the hyper-ellipsoid.
        """
        return self._centre

    @property
    def quadratic_form(self) -> NonLinearForm:
        """
        Returns the mapping x -> (A(x-x_0), x-x_0)_{X} as a NonLinearForm.
        """

        space = self.space
        x0 = self.centre
        A = self.operator

        def mapping(x: Vector) -> float:
            d = space.subtract(x, x0)
            return space.inner_product(A(d), d)

        def gradient(x: Vector) -> Vector:
            d = space.subtract(x, x0)
            return space.multiply(2, A(d))

        def hessian(_: Vector) -> LinearOperator:
            return A

        return NonLinearForm(space, mapping, gradient=gradient, hessian=hessian)

    def is_point(self, x: Vector) -> bool:
        """
        True if x lies in the hyper-ellipsoid.
        """
        return self.quadratic_form(x) <= self.radius**2
