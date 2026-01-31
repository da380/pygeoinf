"""
Module for Backus-Gilbert like methods for solving inference problems. To be done...
"""

from __future__ import annotations

import numpy as np

from .hilbert_space import HilbertSpace, Vector
from .linear_operators import LinearOperator
from .nonlinear_forms import NonLinearForm
from .convex_analysis import SupportFunction



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


class DualMasterCostFunction(NonLinearForm):
    """
    Cost function for the master dual equation (Hilbert form):

        h_U(q)
        = inf_{λ ∈ D}
          { (λ, d̃)_D + σ_B(T* q - G* λ) + σ_V(-λ) }

    i.e.

        φ(λ; q) = (λ, d̃)_D + σ_B(T* q - G* λ) + σ_V(-λ)

    where:
      - σ_B is the support function of the model prior convex set B ⊆ M
      - σ_V is the support function of the data error convex set V ⊆ D

    Minimizing φ(λ; q) over λ ∈ D yields h_U(q).
    """

    def __init__(
        self,
        data_space: HilbertSpace,
        property_space: HilbertSpace,
        model_space: HilbertSpace,
        G: LinearOperator,
        T: LinearOperator,
        model_prior_support: SupportFunction,
        data_error_support: SupportFunction,
        observed_data: Vector,
        q_direction: Vector,
    ) -> None:

        self._validation(data_space, property_space, model_space,
                         G, T, model_prior_support, data_error_support,
                         observed_data, q_direction)
        self._data_space = data_space
        self._property_space = property_space
        self._model_space = model_space
        self._G = G
        self._T = T
        self._model_prior_support = model_prior_support
        self._data_error_support = data_error_support
        self._observed_data = observed_data
        self._q = q_direction

        self._Tstar_q = self._T.adjoint(q_direction)

        super().__init__(
            data_space, self._mapping, subgradient=self._subgradient
        )

    @property
    def observed_data(self) -> Vector:
        """Observed data vector d̃ ∈ D."""
        return self._observed_data

    @property
    def direction(self) -> Vector:
        """Current property direction q ∈ P."""
        return self._q

    def set_direction(self, q: Vector) -> None:
        """Update the property direction q and recompute T* q."""
        if not self._property_space.is_element(q):
            raise ValueError("q must be an element of property_space")
        self._q = q
        self._Tstar_q = self._T.adjoint(q)

    def _mapping(self, lam: Vector) -> float:
        # Term 1: ⟨λ, d̃⟩_D
        term1 = self.domain.inner_product(lam, self._observed_data)

        # Term 2: σ_B(T*q - G*λ)
        Gstar_lam = self._G.adjoint(lam)
        hilbert_residual = self._model_space.subtract(self._Tstar_q, Gstar_lam)
        term2 = self._model_prior_support(hilbert_residual)

        # Term 3: σ_V(-λ)
        neg_lam = self.domain.negative(lam)
        term3 = self._data_error_support(neg_lam)

        return term1 + term2 + term3

    def _subgradient(self, lam: Vector) -> Vector:
        """
        Compute a subgradient of φ(λ; q).

        For the dual master cost function:
            φ(λ) = ⟨λ, d̃⟩ + σ_B(T*q - G*λ) + σ_V(-λ)

        A subgradient is:
            g ∈ d̃ - G*∂σ_B(T*q - G*λ) - ∂σ_V(-λ)

        where ∂σ_B and ∂σ_V are subdifferentials of the support functions.
        The support_point() method returns an element of the subdifferential.
        """
        term1_subgrad = self._observed_data

        Gstar_lam = self._G.adjoint(lam)
        hilbert_residual = self._model_space.subtract(self._Tstar_q, Gstar_lam)

        v = self._model_prior_support.support_point(hilbert_residual)
        w = self._data_error_support.support_point(self.domain.negative(lam))

        if v is None or w is None:
            return self._finite_difference_gradient(lam)

        term2_subgrad = self.domain.negative(self._G(v))
        term3_subgrad = self.domain.negative(w)

        return self.domain.add(
            term1_subgrad,
            self.domain.add(term2_subgrad, term3_subgrad),
        )

    def _gradient(self, lam: Vector) -> Vector:
        """
        Alias for _subgradient for backward compatibility.

        Note: This function is technically a subgradient, not a gradient,
        since the dual master cost function is non-smooth due to the
        support functions. The gradient() method will call this, but
        users should prefer using subgradient() for clarity.
        """
        return self._subgradient(lam)

    def _finite_difference_gradient(
        self, lam: Vector, eps: float = 1e-6
    ) -> Vector:
        if eps <= 0:
            raise ValueError("eps must be positive")

        comps = self.domain.to_components(lam)
        grad = np.zeros_like(comps, dtype=float)

        for i in range(self.domain.dim):
            step = np.zeros_like(comps, dtype=float)
            step[i] = eps

            lam_plus = self.domain.from_components(comps + step)
            lam_minus = self.domain.from_components(comps - step)

            f_plus = self._mapping(lam_plus)
            f_minus = self._mapping(lam_minus)
            grad[i] = (f_plus - f_minus) / (2.0 * eps)

        return self.domain.from_components(grad)

    def _validation(self, data_space, property_space, model_space, G, T, model_prior_support, data_error_support, observed_data, q_direction) -> None:
        if not isinstance(data_space, HilbertSpace):
            raise ValueError("data_space must be a HilbertSpace")
        if not isinstance(property_space, HilbertSpace):
            raise ValueError("property_space must be a HilbertSpace")
        if not isinstance(model_space, HilbertSpace):
            raise ValueError("model_space must be a HilbertSpace")

        if not isinstance(G, LinearOperator):
            raise ValueError("G must be a LinearOperator")
        if not isinstance(T, LinearOperator):
            raise ValueError("T must be a LinearOperator")

        if G.domain != model_space or G.codomain != data_space:
            raise ValueError("G must map from model_space to data_space")
        if T.domain != model_space or T.codomain != property_space:
            raise ValueError("T must map from model_space to property_space")

        if not isinstance(model_prior_support, SupportFunction):
            raise ValueError("model_prior_support must be a SupportFunction")
        if not isinstance(data_error_support, SupportFunction):
            raise ValueError("data_error_support must be a SupportFunction")

        if model_prior_support.primal_domain != model_space:
            raise ValueError(
                "model_prior_support must be defined on model_space"
            )
        if data_error_support.primal_domain != data_space:
            raise ValueError(
                "data_error_support must be defined on data_space"
            )

        if not data_space.is_element(observed_data):
            raise ValueError("observed_data must be an element of data_space")
        if not property_space.is_element(q_direction):
            raise ValueError(
                "q_direction must be an element of property_space"
            )