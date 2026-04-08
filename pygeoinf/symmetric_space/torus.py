from __future__ import annotations
from typing import Callable, Any, Optional, Tuple, List, Union

import numpy as np
from scipy.fft import rfft2, irfft2

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pygeoinf.linear_operators import LinearOperator
from .symmetric_space import AbstractSymmetricLebesgueSpace, SymmetricSobolevSpace


class Lebesgue(AbstractSymmetricLebesgueSpace):
    """Implementation of the Lebesgue space L² on a 2D Torus."""

    def __init__(
        self,
        kmax: int,
        /,
        *,
        radius_x: float = 1.0,
        radius_y: float = 1.0,
    ):
        if kmax <= 0:
            raise ValueError("kmax must be non-negative")
        if radius_x <= 0 or radius_y <= 0:
            raise ValueError("Radii must be positive")

        self._kmax: int = kmax
        self._radius_x: float = radius_x
        self._radius_y: float = radius_y

        self._fft_factor: float = np.sqrt(4 * np.pi**2 * radius_x * radius_y) / (
            4 * kmax**2
        )
        self._inverse_fft_factor: float = 1.0 / self._fft_factor

        dim = 4 * kmax**2
        self._build_index_map(dim)

        AbstractSymmetricLebesgueSpace.__init__(self, 2, kmax, dim, False)

    @classmethod
    def from_covariance(
        cls,
        covariance_function: Callable[[float], float],
        /,
        *,
        radius_x: float = 1.0,
        radius_y: float = 1.0,
        rtol: float = 1e-6,
        min_degree: int = 1,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
    ) -> Lebesgue:
        dummy_space = cls(max(1, min_degree), radius_x=radius_x, radius_y=radius_y)
        optimal_degree = dummy_space.estimate_truncation_degree(
            covariance_function, rtol=rtol, min_degree=min_degree, max_degree=max_degree
        )
        if power_of_two:
            n = int(np.log2(optimal_degree))
            optimal_degree = 2 ** (n + 1)
        return cls(optimal_degree, radius_x=radius_x, radius_y=radius_y)

    @classmethod
    def from_heat_kernel_prior(
        cls,
        kernel_scale: float,
        /,
        *,
        radius_x: float = 1.0,
        radius_y: float = 1.0,
        rtol: float = 1e-6,
        min_degree: int = 1,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
    ) -> Lebesgue:
        return cls.from_covariance(
            cls.heat_kernel(kernel_scale),
            radius_x=radius_x,
            radius_y=radius_y,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
            power_of_two=power_of_two,
        )

    @classmethod
    def from_sobolev_kernel_prior(
        cls,
        kernel_order: float,
        kernel_scale: float,
        /,
        *,
        radius_x: float = 1.0,
        radius_y: float = 1.0,
        rtol: float = 1e-6,
        min_degree: int = 1,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
    ) -> Lebesgue:
        return cls.from_covariance(
            cls.sobolev_kernel(kernel_order, kernel_scale),
            radius_x=radius_x,
            radius_y=radius_y,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
            power_of_two=power_of_two,
        )

    @property
    def kmax(self) -> int:
        return self._kmax

    @property
    def radius_x(self) -> float:
        return self._radius_x

    @property
    def radius_y(self) -> float:
        return self._radius_y

    @property
    def fft_factor(self) -> float:
        return self._fft_factor

    @property
    def inverse_fft_factor(self) -> float:
        return self._inverse_fft_factor

    def points(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.arange(2 * self.kmax) * np.pi / self.kmax
        y = np.arange(2 * self.kmax) * np.pi / self.kmax
        X, Y = np.meshgrid(x, y, indexing="ij")
        return X.flatten(), Y.flatten()

    def project_function(self, f: Callable[[Tuple[float, float]], float]) -> np.ndarray:
        X, Y = self.points()
        Z = np.fromiter((f((x, y)) for x, y in zip(X, Y)), float)
        return Z.reshape((2 * self.kmax, 2 * self.kmax))

    def to_coefficients(self, u: np.ndarray) -> np.ndarray:
        return rfft2(u) * self.fft_factor

    def from_coefficients(self, coeff: np.ndarray) -> np.ndarray:
        return (
            irfft2(coeff, s=(2 * self.kmax, 2 * self.kmax)) * self._inverse_fft_factor
        )

    def to_coefficient_operator(self, kmax: int) -> LinearOperator:
        target_space = self.with_degree(kmax)
        transfer_op = self.degree_transfer_operator(kmax)
        return target_space.coordinate_projection @ transfer_op

    def from_coefficient_operator(self, kmax: int) -> LinearOperator:
        target_space = self.with_degree(kmax)
        transfer_op = self.degree_transfer_operator(kmax)
        return transfer_op.adjoint @ target_space.coordinate_inclusion

    def wavevector_indices(self, kx: int, ky: int) -> List[int]:
        if ky < 0 or ky > self.kmax:
            raise ValueError(
                f"Due to Real-FFT symmetry, ky must be in [0, {self.kmax}]."
            )
        mask = (self._kx_freqs == kx) & (self._ky_freqs == ky)
        return np.where(mask)[0].tolist()

    def spectral_projection_operator(
        self, modes: List[Tuple[int, int]]
    ) -> LinearOperator:
        indices = []
        for kx, ky in modes:
            indices.extend(self.wavevector_indices(kx, ky))
        unique_indices = sorted(list(set(indices)))

        if not unique_indices:
            raise ValueError(
                "No valid wavevector indices found for the provided modes."
            )

        full_coeff_op = self.to_coefficient_operator(self.kmax)
        projection_op = full_coeff_op.codomain.subspace_projection(unique_indices)
        return projection_op @ full_coeff_op

    def integer_to_index(self, i: int) -> int:
        return i

    def index_to_integer(self, k: int) -> int:
        return k

    def laplacian_eigenvalue(self, k: int) -> float:
        return self._eigenvalues[k]

    def laplacian_eigenvector_squared_norm(self, k: int) -> float:
        return self._squared_norms[k]

    def laplacian_eigenvectors_at_point(self, pt: Tuple[float, float]) -> np.ndarray:
        tx, ty = pt
        phase = self._kx_freqs * tx + self._ky_freqs * ty
        vals = np.where(self._is_imag, -np.sin(phase), np.cos(phase))
        norm_factor = np.sqrt(4 * np.pi**2 * self.radius_x * self.radius_y)
        return self._metric @ vals / norm_factor

    def random_point(self) -> Tuple[float, float]:
        return (np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi))

    def geodesic_distance(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        dx = (p2[0] - p1[0] + np.pi) % (2 * np.pi) - np.pi
        dy = (p2[1] - p1[1] + np.pi) % (2 * np.pi) - np.pi
        return float(np.sqrt((dx * self.radius_x) ** 2 + (dy * self.radius_y) ** 2))

    def geodesic_quadrature(
        self, p1: Tuple[float, float], p2: Tuple[float, float], n_points: int
    ) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        arc_length = self.geodesic_distance(p1, p2)
        dx = (p2[0] - p1[0] + np.pi) % (2 * np.pi) - np.pi
        dy = (p2[1] - p1[1] + np.pi) % (2 * np.pi) - np.pi

        t, w = np.polynomial.legendre.leggauss(n_points)
        t_mapped = (t + 1) / 2.0

        angles_x = p1[0] + t_mapped * dx
        angles_y = p1[1] + t_mapped * dy

        scaled_weights = w * (arc_length / 2.0)
        points = list(zip(angles_x, angles_y))

        return points, scaled_weights

    def with_degree(self, degree: int) -> Lebesgue:
        return Lebesgue(degree, radius_x=self.radius_x, radius_y=self.radius_y)

    def degree_transfer_operator(self, target_degree: int) -> LinearOperator:
        codomain = self.with_degree(target_degree)

        forward_indices = []
        for i in range(self.dim):
            kx = self._kx_freqs[i]
            ky = self._ky_freqs[i]
            is_imag = self._is_imag[i]

            if abs(kx) <= target_degree and ky <= target_degree:
                # -target_degree does not exist in the target grid. Drop it to prevent collisions.
                if kx == -target_degree:
                    continue

                # Nyquist frequencies on the target grid are strictly real
                if is_imag and (abs(kx) == target_degree or ky == target_degree):
                    continue

                mask = (
                    (codomain._kx_freqs == kx)
                    & (codomain._ky_freqs == ky)
                    & (codomain._is_imag == is_imag)
                )
                idx_out = np.where(mask)[0]
                if len(idx_out) == 1:
                    forward_indices.append((i, idx_out[0]))

        in_idx = np.array([pair[0] for pair in forward_indices], dtype=int)
        out_idx = np.array([pair[1] for pair in forward_indices], dtype=int)

        metric_ratio = codomain.squared_norms[out_idx] / self.squared_norms[in_idx]

        def mapping(u: np.ndarray) -> np.ndarray:
            c_in = self.to_components(u)
            c_out = np.zeros(codomain.dim)
            c_out[out_idx] = c_in[in_idx]
            return codomain.from_components(c_out)

        def adjoint_mapping(v: np.ndarray) -> np.ndarray:
            c_in = codomain.to_components(v)
            c_out = np.zeros(self.dim)
            c_out[in_idx] = c_in[out_idx] * metric_ratio
            return self.from_components(c_out)

        return LinearOperator(self, codomain, mapping, adjoint_mapping=adjoint_mapping)

    def invariant_covariance_function(
        self, spectral_variances: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        raise NotImplementedError(
            "Due to the grid anisotropy, an exact 1D distance-to-covariance mapping "
            "is not mathematically defined for the Torus. Use purely diagonal preconditioners."
        )

    def estimate_truncation_degree(
        self,
        covariance_function: Callable[[float], float],
        /,
        *,
        rtol: float = 1e-6,
        min_degree: int = 1,
        max_degree: Optional[int] = None,
    ) -> int:
        summation = 0.0
        K = 0
        err = 1.0

        while err > rtol:
            if max_degree is not None and K >= max_degree:
                return max_degree

            shell_energy = 0.0
            if K == 0:
                shell_energy += covariance_function(0.0)
            else:
                for kx in range(-K, K + 1):
                    for ky in [-K, K]:
                        eval_val = (kx / self.radius_x) ** 2 + (ky / self.radius_y) ** 2
                        shell_energy += covariance_function(eval_val)
                for ky in range(-K + 1, K):
                    for kx in [-K, K]:
                        eval_val = (kx / self.radius_x) ** 2 + (ky / self.radius_y) ** 2
                        shell_energy += covariance_function(eval_val)

            summation += shell_energy

            if summation > 0:
                err = shell_energy / summation
            else:
                err = 1.0

            if err <= rtol:
                break

            K += 1
            if K > 10000:
                raise RuntimeError("Failed to converge on a stable truncation degree.")

        return max(K, min_degree)

    def degree_multiplicity(self, degree: int) -> int:
        return 1

    def representative_index(self, degree: int) -> int:
        return degree

    def to_components(self, x: np.ndarray) -> np.ndarray:
        coeff = self.to_coefficients(x)
        return self._coefficient_to_component(coeff)

    def from_components(self, c: np.ndarray) -> np.ndarray:
        coeff = self._component_to_coefficients(c)
        return self.from_coefficients(coeff)

    def is_element(self, x: Any) -> bool:
        if not isinstance(x, np.ndarray):
            return False
        if not x.shape == (2 * self.kmax, 2 * self.kmax):
            return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Lebesgue):
            return NotImplemented
        return (
            self.kmax == other.kmax
            and self.radius_x == other.radius_x
            and self.radius_y == other.radius_y
        )

    def vector_multiply(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 * x2

    def vector_sqrt(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)

    @property
    def zero(self) -> np.ndarray:
        return np.zeros((2 * self.kmax, 2 * self.kmax))

    def inner_product(self, x: np.ndarray, y: np.ndarray) -> float:
        cx = self.to_components(x)
        cy = self.to_components(y)
        return float(np.dot(cx, self._metric @ cy))

    def norm(self, x: np.ndarray) -> float:
        cx = self.to_components(x)
        return float(np.sqrt(np.clip(np.dot(cx, self._metric @ cx), 0.0, None)))

    def _build_index_map(self, dim: int):
        self._eigenvalues = np.zeros(dim)
        self._squared_norms = np.zeros(dim)
        self._kx_freqs = np.zeros(dim)
        self._ky_freqs = np.zeros(dim)
        self._is_imag = np.zeros(dim, dtype=bool)

        idx = 0

        def add_mode(kx_idx, ky_idx, is_real_only=False):
            nonlocal idx
            kx_freq = kx_idx if kx_idx <= self.kmax else kx_idx - 2 * self.kmax
            ky_freq = ky_idx
            eval_val = (kx_freq / self.radius_x) ** 2 + (ky_freq / self.radius_y) ** 2

            self._eigenvalues[idx] = eval_val
            self._squared_norms[idx] = 1.0 if is_real_only else 2.0
            self._kx_freqs[idx] = kx_freq
            self._ky_freqs[idx] = ky_freq
            self._is_imag[idx] = False
            idx += 1

            if not is_real_only:
                self._eigenvalues[idx] = eval_val
                self._squared_norms[idx] = 2.0
                self._kx_freqs[idx] = kx_freq
                self._ky_freqs[idx] = ky_freq
                self._is_imag[idx] = True
                idx += 1

        add_mode(0, 0, True)
        add_mode(self.kmax, 0, True)
        add_mode(0, self.kmax, True)
        add_mode(self.kmax, self.kmax, True)

        for kx in range(1, self.kmax):
            add_mode(kx, 0, False)

        for kx in range(1, self.kmax):
            add_mode(kx, self.kmax, False)

        for ky in range(1, self.kmax):
            for kx in range(2 * self.kmax):
                add_mode(kx, ky, False)

    def _coefficient_to_component(self, coeff: np.ndarray) -> np.ndarray:
        c = np.empty(self.dim, dtype=float)
        k = self.kmax

        c[0] = coeff[0, 0].real
        c[1] = coeff[k, 0].real
        c[2] = coeff[0, k].real
        c[3] = coeff[k, k].real

        idx_end_0 = 4 + 2 * (k - 1)
        c[4:idx_end_0:2] = coeff[1:k, 0].real
        c[5:idx_end_0:2] = coeff[1:k, 0].imag

        idx_end_k = idx_end_0 + 2 * (k - 1)
        c[idx_end_0:idx_end_k:2] = coeff[1:k, k].real
        c[idx_end_0 + 1 : idx_end_k : 2] = coeff[1:k, k].imag

        interior_flat = coeff[:, 1:k].T.flatten()
        c[idx_end_k::2] = interior_flat.real
        c[idx_end_k + 1 :: 2] = interior_flat.imag

        return c

    def _component_to_coefficients(self, c: np.ndarray) -> np.ndarray:
        k = self.kmax
        coeff = np.zeros((2 * k, k + 1), dtype=complex)

        coeff[0, 0] = c[0]
        coeff[k, 0] = c[1]
        coeff[0, k] = c[2]
        coeff[k, k] = c[3]

        if k > 1:
            idx_end_0 = 4 + 2 * (k - 1)
            vals_0 = c[4:idx_end_0:2] + 1j * c[5:idx_end_0:2]
            coeff[1:k, 0] = vals_0
            coeff[-1:-k:-1, 0] = np.conj(vals_0)

            idx_end_k = idx_end_0 + 2 * (k - 1)
            vals_k = c[idx_end_0:idx_end_k:2] + 1j * c[idx_end_0 + 1 : idx_end_k : 2]
            coeff[1:k, k] = vals_k
            coeff[-1:-k:-1, k] = np.conj(vals_k)

            interior_flat = c[idx_end_k::2] + 1j * c[idx_end_k + 1 :: 2]
            coeff[:, 1:k] = interior_flat.reshape((k - 1, 2 * k)).T

        return coeff


class Sobolev(SymmetricSobolevSpace):
    """Implementation of the Sobolev space Hˢ on a 2D Torus."""

    def __init__(
        self,
        kmax: int,
        order: float,
        scale: float,
        /,
        *,
        radius_x: float = 1.0,
        radius_y: float = 1.0,
        safe: bool = True,
    ):
        lebesgue_space = Lebesgue(kmax, radius_x=radius_x, radius_y=radius_y)
        SymmetricSobolevSpace.__init__(self, lebesgue_space, order, scale, safe=safe)

    @staticmethod
    def from_sobolev_parameters(
        order: float,
        scale: float,
        /,
        *,
        radius_x: float = 1.0,
        radius_y: float = 1.0,
        rtol: float = 1e-6,
        power_of_two: bool = False,
        safe: bool = True,
    ) -> Sobolev:
        if safe and order <= 1.0:
            raise ValueError(
                "Point evaluation on a 2D Torus requires Sobolev order > 1.0"
            )

        dummy = Sobolev(
            1, order, scale, radius_x=radius_x, radius_y=radius_y, safe=False
        )
        kmax = dummy.estimate_truncation_degree(dummy.sobolev_function, rtol=rtol)

        if power_of_two:
            n = int(np.log2(kmax))
            kmax = 2 ** (n + 1)

        return Sobolev(
            kmax, order, scale, radius_x=radius_x, radius_y=radius_y, safe=safe
        )

    @classmethod
    def from_covariance(
        cls,
        covariance_function: Callable[[float], float],
        order: float,
        scale: float,
        /,
        *,
        radius_x: float = 1.0,
        radius_y: float = 1.0,
        rtol: float = 1e-6,
        min_degree: int = 1,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
        safe: bool = True,
    ) -> Sobolev:
        dummy_space = cls(
            max(1, min_degree),
            order,
            scale,
            radius_x=radius_x,
            radius_y=radius_y,
            safe=False,
        )
        optimal_degree = dummy_space.estimate_truncation_degree(
            covariance_function, rtol=rtol, min_degree=min_degree, max_degree=max_degree
        )
        if power_of_two:
            n = int(np.log2(optimal_degree))
            optimal_degree = 2 ** (n + 1)

        return cls(
            optimal_degree,
            order,
            scale,
            radius_x=radius_x,
            radius_y=radius_y,
            safe=safe,
        )

    @classmethod
    def from_heat_kernel_prior(
        cls,
        kernel_scale: float,
        order: float,
        scale: float,
        /,
        *,
        radius_x: float = 1.0,
        radius_y: float = 1.0,
        rtol: float = 1e-6,
        min_degree: int = 1,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
        safe: bool = True,
    ) -> Sobolev:
        return cls.from_covariance(
            cls.heat_kernel(kernel_scale),
            order,
            scale,
            radius_x=radius_x,
            radius_y=radius_y,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
            power_of_two=power_of_two,
            safe=safe,
        )

    @classmethod
    def from_sobolev_kernel_prior(
        cls,
        kernel_order: float,
        kernel_scale: float,
        order: float,
        scale: float,
        /,
        *,
        radius_x: float = 1.0,
        radius_y: float = 1.0,
        rtol: float = 1e-6,
        min_degree: int = 1,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
        safe: bool = True,
    ) -> Sobolev:
        return cls.from_covariance(
            cls.sobolev_kernel(kernel_order, kernel_scale),
            order,
            scale,
            radius_x=radius_x,
            radius_y=radius_y,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
            power_of_two=power_of_two,
            safe=safe,
        )

    @property
    def kmax(self) -> int:
        return self.underlying_space.kmax

    @property
    def radius_x(self) -> float:
        return self.underlying_space.radius_x

    @property
    def radius_y(self) -> float:
        return self.underlying_space.radius_y

    def points(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.underlying_space.points()

    @property
    def fft_factor(self) -> float:
        return self.underlying_space.fft_factor

    @property
    def inverse_fft_factor(self) -> float:
        return self.underlying_space.inverse_fft_factor

    def with_order(self, order: float) -> Sobolev:
        return Sobolev(
            self.kmax, order, self.scale, radius_x=self.radius_x, radius_y=self.radius_y
        )

    def with_degree(self, degree: int) -> Sobolev:
        return Sobolev(
            degree,
            self.order,
            self.scale,
            radius_x=self.radius_x,
            radius_y=self.radius_y,
        )

    def project_function(self, f: Callable[[Tuple[float, float]], float]) -> np.ndarray:
        return self.underlying_space.project_function(f)

    def estimate_truncation_degree(
        self,
        covariance_function: Callable[[float], float],
        /,
        *,
        rtol: float = 1e-6,
        min_degree: int = 1,
        max_degree: Optional[int] = None,
    ) -> int:
        """
        Delegates the energy truncation search to the underlying Torus Lebesgue space,
        ensuring it loops geometrically over (kx, ky) shells rather than flat 1D indices.
        """

        # We must wrap the covariance function in the Sobolev weighting factor
        def sobolev_weighted_cov(eval_val: float) -> float:
            return covariance_function(eval_val) * self.sobolev_function(eval_val)

        return self.underlying_space.estimate_truncation_degree(
            sobolev_weighted_cov,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
        )

    # ------------------------------------------------------ #
    #                  Operator Delegations                  #
    # ------------------------------------------------------ #

    def wavevector_indices(self, kx: int, ky: int) -> List[int]:
        return self.underlying_space.wavevector_indices(kx, ky)

    def to_coefficient_operator(self, kmax: int) -> LinearOperator:
        l2_operator = self.underlying_space.to_coefficient_operator(kmax)
        return LinearOperator.from_formal_adjoint(
            self, l2_operator.codomain, l2_operator
        )

    def from_coefficient_operator(self, kmax: int) -> LinearOperator:
        l2_operator = self.underlying_space.from_coefficient_operator(kmax)
        return LinearOperator.from_formal_adjoint(l2_operator.domain, self, l2_operator)

    def spectral_projection_operator(
        self, modes: List[Tuple[int, int]]
    ) -> LinearOperator:
        l2_operator = self.underlying_space.spectral_projection_operator(modes)
        return LinearOperator.from_formal_adjoint(
            self, l2_operator.codomain, l2_operator
        )

    # ------------------------------------------------------ #
    #         Fast Array-Math Overrides & Delegations        #
    # ------------------------------------------------------ #

    def is_element(self, x: Any) -> bool:
        return self.underlying_space.is_element(x)

    def to_components(self, x: np.ndarray) -> np.ndarray:
        return self.underlying_space.to_components(x)

    def from_components(self, c: np.ndarray) -> np.ndarray:
        return self.underlying_space.from_components(c)

    def to_coefficients(self, u: np.ndarray) -> np.ndarray:
        return self.underlying_space.to_coefficients(u)

    def from_coefficients(self, coeff: np.ndarray) -> np.ndarray:
        return self.underlying_space.from_coefficients(coeff)

    def inner_product(self, x: np.ndarray, y: np.ndarray) -> float:
        cx = self.to_components(x)
        cy = self.to_components(y)
        return float(np.dot(cx, self._metric @ cy))

    def norm(self, x: np.ndarray) -> float:
        cx = self.to_components(x)
        return float(np.sqrt(np.clip(np.dot(cx, self._metric @ cx), 0.0, None)))


def plot(
    space: Union[Lebesgue, Sobolev],
    u: np.ndarray,
    /,
    *,
    ax: Optional[Axes] = None,
    contour: bool = False,
    cmap: str = "RdBu",
    symmetric: bool = False,
    contour_lines: bool = False,
    contour_lines_kwargs: Optional[dict] = None,
    num_levels: int = 10,
    colorbar: bool = False,
    colorbar_kwargs: Optional[dict] = None,
    **kwargs,
) -> Tuple[Axes, Any]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    else:
        fig = ax.get_figure()

    x_1d = np.arange(2 * space.kmax) * np.pi / space.kmax
    y_1d = np.arange(2 * space.kmax) * np.pi / space.kmax

    kwargs.setdefault("cmap", cmap)
    if symmetric:
        data_max = 1.2 * np.nanmax(np.abs(u))
        kwargs.setdefault("vmin", -data_max)
        kwargs.setdefault("vmax", data_max)

    if "levels" in kwargs:
        levels = kwargs.pop("levels")
    else:
        vmin = kwargs.get("vmin", np.nanmin(u))
        vmax = kwargs.get("vmax", np.nanmax(u))
        levels = np.linspace(vmin, vmax, num_levels)

    u_plot = u.T

    im: Any
    if contour:
        kwargs.pop("vmin", None)
        kwargs.pop("vmax", None)
        im = ax.contourf(x_1d, y_1d, u_plot, levels=levels, **kwargs)
    else:
        kwargs.setdefault("shading", "auto")
        im = ax.pcolormesh(x_1d, y_1d, u_plot, **kwargs)

    if contour_lines:
        cl_kwargs = contour_lines_kwargs if contour_lines_kwargs is not None else {}
        cl_kwargs.setdefault("colors", "k")
        cl_kwargs.setdefault("linewidths", 0.5)
        ax.contour(x_1d, y_1d, u_plot, levels=levels, **cl_kwargs)

    if colorbar and fig:
        cb_opts = colorbar_kwargs or {}
        cb_opts.setdefault("orientation", "horizontal")
        cb_opts.setdefault("shrink", 0.7)
        cb_opts.setdefault("pad", 0.05)
        fig.colorbar(im, ax=ax, **cb_opts)

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_aspect("equal", adjustable="box")

    return ax, im


def plot_geodesic(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    /,
    *,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    if ax is None:
        ax = plt.gca()

    kwargs.setdefault("color", "black")
    kwargs.setdefault("linewidth", 2)

    dx = (p2[0] - p1[0] + np.pi) % (2 * np.pi) - np.pi
    dy = (p2[1] - p1[1] + np.pi) % (2 * np.pi) - np.pi

    t = np.linspace(0, 1, 500)
    x = p1[0] + t * dx
    y = p1[1] + t * dy

    x_wrapped = x % (2 * np.pi)
    y_wrapped = y % (2 * np.pi)

    jump_idx = np.where(
        (np.abs(np.diff(x_wrapped)) > np.pi) | (np.abs(np.diff(y_wrapped)) > np.pi)
    )[0]

    if len(jump_idx) > 0:
        x_plot = np.insert(x_wrapped, jump_idx + 1, np.nan)
        y_plot = np.insert(y_wrapped, jump_idx + 1, np.nan)
    else:
        x_plot = x_wrapped
        y_plot = y_wrapped

    ax.plot(x_plot, y_plot, **kwargs)

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_aspect("equal", adjustable="box")

    return ax


def plot_geodesic_network(
    paths: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    /,
    *,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    line_kwargs = kwargs.copy()
    src_style = line_kwargs.pop("source_kwargs", {})
    rec_style = line_kwargs.pop("receiver_kwargs", {})

    line_kwargs.setdefault("color", "black")
    line_kwargs.setdefault("linewidth", 0.8)
    line_kwargs.setdefault("alpha", 0.5)

    for p1, p2 in paths:
        plot_geodesic(p1, p2, ax=ax, **line_kwargs)

    sources = list(set([tuple(p[0]) for p in paths]))
    receivers = list(set([tuple(p[1]) for p in paths]))

    if sources:
        src_x, src_y = zip(*sources)
        src_x = np.array(src_x) % (2 * np.pi)
        src_y = np.array(src_y) % (2 * np.pi)
        src_style.setdefault("marker", "*")
        src_style.setdefault("color", "gold")
        src_style.setdefault("s", 150)
        src_style.setdefault("edgecolor", "black")
        src_style.setdefault("zorder", 5)
        ax.scatter(src_x, src_y, **src_style)

    if receivers:
        rec_x, rec_y = zip(*receivers)
        rec_x = np.array(rec_x) % (2 * np.pi)
        rec_y = np.array(rec_y) % (2 * np.pi)
        rec_style.setdefault("marker", "o")
        rec_style.setdefault("color", "red")
        rec_style.setdefault("s", 50)
        rec_style.setdefault("edgecolor", "white")
        rec_style.setdefault("zorder", 5)
        ax.scatter(rec_x, rec_y, **rec_style)

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_aspect("equal", adjustable="box")

    return ax
