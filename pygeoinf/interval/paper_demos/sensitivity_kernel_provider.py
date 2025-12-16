"""
Sensitivity Kernel Provider

Provider to expose real sensitivity kernels as Function coefficients
usable by the rest of pygeoinf.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from .sensitivity_kernel_catalog import SensitivityKernelCatalog
from .kernel_interpolator import KernelInterpolator
from .discontinuity_kernels import DiscontinuityKernel
from pygeoinf.interval.function_providers.base import IndexedFunctionProvider


class SensitivityKernelProvider(IndexedFunctionProvider):
    def __init__(self,
                 lebesgue_space,
                 catalog: SensitivityKernelCatalog,
                 interpolation_method: str = 'cubic',
                 include_discontinuities: bool = True,
                 cache_kernels: bool = True,
                 kernel_type: str = 'vp'):
        super().__init__(lebesgue_space)
        self.catalog = catalog
        self.method = interpolation_method
        self.include_discontinuities = include_discontinuities
        self.cache_kernels = cache_kernels
        self.kernel_type = kernel_type
        self.mode_ids = catalog.list_modes()

        if interpolation_method not in ['linear', 'cubic', 'spline']:
            raise ValueError("Unknown interpolation method")
        if kernel_type not in ['vp', 'vs', 'rho']:
            raise ValueError(f"Unknown kernel type: {kernel_type}. Must be 'vp', 'vs', or 'rho'.")

        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def _collapse_duplicates(self, depths: np.ndarray, values: np.ndarray):
        """Collapse duplicate depth entries by averaging their values.

        Returns arrays (depths_unique, values_avg) sorted by depth.
        """
        if len(depths) == 0:
            return depths, values

        # Round depths to a reasonable precision to avoid tiny floating differences
        # that should be considered identical in these data files.
        depths_arr = np.asarray(depths)
        values_arr = np.asarray(values)
        # Use string conversion grouping to preserve exact duplicates when present
        uniq_depths, inv = np.unique(depths_arr, return_inverse=True)
        summed = np.zeros_like(uniq_depths, dtype=float)
        counts = np.zeros_like(uniq_depths, dtype=int)
        for idx, v in zip(inv, values_arr):
            summed[idx] += v
            counts[idx] += 1
        avg = summed / counts
        return uniq_depths, avg

    def _get_cached_kernel(self, mode_id: str, param: str) -> Optional[np.ndarray]:
        if not self.cache_kernels:
            return None
        if mode_id in self._cache and param in self._cache[mode_id]:
            return self._cache[mode_id][param]
        return None

    def _cache_kernel(self, mode_id: str, param: str, kernel: np.ndarray):
        if not self.cache_kernels:
            return
        if mode_id not in self._cache:
            self._cache[mode_id] = {}
        self._cache[mode_id][param] = kernel

    def get_vp_kernel(self, mode_id: str):
        """Get vp sensitivity kernel as a Function object (not basis coefficients)."""
        cached = self._get_cached_kernel(mode_id, 'vp')
        if cached is not None:
            return cached

        data = self.catalog.get_mode(mode_id)
        depths_u, values_u = self._collapse_duplicates(data.vp_depths, data.vp_values)
        interpolator = KernelInterpolator(depths_u, values_u, method=self.method)
        kernel = interpolator.to_function(self.space)
        self._cache_kernel(mode_id, 'vp', kernel)
        return kernel

    def get_vs_kernel(self, mode_id: str):
        """Get vs sensitivity kernel as a Function object (not basis coefficients)."""
        cached = self._get_cached_kernel(mode_id, 'vs')
        if cached is not None:
            return cached
        data = self.catalog.get_mode(mode_id)
        depths_u, values_u = self._collapse_duplicates(data.vs_depths, data.vs_values)
        interpolator = KernelInterpolator(depths_u, values_u, method=self.method)
        kernel = interpolator.to_function(self.space)
        self._cache_kernel(mode_id, 'vs', kernel)
        return kernel

    def get_rho_kernel(self, mode_id: str):
        """Get rho sensitivity kernel as a Function object (not basis coefficients)."""
        cached = self._get_cached_kernel(mode_id, 'rho')
        if cached is not None:
            return cached
        data = self.catalog.get_mode(mode_id)
        depths_u, values_u = self._collapse_duplicates(data.rho_depths, data.rho_values)
        interpolator = KernelInterpolator(depths_u, values_u, method=self.method)
        kernel = interpolator.to_function(self.space)
        self._cache_kernel(mode_id, 'rho', kernel)
        return kernel

    def get_topo_kernel(self, mode_id: str) -> DiscontinuityKernel:
        cached = self._get_cached_kernel(mode_id, 'topo')
        if cached is not None:
            data = self.catalog.get_mode(mode_id)
            return DiscontinuityKernel(data.topo_depths, cached)

        data = self.catalog.get_mode(mode_id)
        kernel = DiscontinuityKernel(data.topo_depths, data.topo_values)
        self._cache_kernel(mode_id, 'topo', data.topo_values)
        return kernel

    def get_all_kernels(self, mode_id: str) -> Dict[str, any]:
        kernels = {
            'vp': self.get_vp_kernel(mode_id),
            'vs': self.get_vs_kernel(mode_id),
            'rho': self.get_rho_kernel(mode_id),
        }
        if self.include_discontinuities:
            kernels['topo'] = self.get_topo_kernel(mode_id)
        return kernels

    def get_mode_metadata(self, mode_id: str) -> Dict[str, any]:
        data = self.catalog.get_mode(mode_id)
        return {
            'mode_id': data.mode_id,
            'n': data.n,
            'l': data.l,
            'period': data.period,
            'frequency': 1.0 / data.period if data.period else None,
            'vp_ref': data.vp_ref,
            'vs_ref': data.vs_ref,
            'group_velocity': data.group_velocity,
        }

    def create_forward_operator(self, mode_ids: List[str], model_components: Optional[List[str]] = None):
        raise NotImplementedError("Forward operator construction requires integration with pygeoinf.linear_operators")

    def compute_data_kernel_matrix(self, mode_ids: List[str]) -> Dict[str, np.ndarray]:
        n_modes = len(mode_ids)
        vp_kernels = [self.get_vp_kernel(mid) for mid in mode_ids]
        vs_kernels = [self.get_vs_kernel(mid) for mid in mode_ids]
        rho_kernels = [self.get_rho_kernel(mid) for mid in mode_ids]

        matrices = {}
        for param, kernels in [('vp', vp_kernels), ('vs', vs_kernels), ('rho', rho_kernels)]:
            matrix = np.zeros((n_modes, n_modes))
            for i in range(n_modes):
                for j in range(i, n_modes):
                    inner_prod = self.space.inner_product(kernels[i], kernels[j])
                    matrix[i, j] = inner_prod
                    matrix[j, i] = inner_prod
            matrices[param] = matrix

        return matrices

    def plot_kernel(self, mode_id: str, param: str = 'all', ax=None, **kwargs):
        import matplotlib.pyplot as plt
        from .depth_coordinates import DepthCoordinateSystem

        if param == 'all':
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            for i, p in enumerate(['vp', 'vs', 'rho']):
                self._plot_single_kernel(mode_id, p, axes[i], **kwargs)

            if self.include_discontinuities:
                topo_kernel = self.get_topo_kernel(mode_id)
                topo_kernel.plot(ax=axes[3])
            else:
                axes[3].text(0.5, 0.5, 'Discontinuities not included', ha='center', va='center', transform=axes[3].transAxes)

            metadata = self.get_mode_metadata(mode_id)
            fig.suptitle(f"Mode {mode_id}: n={metadata['n']}, l={metadata['l']}, T={metadata['period']:.1f}s", fontsize=14, fontweight='bold')
            plt.tight_layout()
            return fig
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6))
            if param == 'topo':
                topo_kernel = self.get_topo_kernel(mode_id)
                topo_kernel.plot(ax=ax, **kwargs)
            else:
                self._plot_single_kernel(mode_id, param, ax, **kwargs)
            return ax

    def _plot_single_kernel(self, mode_id: str, param: str, ax, **kwargs):
        from .depth_coordinates import DepthCoordinateSystem
        if param == 'vp':
            kernel = self.get_vp_kernel(mode_id)
        elif param == 'vs':
            kernel = self.get_vs_kernel(mode_id)
        elif param == 'rho':
            kernel = self.get_rho_kernel(mode_id)
        else:
            raise ValueError(f"Unknown parameter: {param}")

        x_norm = np.linspace(0, 1, 500)
        kernel_func = self.space.create_function(kernel)
        y = kernel_func(x_norm)
        x_depth = DepthCoordinateSystem.normalized_to_depth(x_norm)
        ax.plot(x_depth, y, label=param, **kwargs)
        ax.set_xlabel('Depth [km]')
        ax.set_ylabel(f'{param} Sensitivity')
        ax.set_title(f'{param.upper()} Kernel - Mode {mode_id}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    def clear_cache(self):
        self._cache.clear()
        if hasattr(self.catalog, 'clear_cache'):
            self.catalog.clear_cache()

    def get_cache_info(self) -> Dict[str, any]:
        total_kernels = sum(len(params) for params in self._cache.values())
        return {
            'n_modes_cached': len(self._cache),
            'n_kernels_cached': total_kernels,
            'cached_modes': list(self._cache.keys()),
        }

    def get_function_by_index(self, index: int):
        """Get the kernel function for the i-th mode (implements IndexedFunctionProvider)."""
        if index < 0 or index >= len(self.mode_ids):
            raise IndexError(f"Kernel index {index} out of range [0, {len(self.mode_ids)})")

        mode_id = self.mode_ids[index]

        # Get the kernel Function directly based on kernel_type
        if self.kernel_type == 'vp':
            return self.get_vp_kernel(mode_id)
        elif self.kernel_type == 'vs':
            return self.get_vs_kernel(mode_id)
        elif self.kernel_type == 'rho':
            return self.get_rho_kernel(mode_id)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def __len__(self):
        """Return the number of available modes."""
        return len(self.mode_ids)

    def __repr__(self) -> str:
        cache_info = self.get_cache_info()
        return (
            f"SensitivityKernelProvider(n_modes_available={len(self.catalog)}, "
            f"n_modes_cached={cache_info['n_modes_cached']}, method='{self.method}', "
            f"kernel_type='{self.kernel_type}', "
            f"include_discontinuities={self.include_discontinuities})"
        )
