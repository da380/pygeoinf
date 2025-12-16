"""
Sensitivity Kernel Catalog

Catalog system for managing collections of sensitivity kernel data.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from .sensitivity_kernel_loader import (
    SensitivityKernelData,
    parse_mode_id,
    format_mode_id
)


class SensitivityKernelCatalog:
    def __init__(self, data_dir: Path, mode_list_file: Optional[str] = "data_list_SP12RTS"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        if mode_list_file:
            mode_list_path = self.data_dir / mode_list_file
            if mode_list_path.exists():
                self.available_modes = self._load_mode_list(mode_list_path)
            else:
                self.available_modes = self._scan_directory()
        else:
            self.available_modes = self._scan_directory()

        prem_file = self.data_dir / "PREM_depth_layers_all"
        if prem_file.exists():
            self.prem_depths = self._load_prem_depths(prem_file)
        else:
            self.prem_depths = None

        self._cache: Dict[str, SensitivityKernelData] = {}

    def _load_mode_list(self, filepath: Path) -> List[str]:
        modes = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    modes.append(line)
        return sorted(modes)

    def _scan_directory(self) -> List[str]:
        modes = set()

        for filepath in self.data_dir.glob("sens_kernels_*_iso.dat"):
            filename = filepath.stem
            parts = filename.split('_')
            if len(parts) >= 2:
                mode_id = parts[2]
                modes.add(mode_id)

        if not modes:
            for filepath in self.data_dir.glob("vp-sens_*_iso.dat"):
                filename = filepath.stem
                parts = filename.split('_')
                if len(parts) >= 2:
                    mode_id = parts[1]
                    modes.add(mode_id)

        return sorted(list(modes))

    def _load_prem_depths(self, filepath: Path) -> np.ndarray:
        depths = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        radius = float(parts[1])
                        depth = 6371.0 - radius
                        depths.append(depth)
        return np.array(depths)

    def get_mode(self, mode_id: str) -> SensitivityKernelData:
        if mode_id not in self.available_modes:
            raise ValueError(f"Mode {mode_id} not found in catalog. Available modes: {len(self.available_modes)}")

        if mode_id not in self._cache:
            self._cache[mode_id] = SensitivityKernelData(mode_id, self.data_dir)

        return self._cache[mode_id]

    def get_mode_by_nl(self, n: int, l: int) -> SensitivityKernelData:
        mode_id = format_mode_id(n, l)
        return self.get_mode(mode_id)

    def list_modes(self,
                   n_min: Optional[int] = None,
                   n_max: Optional[int] = None,
                   l_min: Optional[int] = None,
                   l_max: Optional[int] = None) -> List[str]:
        filtered = []
        for mode_id in self.available_modes:
            n, l = parse_mode_id(mode_id)
            if n_min is not None and n < n_min:
                continue
            if n_max is not None and n > n_max:
                continue
            if l_min is not None and l < l_min:
                continue
            if l_max is not None and l > l_max:
                continue
            filtered.append(mode_id)
        return filtered

    def get_frequency_range(self) -> Tuple[float, float]:
        periods = []
        for mode_id in self.available_modes:
            mode = self.get_mode(mode_id)
            if mode.period is not None:
                periods.append(mode.period)

        if not periods:
            raise ValueError("No period information available in catalog")

        periods = np.array(periods)
        frequencies = 1.0 / periods
        return frequencies.min(), frequencies.max()

    def get_period_range(self) -> Tuple[float, float]:
        periods = []
        for mode_id in self.available_modes:
            mode = self.get_mode(mode_id)
            if mode.period is not None:
                periods.append(mode.period)

        if not periods:
            raise ValueError("No period information available in catalog")

        periods = np.array(periods)
        return periods.min(), periods.max()

    def find_modes_by_period(self,
                            period_min: float,
                            period_max: float) -> List[str]:
        matching_modes = []
        for mode_id in self.available_modes:
            mode = self.get_mode(mode_id)
            if mode.period is not None:
                if period_min <= mode.period <= period_max:
                    matching_modes.append(mode_id)
        return matching_modes

    def clear_cache(self):
        self._cache.clear()

    def preload_modes(self, mode_ids: Optional[List[str]] = None):
        if mode_ids is None:
            mode_ids = self.available_modes

        for mode_id in mode_ids:
            self.get_mode(mode_id)

    def get_catalog_summary(self) -> Dict[str, any]:
        n_values = []
        l_values = []

        for mode_id in self.available_modes:
            n, l = parse_mode_id(mode_id)
            n_values.append(n)
            l_values.append(l)

        n_values = np.array(n_values)
        l_values = np.array(l_values)

        summary = {
            'n_modes': len(self.available_modes),
            'n_range': (n_values.min(), n_values.max()),
            'l_range': (l_values.min(), l_values.max()),
            'n_unique': len(np.unique(n_values)),
            'l_unique': len(np.unique(l_values)),
            'cached_modes': len(self._cache),
            'has_prem_depths': self.prem_depths is not None,
        }

        try:
            T_min, T_max = self.get_period_range()
            summary['period_range'] = (T_min, T_max)
        except (ValueError, AttributeError):
            summary['period_range'] = None

        return summary

    def __repr__(self) -> str:
        return (
            f"SensitivityKernelCatalog(n_modes={len(self.available_modes)}, "
            f"cached={len(self._cache)}, data_dir='{self.data_dir.name}')"
        )

    def __len__(self) -> int:
        return len(self.available_modes)
