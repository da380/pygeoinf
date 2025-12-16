"""
Sensitivity Kernel Data Loader

This module provides functionality for loading real sensitivity kernel data
from disk. The data comes from normal mode calculations and contains discrete
sensitivity kernels for vp, vs, rho (volumetric parameters) and topography
(discontinuity parameters).

Classes:
    SensitivityKernelData: Container for loaded kernel data for a single mode

Functions:
    parse_mode_id: Convert mode string "00s03" to (n, l) tuple
    format_mode_id: Convert (n, l) tuple to mode string "00s03"
    load_kernel_file: Parse 2-column kernel data file
    parse_header: Extract metadata from combined kernel file headers
"""

from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import re


def parse_mode_id(mode_id: str) -> Tuple[int, int]:
    pattern = r'^(\d{2})s(\d{2})$'
    match = re.match(pattern, mode_id)
    if not match:
        raise ValueError(f"Invalid mode_id format: {mode_id}. Expected format: 'NNsLL'")

    n = int(match.group(1))
    l = int(match.group(2))
    return n, l


def format_mode_id(n: int, l: int) -> str:
    if not (0 <= n <= 99):
        raise ValueError(f"Overtone n={n} out of range [0, 99]")
    if not (0 <= l <= 99):
        raise ValueError(f"Angular order l={l} out of range [0, 99]")

    return f"{n:02d}s{l:02d}"


def load_kernel_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not filepath.exists():
        raise FileNotFoundError(f"Kernel file not found: {filepath}")

    depths = []
    values = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid format at line {line_num} in {filepath.name}: "
                    f"expected 2 columns, got {len(parts)}"
                )

            try:
                depth = float(parts[0])
                value = float(parts[1])
                depths.append(depth)
                values.append(value)
            except ValueError as e:
                raise ValueError(
                    f"Invalid number format at line {line_num} in {filepath.name}: {e}"
                )

    if len(depths) == 0:
        raise ValueError(f"No data found in {filepath.name}")

    return np.array(depths), np.array(values)


def parse_header(filepath: Path) -> Dict[str, float]:
    if not filepath.exists():
        raise FileNotFoundError(f"Kernel file not found: {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError(f"File too short to contain header: {filepath.name}")

    line1 = lines[0].strip()
    if not line1.startswith('#'):
        raise ValueError(f"Expected header line starting with # in {filepath.name}")

    parts1 = line1[1:].split()
    if len(parts1) < 3:
        raise ValueError(
            f"Invalid header format in {filepath.name}: expected at least 3 values, got {len(parts1)}"
        )

    try:
        period = float(parts1[0])
        vp_ref = float(parts1[1])
        vs_ref = float(parts1[2])
    except ValueError as e:
        raise ValueError(f"Invalid number in header of {filepath.name}: {e}")

    line2 = lines[1].strip()
    if not line2.startswith('#'):
        raise ValueError(f"Expected second header line starting with # in {filepath.name}")

    parts2 = line2[1:].split()
    if len(parts2) < 2:
        raise ValueError(
            f"Invalid second header line in {filepath.name}: expected at least 2 values, got {len(parts2)}"
        )

    try:
        group_velocity = float(parts2[1])
    except ValueError as e:
        raise ValueError(f"Invalid group velocity in header of {filepath.name}: {e}")

    return {
        'period': period,
        'vp_ref': vp_ref,
        'vs_ref': vs_ref,
        'group_velocity': group_velocity
    }


class SensitivityKernelData:
    def __init__(self, mode_id: str, data_dir: Path):
        self.mode_id = mode_id
        self.n, self.l = parse_mode_id(mode_id)
        self.data_dir = Path(data_dir)

        try:
            self._load_from_combined()
        except FileNotFoundError:
            self._load_from_individual()
            self.period = None
            self.vp_ref = None
            self.vs_ref = None
            self.group_velocity = None

    def _load_from_combined(self):
        sens_kernels_file = self.data_dir / f"sens_kernels_{self.mode_id}_iso.dat"
        if not sens_kernels_file.exists():
            raise FileNotFoundError(f"Combined file not found: {sens_kernels_file}")

        header = parse_header(sens_kernels_file)
        self.period = header['period']
        self.vp_ref = header['vp_ref']
        self.vs_ref = header['vs_ref']
        self.group_velocity = header['group_velocity']

        data = []
        with open(sens_kernels_file, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    parts = line.split()
                    if len(parts) == 4:
                        data.append([float(x) for x in parts])

        data = np.array(data)
        depths = data[:, 0]

        self.vp_depths = depths.copy()
        self.vp_values = data[:, 1]
        self.vs_depths = depths.copy()
        self.vs_values = data[:, 2]
        self.rho_depths = depths.copy()
        self.rho_values = data[:, 3]

        topo_kernels_file = self.data_dir / f"topo_kernels_{self.mode_id}_iso.dat"
        if topo_kernels_file.exists():
            topo_data = []
            with open(topo_kernels_file, 'r') as f:
                for line in f:
                    if not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            topo_data.append([float(parts[0]), float(parts[1])])

            if topo_data:
                topo_data = np.array(topo_data)
                self.topo_depths = topo_data[:, 0]
                self.topo_values = topo_data[:, 1]
            else:
                self.topo_depths = np.array([])
                self.topo_values = np.array([])
        else:
            self.topo_depths = np.array([])
            self.topo_values = np.array([])

    def _load_from_individual(self):
        vp_file = self.data_dir / f"vp-sens_{self.mode_id}_iso.dat"
        if not vp_file.exists():
            raise FileNotFoundError(f"vp kernel file not found: {vp_file}")
        self.vp_depths, self.vp_values = load_kernel_file(vp_file)

        vs_file = self.data_dir / f"vs-sens_{self.mode_id}_iso.dat"
        if not vs_file.exists():
            raise FileNotFoundError(f"vs kernel file not found: {vs_file}")
        self.vs_depths, self.vs_values = load_kernel_file(vs_file)

        rho_file = self.data_dir / f"rho-sens_{self.mode_id}_iso.dat"
        if not rho_file.exists():
            raise FileNotFoundError(f"rho kernel file not found: {rho_file}")
        self.rho_depths, self.rho_values = load_kernel_file(rho_file)

        topo_file = self.data_dir / f"topo-sens_{self.mode_id}_iso.dat"
        if topo_file.exists():
            self.topo_depths, self.topo_values = load_kernel_file(topo_file)
        else:
            self.topo_depths = np.array([])
            self.topo_values = np.array([])

    @classmethod
    def from_combined_file(cls, mode_id: str, data_dir: Path) -> 'SensitivityKernelData':
        instance = cls.__new__(cls)
        instance.mode_id = mode_id
        instance.n, instance.l = parse_mode_id(mode_id)
        instance.data_dir = Path(data_dir)
        instance._load_from_combined()
        return instance

    @classmethod
    def from_individual_files(cls, mode_id: str, data_dir: Path) -> 'SensitivityKernelData':
        instance = cls.__new__(cls)
        instance.mode_id = mode_id
        instance.n, instance.l = parse_mode_id(mode_id)
        instance.data_dir = Path(data_dir)
        instance._load_from_individual()
        instance.period = None
        instance.vp_ref = None
        instance.vs_ref = None
        instance.group_velocity = None
        return instance

    def __repr__(self) -> str:
        period_str = f"{self.period:.1f}s" if self.period else "unknown"
        return (
            f"SensitivityKernelData(mode={self.mode_id}, n={self.n}, l={self.l}, "
            f"period={period_str}, n_volumetric={len(self.vp_depths)}, "
            f"n_discontinuities={len(self.topo_depths)})"
        )

    def get_kernel_summary(self) -> Dict[str, any]:
        summary = {
            'mode_id': self.mode_id,
            'n': self.n,
            'l': self.l,
            'period': self.period,
            'vp_ref': self.vp_ref,
            'vs_ref': self.vs_ref,
            'group_velocity': self.group_velocity,
            'vp_kernel': {
                'n_points': len(self.vp_depths),
                'depth_range': (self.vp_depths.min(), self.vp_depths.max()),
                'value_range': (self.vp_values.min(), self.vp_values.max()),
                'mean': self.vp_values.mean(),
                'std': self.vp_values.std(),
            },
            'vs_kernel': {
                'n_points': len(self.vs_depths),
                'depth_range': (self.vs_depths.min(), self.vs_depths.max()),
                'value_range': (self.vs_values.min(), self.vs_values.max()),
                'mean': self.vs_values.mean(),
                'std': self.vs_values.std(),
            },
            'rho_kernel': {
                'n_points': len(self.rho_depths),
                'depth_range': (self.rho_depths.min(), self.rho_depths.max()),
                'value_range': (self.rho_values.min(), self.rho_values.max()),
                'mean': self.rho_values.mean(),
                'std': self.rho_values.std(),
            },
            'topo_kernel': {
                'n_points': len(self.topo_depths),
                'depth_range': (self.topo_depths.min(), self.topo_depths.max()) if len(self.topo_depths) > 0 else None,
                'value_range': (self.topo_values.min(), self.topo_values.max()) if len(self.topo_values) > 0 else None,
            }
        }
        return summary
