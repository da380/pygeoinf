"""Line integral linear form helper inside the pygeoinf package.

Provides `LineIntegralForm` which is a lightweight wrapper around
`pygeoinf.linear_forms.LinearForm` to compute line integrals of a field
represented by domain vectors along a piecewise-linear path.
"""
from __future__ import annotations

from typing import Callable, Optional, Any

import numpy as np

from .linear_forms import LinearForm


class LineIntegralForm(LinearForm):
    """Linear form computing the line integral of a field along a path.

    See `masters/path_integral/path_form.py` for a fuller description.
    """

    def __init__(
        self,
        domain,
        path_points: np.ndarray,
        sampler: Optional[Callable[[Any, np.ndarray], np.ndarray]] = None,
        *,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> None:
        path = np.asarray(path_points, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError("path_points must have shape (N, 3)")

        self._path = path
        self._sampler = sampler

        def mapping(x: Any) -> float:
            vals = self._sample_at_points(x, self._path)
            seg = np.linalg.norm(np.diff(self._path, axis=0), axis=1)
            if len(seg) == 0:
                return 0.0
            return 0.5 * np.sum((vals[:-1] + vals[1:]) * seg)

        super().__init__(domain, mapping=mapping, parallel=parallel, n_jobs=n_jobs)

    def _sample_at_points(self, x: Any, points: np.ndarray) -> np.ndarray:
        if self._sampler is not None:
            return np.asarray(self._sampler(x, points))

        if callable(x):
            return np.asarray([float(x(tuple(p))) for p in points])

        for name in ("evaluate", "eval", "evaluate_at", "sample_at"):
            if hasattr(x, name):
                fn = getattr(x, name)
                return np.asarray(fn(points))

        raise TypeError(
            "Cannot sample vector at points: provide callable vectors or a sampler"
        )
