"""The survey's forward map, ``m -> receiver data``, as a library operator.

``WaveformOperator`` wraps the C++ kernels of ``acoustic`` in an
:class:`~pygeoinf2.algebra.operators.Operator` built from callables: the value
is the receiver data of every shot, the derivative at ``m`` is a
:class:`~pygeoinf2.algebra.operators.LinearOperator` whose action is one
linearized run per shot and whose adjoint is one backward run per shot, and
the second derivative ``F''(m)[dm, .]`` is another, whose action runs three
recursions in lockstep and whose adjoint runs the linearized field forward and
two adjoint fields backward.

The adjoints are written as derivative components, the grid gradient that an
adjoint method produces, and the library applies the model space's inverse
metric to them (``from_derivative_callables``); the grid gradient of
``x -> g . x`` has components ``(cells / area) to_components(g)`` because the
box's components are orthonormal in ``L2`` and a grid sum is ``cells / area``
times an integral.

The forward fields of the last model asked about are cached, one per shot:
the derivative, its adjoint and the second derivative at that model all read
them, and a value at the same model returns the cached data. The fields stay
in C++, on whichever ranks hold the grid; the cache holds their handles and
releases them when the model changes, so it holds one model, and a
derivative held past a change of model recomputes its fields when next
applied. Applications at one model may run on several threads at once;
applications at two models at once may not, the second releasing the fields
the first is reading. Shots run on threads when ``n_jobs`` is set, the kernels releasing
the interpreter lock; the curvature, which needs a linearized field stored
beside each forward field, runs ``curvature_jobs`` shots at a time, so that at
most that many such fields exist at once. Across MPI ranks the kernels'
calls are serialized, and the shots run one after another, each on every
rank.

The file is not called ``operator.py`` because that name would shadow the
standard library's ``operator`` module for any script run from this
directory.
"""

from __future__ import annotations

import hashlib
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np

from pygeoinf2 import EvaluationError, LinearOperator, Operator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.symmetric_space.fourier import Sobolev

from survey import CFL_LIMIT, DEPTH_CELLS, Survey

__all__ = ["WaveformOperator"]


def _grid(x: Any) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=float)


def _digest(m: np.ndarray) -> bytes:
    return hashlib.blake2b(m.tobytes(), digest_size=16).digest()


class _Serial:
    """A stand-in pool that maps in the calling thread."""

    def map(self, work: Callable[[int], Any], items: Any) -> Any:
        return map(work, items)


_SERIAL = _Serial()


def _pool(jobs: int | None) -> ThreadPoolExecutor | None:
    if jobs is None or jobs <= 1:
        return None
    return ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="shot")


class WaveformOperator:
    """The forward map of a survey on a Sobolev box, with its two derivatives.

    Attributes:
        operator: the forward map, carrying its first and second derivatives.
        first_order: the same map without the second derivative, for which a
            Newton method gets the Gauss-Newton Hessian; it shares the cache.
        calls: how many times each kernel ran, counting one per shot.
    """

    def __init__(
        self,
        survey: Survey,
        space: Sobolev,
        /,
        *,
        n_jobs: int | None = None,
        curvature_jobs: int | None = None,
    ) -> None:
        """
        Args:
            survey: the grid, the shots and the receivers.
            space: the model space, a box over the survey's grid.
            n_jobs: threads over the shots. Serial if ``None`` or one.
            curvature_jobs: threads over the shots for the curvature term,
                each holding one linearized field while it runs. Serial if
                ``None`` or one.

        Raises:
            ValueError: if the space's grid is not the survey's.
        """
        if tuple(space.shape) != survey.shape:
            raise ValueError(f"The space's grid {space.shape} is not {survey.shape}.")
        self.survey = survey
        self.space = space
        self.data_space = EuclideanSpace(survey.data_size)
        self.calls: Counter[str] = Counter()
        self._kernels = survey.model
        self._scale = space.dim / space.volume
        self._lock = threading.Lock()
        self._key: bytes | None = None
        self._fields: list[int] | None = None
        self._data: np.ndarray | None = None
        self._pool = _pool(n_jobs)
        self._curvature_pool = _pool(curvature_jobs)
        self.operator = Operator.from_callables(
            space,
            self.data_space,
            self.value,
            derivative=self.derivative,
            second_derivative=self.second_derivative,
        )
        self.first_order = Operator.from_callables(
            space, self.data_space, self.value, derivative=self.derivative
        )

    # ----------------------------------------------------------------- #
    #                              Plumbing                             #
    # ----------------------------------------------------------------- #

    def _map(
        self, work: Callable[[int], Any], pool: ThreadPoolExecutor | None = None
    ) -> list[Any]:
        shots = range(self.survey.n_shots)
        pool = self._pool if pool is None else pool
        if pool is None:
            return [work(shot) for shot in shots]
        return list(pool.map(work, shots))

    def _shot_traces(self, y: Any) -> list[np.ndarray]:
        traces = np.asarray(y, dtype=float).reshape(self.survey.data_shape)
        return [np.ascontiguousarray(trace) for trace in traces]

    def admissible(self, m: Any) -> np.ndarray:
        """The model as a grid, if the scheme is stable on it.

        Raises:
            EvaluationError: for a model with a non-positive or non-finite
                value, or fast enough somewhere to break the stability limit;
                a line search retreats from such a point.
        """
        m = _grid(m)
        if not np.all(np.isfinite(m)) or np.min(m) <= 0.0:
            raise EvaluationError("The model is not positive everywhere.")
        if self.survey.cfl(m) > 0.98 * CFL_LIMIT:
            raise EvaluationError(
                f"The model breaks the stability limit: c dt / h = "
                f"{self.survey.cfl(m):.3f} against {CFL_LIMIT:.3f}."
            )
        return m

    def fields(self, m: Any) -> tuple[list[int], np.ndarray]:
        """The handle of the forward field of every shot at ``m``, and the data; cached."""
        m = self.admissible(m)
        key = _digest(m)
        with self._lock:
            if key != self._key:
                # Drop the old fields before making the new ones.
                self._release_cache()
                wavelet = np.ascontiguousarray(self.survey.wavelet[:, None])

                def run(shot: int) -> tuple[int, np.ndarray]:
                    return self._kernels.forward(
                        m, [int(self.survey.source_x[shot])], [DEPTH_CELLS], wavelet
                    )

                results = self._map(run)
                self.calls["forward"] += len(results)
                self._fields = [field for field, _ in results]
                self._data = np.stack([data for _, data in results])
                self._key = key
            return self._fields, self._data

    def _release_cache(self) -> None:
        for handle in self._fields or ():
            self._kernels.release(handle)
        self._key = self._fields = self._data = None

    def clear(self) -> None:
        """Release the cached fields."""
        with self._lock:
            self._release_cache()

    def representer(self, grid_gradient: np.ndarray) -> np.ndarray:
        """The model-space vector pairing with ``x`` as ``g . x`` does."""
        return self.space.representer(
            self._scale * self.space.to_components(grid_gradient)
        )

    # ----------------------------------------------------------------- #
    #                     The operator and its derivatives               #
    # ----------------------------------------------------------------- #

    def value(self, m: Any) -> np.ndarray:
        """The receiver data of every shot, flattened shot by step by receiver."""
        return self.fields(m)[1].ravel().copy()

    def derivative(self, m: Any) -> LinearOperator:
        """``F'(m)``: one linearized run per shot, and one adjoint run per shot."""
        point = self.admissible(m).copy()
        kernels = self._kernels

        def value(dm: Any) -> np.ndarray:
            fields = self.fields(point)[0]
            dm = _grid(dm)
            data = self._map(
                lambda s: kernels.born(point, fields[s], dm, keep=False)[1]
            )
            self.calls["born"] += len(data)
            return np.stack(data).ravel()

        def derivative_components(r: Any) -> np.ndarray:
            fields = self.fields(point)[0]
            traces = self._shot_traces(r)
            parts = self._map(lambda s: kernels.gradient(point, fields[s], traces[s]))
            self.calls["adjoint"] += len(parts)
            return self._scale * self.space.to_components(np.sum(parts, axis=0))

        return LinearOperator.from_derivative_callables(
            self.space, self.data_space, value, derivative_components
        )

    def second_derivative(self, m: Any, dm: Any) -> LinearOperator:
        """``F''(m)[dm, .]``, from the model space to the data space.

        Its action runs the two linearized fields and the second-order one in
        lockstep per shot; its adjoint stores the linearized field along
        ``dm`` and makes one backward pass carrying both adjoint fields.
        """
        point = self.admissible(m).copy()
        direction = _grid(dm).copy()
        kernels = self._kernels

        def value(dm2: Any) -> np.ndarray:
            fields = self.fields(point)[0]
            dm2 = _grid(dm2)
            data = self._map(
                lambda s: kernels.second_born(point, fields[s], direction, dm2)
            )
            self.calls["second_born"] += len(data)
            return np.stack(data).ravel()

        def derivative_components(w: Any) -> np.ndarray:
            fields = self.fields(point)[0]
            traces = self._shot_traces(w)

            def shot(s: int) -> np.ndarray:
                linearized, _ = kernels.born(point, fields[s], direction, keep=True)
                return kernels.curvature(
                    point, fields[s], linearized, traces[s], direction
                )

            parts = self._map(shot, self._curvature_pool or _SERIAL)
            self.calls["born"] += len(parts)
            self.calls["curvature"] += len(parts)
            return self._scale * self.space.to_components(np.sum(parts, axis=0))

        return LinearOperator.from_derivative_callables(
            self.space, self.data_space, value, derivative_components
        )

    # ----------------------------------------------------------------- #
    #                 The same, from the unfused kernels                 #
    # ----------------------------------------------------------------- #
    #
    # The four kernels the derivation names -- forward, born, adjoint and
    # correlate, with adjoint_born for the second adjoint -- composed as the
    # derivation composes them, storing every field. The operator above uses
    # the fused kernels; these are what they are checked against.

    def unfused_gradient(self, m: Any, r: Any) -> np.ndarray:
        """``-(dA[.] U)^T L`` per shot, ``A^T L = S^T r``: the grid gradient."""
        point = self.admissible(m)
        fields = self.fields(point)[0]
        traces = self._shot_traces(r)
        kernels = self._kernels
        total = np.zeros(self.survey.shape)
        for s in range(self.survey.n_shots):
            lam = kernels.adjoint(point, traces[s])
            total += kernels.correlate(fields[s], lam)
            kernels.release(lam)
        return total

    def unfused_second_born(self, m: Any, dm: Any, dm2: Any) -> np.ndarray:
        """``-S A^-1 (dA[dm] dU2 + dA[dm2] dU)`` per shot, from stored fields."""
        point = self.admissible(m)
        fields = self.fields(point)[0]
        dm, dm2 = _grid(dm), _grid(dm2)
        kernels = self._kernels
        data = []
        for s in range(self.survey.n_shots):
            du = kernels.born(point, fields[s], dm)[0]
            du2 = kernels.born(point, fields[s], dm2)[0]
            data.append(
                kernels.born(point, du2, dm, keep=False)[1]
                + kernels.born(point, du, dm2, keep=False)[1]
            )
            kernels.release(du)
            kernels.release(du2)
        return np.stack(data).ravel()

    def unfused_curvature(self, m: Any, dm: Any, w: Any) -> np.ndarray:
        """``correlate(dU, L_w) - correlate(U, M_w)`` per shot, the grid gradient.

        ``A^T L_w = S^T w`` and ``A^T M_w = dA[dm]^T L_w``.
        """
        point = self.admissible(m)
        fields = self.fields(point)[0]
        dm = _grid(dm)
        traces = self._shot_traces(w)
        kernels = self._kernels
        total = np.zeros(self.survey.shape)
        for s in range(self.survey.n_shots):
            du = kernels.born(point, fields[s], dm)[0]
            lam = kernels.adjoint(point, traces[s])
            second = kernels.adjoint_born(point, lam, dm)
            total += kernels.correlate(du, lam) - kernels.correlate(fields[s], second)
            for handle in (du, lam, second):
                kernels.release(handle)
        return total
