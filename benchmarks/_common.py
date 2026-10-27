"""What the three feasible-set benchmarks share: counting, timing, tables.

Each benchmark builds its problem in a function that the slow tests also
call at a reduced size, runs the algorithms it compares, and prints one
markdown table per question, which is what the appendix of
``pygeoinf2/DECISIONS.md`` records.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from pygeoinf2.algebra.operators import LinearOperator


@dataclass
class Counter:
    """How many times a forward operator and its adjoint were applied."""

    forward: int = 0
    adjoint: int = 0
    history: list[tuple[int, int]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.forward + self.adjoint

    def reset(self) -> None:
        self.forward = self.adjoint = 0

    def mark(self) -> tuple[int, int]:
        """The counts since the last mark, which is reset."""
        taken = (self.forward, self.adjoint)
        self.history.append(taken)
        self.reset()
        return taken


def counted(operator: LinearOperator, /) -> tuple[LinearOperator, Counter]:
    """The same operator, counting applications of itself and its adjoint.

    Matrix-free: what is handed back applies the operator given, so a dense
    operator stays dense underneath and the count is of applications, not
    of the arithmetic inside one.
    """
    counter = Counter()

    def value(x: Any) -> Any:
        counter.forward += 1
        return operator(x)

    def adjoint(y: Any) -> Any:
        counter.adjoint += 1
        return operator.adjoint(y)

    wrapped = LinearOperator.from_callables(
        operator.domain,
        operator.codomain,
        value,
        adjoint=adjoint,
        traits=operator.traits,
    )
    return wrapped, counter


def timed(function: Callable[[], Any], /) -> tuple[Any, float]:
    """The function's result and its wall time in seconds."""
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], /) -> str:
    """A markdown table, numbers formatted compactly."""

    def cell(value: Any) -> str:
        if isinstance(value, float):
            if value == 0.0:
                return "0"
            if abs(value) >= 1e4 or abs(value) < 1e-3:
                return f"{value:.3g}"
            return f"{value:.4g}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def axis_directions(space: Any, /) -> list[Any]:
    """``±e_i`` for every axis of a coordinate space."""
    return [
        space.scale(sign, space.basis_vector(index))
        for index in range(space.dim)
        for sign in (1.0, -1.0)
    ]


def planar_directions(space: Any, first: int, second: int, count: int, /) -> list[Any]:
    """``count`` unit directions in the plane of two axes, equally spaced in angle."""
    directions = []
    for angle in np.linspace(0.0, 2.0 * np.pi, count, endpoint=False):
        components = np.zeros(space.dim)
        components[first], components[second] = np.cos(angle), np.sin(angle)
        directions.append(space.from_components(components))
    return directions


def parser(description: str, /) -> argparse.ArgumentParser:
    """The argument parser every benchmark starts from."""
    parsed = argparse.ArgumentParser(description=description)
    parsed.add_argument("--seed", type=int, default=0, help="the random seed")
    parsed.add_argument(
        "--n-jobs", type=int, default=4, help="workers for the parallel sweeps"
    )
    return parsed
