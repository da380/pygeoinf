"""
Measures that know their density relative to a reference measure.

The dimension-robust notion of a density. A density with respect to a
discretised space's own volume measure is an artefact of the truncation,
which is why ``log_density`` and its relatives refuse on a space that
discretises functions; a density with respect to a *reference measure* -- a
prior, most of all -- survives refinement, and it is the object every
function-space sampler is built on. An inference result that carries its
reference is one a sampler can take whole: the measure to draw from, and the
log density that weights the draws, arrive together and cannot be mispaired.

The convention, fixed here once: with ``f`` the :attr:`relative_log_density`
and ``c`` the :attr:`log_normalizer`,

.. code-block:: text

    d mu / d nu (x) == exp(f(x) - c),    c == log integral exp(f) d nu

so ``f`` alone is the density *up to a constant* -- which is all a
Metropolis ratio needs -- and ``c``, when it is known, makes it exact and is
the evidence of a Bayesian update. A measure is built from a reference and a
density, never from two arbitrary measures, because the density between two
unrelated measures on a function space generally does not exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from numpy.random import Generator

from ..algebra.operators import Functional
from .base import ProbabilityMeasure

if TYPE_CHECKING:
    from ..numerics.randomized import Estimate

__all__ = ["AbsolutelyContinuousMeasure", "combine_log_normalizer"]


class AbsolutelyContinuousMeasure[X](ProbabilityMeasure[X]):
    """A measure carrying its log density relative to a reference.

    On its own it is a *target*: it states where its probability sits
    relative to the reference, and that is what an MCMC kernel or an
    importance weight consumes, but it cannot be sampled -- sampling it is
    the sampler's job. Subclasses that can also be sampled say so by being
    something else as well: the linear-Gaussian posterior,
    :class:`~pygeoinf.inference.gaussian.GaussianPosterior`, is a
    ``GaussianMeasure`` first and one of these second, so it draws exactly
    *and* states its density against the prior.

    Args:
        reference: the measure ``nu`` the density is taken against. The
            domain is the reference's.
        relative_log_density: ``f``, a
            :class:`~pygeoinf.algebra.operators.Functional` on the domain,
            with ``d mu / d nu == exp(f - c)``. A functional rather than a
            bare callable so that it carries its derivative, which is what a
            gradient-based sampler steps along.
        log_normalizer: ``c == log integral exp(f) d nu``, as an
            :class:`~pygeoinf.numerics.randomized.Estimate`, or a
            zero-argument callable computing one on first use, or ``None``
            when it is unknown and the density is meant up to a constant.
    """

    def __init__(  # noqa: positional - cooperative __init__, see below
        self,
        reference: ProbabilityMeasure[X],
        relative_log_density: Functional[X] | None = None,
        /,
        *,
        log_normalizer: "Estimate | Callable[[], Estimate] | None" = None,
    ) -> None:
        from ..algebra.spaces import HilbertSpace

        if isinstance(reference, HilbertSpace):
            # Mid-MRO: a sibling's ``super().__init__(domain)`` landing here
            # on its way to ProbabilityMeasure, as GaussianMeasure's does in
            # GaussianPosterior, which inherits from both. Pass the domain
            # along; the subclass sets the density triple afterwards through
            # ``_init_density``.
            if relative_log_density is not None:
                raise TypeError("A domain takes no density; pass a reference.")
            super().__init__(reference)
            return
        if relative_log_density is None:
            raise TypeError(
                "AbsolutelyContinuousMeasure needs a reference and a density."
            )
        super().__init__(reference.domain)
        self._init_density(
            reference, relative_log_density, log_normalizer=log_normalizer
        )

    def _init_density(
        self,
        reference: ProbabilityMeasure[X],
        relative_log_density: Functional[X],
        /,
        *,
        log_normalizer: "Estimate | Callable[[], Estimate] | None" = None,
    ) -> None:
        """Set the density triple; the one entry point, for subclasses too.

        A subclass whose primary ``__init__`` is another class's -- the
        linear-Gaussian posterior's is ``GaussianMeasure``'s -- calls this
        afterwards rather than threading the density through a cooperative
        ``super()`` chain.
        """
        if relative_log_density.domain != reference.domain:
            raise ValueError(
                f"The density lives on {relative_log_density.domain!r}, not on "
                f"the reference's domain {reference.domain!r}."
            )
        self._reference = reference
        self._relative_log_density = relative_log_density
        self._log_normalizer_spec = log_normalizer
        self._log_normalizer_value: Estimate | None = None

    @property
    def reference(self) -> ProbabilityMeasure[X]:
        """The measure the density is taken against."""
        return self._reference

    @property
    def relative_log_density(self) -> Functional[X]:
        """``f`` with ``d mu / d nu == exp(f - log_normalizer)``.

        Not to be confused with :meth:`log_density`, which on a measure means
        the density against the space's own volume measure and is refused on
        a function-space discretisation. A subclass may have both, and
        ``relative_`` says which one this is.
        """
        return self._relative_log_density

    @property
    def log_normalizer(self) -> "Estimate | None":
        """``log integral exp(f) d nu``, computed on first use, or ``None``.

        For a Bayesian posterior over its prior this is the log of the
        normalizing constant of the likelihood's potential -- the evidence,
        in the convention the density functional fixes.
        """
        if self._log_normalizer_value is None and self._log_normalizer_spec is not None:
            spec = self._log_normalizer_spec
            self._log_normalizer_value = spec() if callable(spec) else spec
            self._log_normalizer_spec = self._log_normalizer_value
        return self._log_normalizer_value

    def relative_to(self, other: "AbsolutelyContinuousMeasure[X]", /) -> Functional[X]:
        """``log (d self / d other)``, up to an additive constant.

        Defined through the shared reference: ``f_self - f_other``, which is
        exact up to ``c_other - c_self``; when both normalizers are known the
        caller has the constant too. The two measures must carry the *same*
        reference object -- a density between measures over different
        references is exactly the construction this class refuses.

        Args:
            other: the measure to take the density against.

        Returns:
            The log density ratio as a functional, up to a constant.

        Raises:
            ValueError: if the references are not the same measure.
        """
        if other.reference is not self.reference:
            raise ValueError(
                "relative_to needs the same reference on both sides: a "
                "density between measures over different references is not "
                "defined by this construction."
            )
        return self._relative_log_density - other.relative_log_density

    @property
    def can_sample(self) -> bool:
        """False: a density states where the probability is, not how to draw.

        A subclass that also draws, the linear-Gaussian posterior being one,
        is a ``GaussianMeasure`` first and answers as one.
        """
        return False

    def sample(self, *, rng: Generator | None = None) -> X:
        """Refused: a density states where probability is, not how to draw it.

        Raises:
            NotImplementedError: always, on the base class. A posterior that
                can draw is also a ``GaussianMeasure`` (or a sampled result);
                for anything else, drawing from a density is a sampler's job.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is a density relative to its reference "
            "and cannot be sampled directly; hand it to a sampler."
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(reference={type(self._reference).__name__} "
            f"on {self.domain!r})"
        )


def combine_log_normalizer(
    terms: list[tuple[float, "Estimate | float"]],
) -> "Estimate":
    """A linear combination of exact floats and ``Estimate`` values.

    The standard errors add in quadrature, which assumes independence of the
    stochastic parts -- true where each comes from its own probe draws. This
    is how a posterior's log normalizer is assembled from the prior's and
    the likelihood's when some of the terms are stochastic estimates.

    Args:
        terms: ``(weight, value)`` pairs, each value an exact float or an
            :class:`~pygeoinf.numerics.randomized.Estimate`.

    Returns:
        The weighted sum as an ``Estimate``, with the standard errors
        combined in quadrature and the sample count the largest of the
        terms'.
    """
    import numpy as np

    from ..numerics.randomized import Estimate

    value = 0.0
    variance = 0.0
    samples = 0
    for weight, term in terms:
        if isinstance(term, Estimate):
            value += weight * term.value
            variance += (weight * term.standard_error) ** 2
            samples = max(samples, term.samples)
        else:
            value += weight * float(term)
    return Estimate(value, float(np.sqrt(variance)), samples)
