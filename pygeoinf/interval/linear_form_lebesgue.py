from pygeoinf import LinearForm
from typing import TYPE_CHECKING, Optional, Callable, Union
if TYPE_CHECKING:
    from pygeoinf.interval import Function, Lebesgue, Sobolev
import numpy as np


class LinearFormLebesgue(LinearForm):
    def __init__(
        self,
        domain: 'Lebesgue',
        /,
        *,
        mapping: Optional[Callable[['Function'], float]] = None,
        kernel: Optional['Function'] = None,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> None:

        if mapping is None and kernel is None:
            raise ValueError("Either mapping or kernel must be provided")
        if mapping is not None and kernel is not None:
            raise ValueError("Specify either mapping or kernel, not both")
        if mapping is None and kernel is not None:
            self._kernel = kernel
            mapping = self._mapping_impl
        if mapping is not None and kernel is None:
            self._kernel = None  # type: ignore

        # We will do something cheeky here. By  default LinearOperator computes
        # the components if they don't exist, but we don't want to work with
        # components, so we will give it fake components just so it stops from
        # trying to compute them(which would call for basis, which we may not
        # want to have)
        fake_components = np.zeros((domain.dim,))  # type: ignore
        super().__init__(
            domain,
            mapping=mapping,
            components=fake_components,
            parallel=parallel,
            n_jobs=n_jobs
        )

    def _mapping_impl(self, v: 'Function') -> float:
        return (self._kernel * v).integrate()

    @property
    def kernel(self) -> Union['Function', None]:
        return self._kernel
