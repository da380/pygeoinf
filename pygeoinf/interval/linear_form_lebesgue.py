from pygeoinf import LinearForm
from typing import TYPE_CHECKING, Optional, Callable, Union
if TYPE_CHECKING:
    from pygeoinf.interval import Function, Lebesgue, Sobolev


class LinearFormLebesgue(LinearForm):
    def __init__(
        self,
        domain: 'Lebesgue',
        kernel: 'Function',
        /,
        *,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> None:

        self._kernel = kernel

        super().__init__(
            domain,
            mapping=self._mapping_impl,
            parallel=parallel,
            n_jobs=n_jobs
        )

    def _mapping_impl(self, v: 'Function') -> float:
        return (self._kernel * v).integrate()

    @property
    def kernel(self) -> 'Function':
        return self._kernel


class LinearFormSobolev(LinearForm):
    def __init__(
        self,
        domain: 'Sobolev',
        /,
        *,
        mapping: Optional[Callable[['Function'], float]] = None,
        kernel: 'Function',
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

        super().__init__(
            domain,
            mapping=mapping,
            parallel=parallel,
            n_jobs=n_jobs
        )

    def _mapping_impl(self, v: 'Function') -> float:
        return (self._kernel * v).integrate()

    @property
    def kernel(self) -> Union['Function', None]:
        return self._kernel
