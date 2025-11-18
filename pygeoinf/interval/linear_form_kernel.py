from pygeoinf import LinearForm, HilbertSpace
from typing import TYPE_CHECKING, Optional, Callable, Union, List
from pygeoinf.interval.functions import Function
from pygeoinf.interval.configs import IntegrationConfig, ParallelConfig
if TYPE_CHECKING:
    from pygeoinf.interval import Lebesgue
import numpy as np


class LinearFormKernel(LinearForm):
    def __init__(
        self,
        domain: HilbertSpace,
        /,
        *,
        mapping: Optional[Callable[['Function'], float]] = None,
        kernel: Optional[Union['Function', List['Function']]] = None,
        components: Optional[np.ndarray] = None,
        integration_config: IntegrationConfig,
        parallel_config: ParallelConfig = ParallelConfig(
            enabled=False, n_jobs=-1
        ),
    ) -> None:

        # Store configs
        self.integration = integration_config
        self.parallel = parallel_config
        if kernel is not None:
            self._kernel = kernel
            mapping = self._mapping_impl
            self._components = np.zeros((domain.dim,))
        elif mapping is not None:
            self._kernel = None  # type: ignore
            self._components = np.zeros((domain.dim,))
        elif components is not None:
            self._kernel = None  # type: ignore
            self._components = components
        else:
            raise ValueError("Either mapping, kernel, or components must be provided")

        # Get weight from domain if it has one (e.g., Lebesgue space)
        # Otherwise assume no weight (weight = None means standard inner product)
        self._weight = getattr(domain, '_weight', None)
        # We will do something cheeky here. By  default LinearOperator computes
        # the components if they don't exist, but we don't want to work with
        # components, so we will give it fake components just so it stops from
        # trying to compute them(which would call for basis, which we may not
        # want to have)
        self._fake_components = True
        super().__init__(
            domain,
            mapping=mapping,
            components=self._components,
            parallel=self.parallel.enabled,
            n_jobs=self.parallel.n_jobs
        )

    def _mapping_impl(self, v: Union[Function, List[Function]]) -> float:
        # If kernel is a list, expect v to be list/tuple of functions
        if isinstance(self._kernel, list):
            if not isinstance(v, list):
                raise ValueError("Input must be a list of functions")
            return sum(
                (k * vi).integrate(
                    weight=self._weight,
                    method=self.integration.method,
                    n_points=self.integration.n_points
                )
                for k, vi in zip(self._kernel, v)
            )
        else:
            if not isinstance(v, Function):
                raise ValueError("Input must be a function")
            return (self._kernel * v).integrate(
                weight=self._weight,
                method=self.integration.method,
                n_points=self.integration.n_points
            )

    @property
    def kernel(self) -> Union[Function, List[Function], None]:
        if self._kernel is None:
            return self._kernel
        elif self._weight is None:
            # No weight function - return kernel as-is
            return self._kernel
        else:
            # With weight function - adjust kernel by 1/weight
            return self._kernel * Function(
                self.domain,
                evaluate_callable=lambda x: 1 / self._weight(x)
            )

    @property
    def components(self) -> np.ndarray:
        # If we actually need the components, then we compute them using the
        # super method
        if self._fake_components:
            # _compute_components modifies self._components in place
            super()._compute_components(
                self._mapping_impl,
                self.parallel.enabled,
                self.parallel.n_jobs
            )
            self._fake_components = False
        return self._components
