## Phase 3 Complete: Hat/Spline Providers Set Support

`HatFunctionProvider` and `SplineFunctionProvider` in `fem.py` now pass `support=` to every `Function` they return, completing support-metadata propagation from basis providers through to coefficient algebra.

**Files created/changed:**
- `intervalinf/intervalinf/providers/functions/fem.py`
- `intervalinf/tests/providers/test_standalone_providers.py`

**Functions created/changed:**
- `HatFunctionProvider.get_function_by_index`: computes `(nodes[i-1], nodes[i+1])` (clamped at boundaries) and passes `support=` to `Function`
- `SplineFunctionProvider.get_function_by_index`: derives B-spline local support `(knots[i], knots[i+degree+1])`; `None` for degenerate zero-width spans
- `SplineFunctionProvider.get_function_by_parameters`: infers support from `parameters['knots']`, `parameters['degree']`, `parameters['index']` when all keys are present

**Tests created/changed:**
- `TestFEMProviderSupport.test_hat_provider_basis_function_has_compact_support`
- `TestFEMProviderSupport.test_hat_provider_interior_node_support_interval`
- `TestFEMProviderSupport.test_hat_provider_left_boundary_node_support`
- `TestFEMProviderSupport.test_hat_provider_right_boundary_node_support`
- `TestFEMProviderSupport.test_hat_provider_homogeneous_basis_has_compact_support`
- `TestFEMProviderSupport.test_spline_provider_basis_function_has_compact_support`
- `TestFEMProviderSupport.test_spline_provider_support_is_finite_interval`
- `TestFEMProviderSupport.test_spline_provider_support_within_domain`
- `TestFEMProviderSupport.test_spline_get_function_by_parameters_has_support`

**Review Status:** APPROVED

**Git Commit Message:**
```
feat: set support metadata on hat and spline basis functions

- HatFunctionProvider.get_function_by_index computes local support
  as (nodes[i-1], nodes[i+1]), clamped to domain at boundaries
- SplineFunctionProvider.get_function_by_index derives B-spline
  support (knots[i], knots[i+degree+1]); None for degenerate spans
- SplineFunctionProvider.get_function_by_parameters infers support
  when knots/degree/index keys present in parameters dict
- Add 9 tests in TestFEMProviderSupport covering boundary nodes,
  interior nodes, homogeneous BCs, and spline support correctness
```
