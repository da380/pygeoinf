## Plan: Preserve Lebesgue Function support metadata

Ensure that compact-support metadata (`Function.support`) is not accidentally dropped when working with Lebesgue spaces (basis functions, coefficient reconstructions, restrictions, and algebra), and that identically-zero results use `support=[]`.

**Phases**
1. **Phase 1: Preserve support under restriction and disjoint products**
    - **Objective:**
        - `Function.restrict()` preserves support by intersecting with the restricted domain.
        - Multiplication of functions with disjoint compact supports returns an identically-zero function with `support=[]` (not `None`).
    - **Files/Functions to Modify/Create:**
        - intervalinf/core/functions.py: `Function.restrict`, disjoint-support branch in `Function.__mul__`.
        - intervalinf/tests/core/test_functions.py (or existing appropriate test file).
    - **Tests to Write:**
        - `test_restrict_intersects_support_single_interval`
        - `test_restrict_no_overlap_gives_empty_support`
        - `test_restrict_support_none_stays_none`
        - `test_mul_disjoint_support_returns_empty_support`
    - **Steps:**
        1. Write failing tests for support intersection and disjoint-support multiplication.
        2. Implement minimal changes in `Function.restrict` and the disjoint-support path in `__mul__`.
        3. Run the focused tests; ensure they pass.

2. **Phase 2: Infer/preserve support for coefficient-based Lebesgue elements**
    - **Objective:**
        - `Lebesgue.from_components(coefficients)` infers support as the union of supports of basis functions with `|c_i| > tol`.
        - Zero coefficients produce `support=[]`.
        - `Lebesgue.add/multiply/axpy` preserve/update support when operating on coefficient-backed functions.
    - **Files/Functions to Modify/Create:**
        - intervalinf/spaces/lebesgue.py: `from_components`, `add`, `multiply`, `axpy`.
        - intervalinf/tests/spaces/test_lebesgue.py (or existing appropriate test file).
    - **Tests to Write:**
        - `test_basis_vector_support_matches_basis_function_hat` (or boxcar)
        - `test_from_components_union_support_sparse`
        - `test_from_components_zero_coeffs_support_empty`
        - `test_from_components_global_basis_support_none` (Fourier)
        - `test_lebesgue_add_preserves_support_coefficients`
        - `test_lebesgue_multiply_preserves_support_coefficients`
        - `test_lebesgue_axpy_updates_support_in_place`
    - **Steps:**
        1. Write failing tests for inferred/propagated support.
        2. Implement tolerance-based support inference in `from_components`.
        3. Update algebra helpers to carry/update support.
        4. Run focused tests; ensure they pass.

3. **Phase 3: Ensure locally-supported providers set support correctly**
    - **Objective:**
        - Hat / spline / wavelet basis providers attach correct compact support intervals to their basis functions, enabling inference in Phase 2.
    - **Files/Functions to Modify/Create:**
        - intervalinf/providers/functions/fem.py (hat/spline)
        - intervalinf/providers/functions/wavelets.py (wavelets)
        - tests as needed near the provider or space tests.
    - **Tests to Write:**
        - Provider/space tests asserting `get_basis_function(i).support` is a finite interval for locally supported bases.
    - **Steps:**
        1. Identify providers used by `Lebesgue(..., basis=...)` that should be compactly supported.
        2. Add support intervals in provider outputs.
        3. Add/adjust tests; run focused tests.

4. **Phase 4: Update reference documentation**
    - **Objective:**
        - Update intervalinf reference docs to reflect the new support-preservation behavior and the meaning of `support=[]`.
    - **Files/Functions to Modify/Create:**
        - intervalinf/plans/intervalinf-reference.md
    - **Tests to Write:**
        - None.
    - **Steps:**
        1. Update the reference document summary of Function support semantics and propagation.

**Defaults confirmed**
1. Identically-zero functions use `support=[]`.
2. Coefficient sparsity detection uses a tolerance: include basis function i in the union when `|c_i| > tol`.
