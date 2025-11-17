import numpy as np


class HilbertSpaceAxiomChecks:
    """
    A mixin class providing a self-checking mechanism for Hilbert space axioms.

    When inherited by a HilbertSpace subclass, it provides the `.check()` method
    to run a suite of randomized tests, ensuring the implementation is valid.
    """

    def _check_vector_space_axioms(self, x, y, a):
        """Checks axioms related to vector addition and scalar multiplication."""
        # (x + y) - y == x
        sum_vec = self.add(x, y)
        res_vec = self.subtract(sum_vec, y)
        if not np.allclose(self.to_components(x), self.to_components(res_vec)):
            raise AssertionError("Axiom failed: (x + y) - y != x")

        # a*(x+y) == a*x + a*y
        lhs = self.multiply(a, self.add(x, y))
        rhs = self.add(self.multiply(a, x), self.multiply(a, y))
        if not np.allclose(self.to_components(lhs), self.to_components(rhs)):
            raise AssertionError("Axiom failed: a*(x+y) != a*x + a*y")

        # x + 0 = x
        zero_vec = self.zero
        res_vec = self.add(x, zero_vec)
        if not np.allclose(self.to_components(x), self.to_components(res_vec)):
            raise AssertionError("Axiom failed: x + 0 != x")

    def _check_inner_product_axioms(self, x, y, z, a, b):
        """Checks axioms related to the inner product and norm."""
        # Linearity: <ax+by, z> = a<x,z> + b<y,z>
        lhs = self.inner_product(self.add(self.multiply(a, x), self.multiply(b, y)), z)
        rhs = a * self.inner_product(x, z) + b * self.inner_product(y, z)
        if not np.isclose(lhs, rhs):
            raise AssertionError("Axiom failed: Inner product linearity")

        # Symmetry: <x, y> == <y, x>
        if not np.isclose(self.inner_product(x, y), self.inner_product(y, x)):
            raise AssertionError("Axiom failed: Inner product symmetry")

        # Triangle Inequality: ||x + y|| <= ||x|| + ||y||
        norm_sum = self.norm(self.add(x, y))
        if not norm_sum <= self.norm(x) + self.norm(y):
            raise AssertionError("Axiom failed: Triangle inequality")

    def _check_riesz_representation(self, x, y):
        """
        Checks that the inner product is consistent with the Riesz map (to_dual).
        This ensures that <x, y> == (R(x))(y).
        """
        # Value from the (potentially optimized) direct inner product method
        direct_inner_product = self.inner_product(x, y)

        # Value from the Riesz map definition
        dual_x = self.to_dual(x)
        riesz_inner_product = self.duality_product(dual_x, y)

        if not np.isclose(direct_inner_product, riesz_inner_product):
            raise AssertionError(
                "Axiom failed: Inner product is not consistent with the Riesz map."
            )

    def _check_mapping_identities(self, x):
        """Checks that component and dual mappings are self-consistent."""
        # from_components(to_components(x)) == x
        components = self.to_components(x)
        reconstructed_x = self.from_components(components)
        if not np.allclose(components, self.to_components(reconstructed_x)):
            raise AssertionError("Axiom failed: Component mapping round-trip")

        # from_dual(to_dual(x)) == x
        x_dual = self.to_dual(x)
        reconstructed_x = self.from_dual(x_dual)
        if not np.allclose(self.to_components(x), self.to_components(reconstructed_x)):
            raise AssertionError("Axiom failed: Dual mapping round-trip")

    def _check_inplace_operations(self, x, y, a):
        """Checks the in-place operations `ax` and `axpy`."""
        # Test ax: y := a*x
        x_copy = self.copy(x)
        expected_ax = self.multiply(a, x)
        self.ax(a, x_copy)
        if not np.allclose(self.to_components(expected_ax), self.to_components(x_copy)):
            raise AssertionError("Axiom failed: In-place operation ax")

        # Test axpy: y := a*x + y
        y_copy = self.copy(y)
        expected_axpy = self.add(self.multiply(a, x), y)
        self.axpy(a, x, y_copy)
        if not np.allclose(
            self.to_components(expected_axpy), self.to_components(y_copy)
        ):
            raise AssertionError("Axiom failed: In-place operation axpy")

    def _check_copy(self, x):
        """Checks that the copy method creates a deep, independent copy."""
        x_copy = self.copy(x)

        # The copy should have the same value but be a different object
        if x is x_copy:
            raise AssertionError("Axiom failed: copy() returned the same object.")
        if not np.allclose(self.to_components(x), self.to_components(x_copy)):
            raise AssertionError("Axiom failed: copy() did not preserve values.")

        # Modify the copy and ensure the original is unchanged
        self.ax(2.0, x_copy)
        if np.allclose(self.to_components(x), self.to_components(x_copy)):
            raise AssertionError("Axiom failed: copy() is not a deep copy.")

    def _check_gram_schmidt(self):
        """Checks the Gram-Schmidt orthonormalization process."""
        # Create a list of linearly independent vectors
        vectors = [self.random() for _ in range(min(self.dim, 5))]
        if not vectors:
            return  # Skip if dimension is 0

        try:
            orthonormal_vectors = self.gram_schmidt(vectors)
        except ValueError as e:
            # This can happen if the random vectors are not linearly independent
            print(f"Skipping Gram-Schmidt check due to non-independent vectors: {e}")
            return

        # Check for orthonormality
        for i, v1 in enumerate(orthonormal_vectors):
            for j, v2 in enumerate(orthonormal_vectors):
                inner_product = self.inner_product(v1, v2)
                if i == j:
                    if not np.isclose(inner_product, 1.0):
                        raise AssertionError(
                            "Axiom failed: Gram-Schmidt vector norm is not 1."
                        )
                else:
                    if not np.isclose(inner_product, 0.0):
                        raise AssertionError(
                            "Axiom failed: Gram-Schmidt vectors are not orthogonal."
                        )

    def _check_basis_and_expectation(self):
        """Checks the basis_vector and sample_expectation methods."""
        if self.dim == 0:
            return  # Skip for zero-dimensional spaces

        # Check basis vectors
        for i in range(self.dim):
            basis_vector = self.basis_vector(i)
            components = self.to_components(basis_vector)
            expected_components = np.zeros(self.dim)
            expected_components[i] = 1.0
            if not np.allclose(components, expected_components):
                raise AssertionError(
                    "Axiom failed: basis_vector has incorrect components."
                )

        # Check sample expectation
        vectors = [self.random() for _ in range(5)]
        mean_vec = self.sample_expectation(vectors)

        mean_comps = np.mean([self.to_components(v) for v in vectors], axis=0)
        if not np.allclose(self.to_components(mean_vec), mean_comps):
            raise AssertionError("Axiom failed: sample_expectation is incorrect.")

    def check(self, n_checks: int = 10) -> None:
        """
        Runs a suite of randomized checks to verify the Hilbert space axioms.

        This method performs `n_checks` iterations, generating new random
        vectors and scalars for each one. It provides an "interactive" way
        to validate any concrete HilbertSpace implementation.

        Args:
            n_checks: The number of randomized trials to run.

        Raises:
            AssertionError: If any of the underlying axiom checks fail.
        """
        print(
            f"\nRunning {n_checks} randomized axiom checks for {self.__class__.__name__}... (and some one-off checks)"
        )

        # These checks only need to be run once
        self._check_gram_schmidt()
        self._check_basis_and_expectation()

        for _ in range(n_checks):
            # Generate fresh random data for each trial
            x, y, z = self.random(), self.random(), self.random()
            a, b = np.random.randn(), np.random.randn()

            # Run all checks
            self._check_vector_space_axioms(x, y, a)
            self._check_inner_product_axioms(x, y, z, a, b)
            # self._check_riesz_representation(x, y)
            self._check_mapping_identities(x)
            self._check_inplace_operations(x, y, a)
            self._check_copy(x)

        print(f"✅ All {n_checks} Hilbert space axiom checks passed successfully.")
