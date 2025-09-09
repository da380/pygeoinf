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
            f"\nRunning {n_checks} randomized axiom checks for {self.__class__.__name__}..."
        )
        for _ in range(n_checks):
            # Generate fresh random data for each trial
            x, y, z = self.random(), self.random(), self.random()
            a, b = np.random.randn(), np.random.randn()

            # Run all checks
            self._check_vector_space_axioms(x, y, a)
            self._check_inner_product_axioms(x, y, z, a, b)
            self._check_mapping_identities(x)

        print(f"✅ All {n_checks} Hilbert space axiom checks passed successfully.")
