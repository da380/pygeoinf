"""
This module contains functions used to check instances of the classed defined within hilbert.py
"""

import numpy as np


class HilbertSpaceChecks:
    """
    Class defining consistency checks for a HilbertSpace.
    """

    def __init__(self, space, /, *, rtol=1.0e-6, atol=1.0e-6):
        self._space = space
        self._rtol = rtol
        self._atol = atol

    @property
    def space(self):
        """
        Return the space
        """
        return self._space

    @property
    def rtol(self):
        """
        Return the relative tolerance for numerical tests.
        """
        return self._rtol

    @property
    def atol(self):
        """
        Return the absolute tolerance for numerical tests.
        """
        return self._atol

    def _list_of_checks(self):
        return [
            attribute
            for attribute in dir(self)
            if callable(getattr(self, attribute)) and attribute.startswith("check")
        ]

    def passed_checks(self):
        """
        Return a list of checks that have passed.
        """
        results_and_checks = [
            [getattr(self, check)(), check] for check in self._list_of_checks()
        ]
        return [check for value, check in results_and_checks if value]

    def failed_checks(self):
        """
        Return a list of checks that have failed.
        """
        results_and_checks = [
            [getattr(self, check)(), check] for check in self._list_of_checks()
        ]
        return [check for value, check in results_and_checks if not value]

    def all_checks_passed(self, /, *, trials=1):
        """
        True if all tests have passed. By default one random trial
        of the tests is performed. Optionally, multiple random trials
        can be run by setting the trials variable > 1.
        """
        okay = True
        for _ in range(trials):
            okay = okay and len(self.failed_checks()) == 0
        return okay

    def check_norm_non_negativity(self):
        """
        True if the norm returns non-negative values.
        """
        x = self.space.random()
        return self.space.norm(x) >= 0

    def check_norm_homogeneity(self):
        """
        True if the norm is positively homogeneous of order 1.
        """
        a = np.random.randn()
        x1 = self.space.random()
        x2 = self.space.multiply(a, x1)
        norm1 = np.abs(a) * self.space.norm(x1)
        norm2 = self.space.norm(x2)
        diff = np.abs(norm2 - norm1) / norm1 if norm1 > 0 else 0
        return diff < self.rtol

    def check_triangle_inequality(self):
        """
        True if triangle inequality is satisfied.
        """
        x1 = self.space.random()
        x2 = self.space.random()
        x3 = self.space.add(x1, x2)
        norm1 = self.space.norm(x1)
        norm2 = self.space.norm(x2)
        norm3 = self.space.norm(x3)
        diff = (norm1 + norm2 - norm3) / (norm1 + norm2)
        return diff > -self.rtol

    def check_inner_product_symmetry(self):
        """
        True if the inner product is symmetric in its arguments.
        """
        x1 = self.space.random()
        x2 = self.space.random()
        ip1 = self.space.inner_product(x1, x2)
        ip2 = self.space.inner_product(x2, x1)
        diff = np.abs(ip2 - ip1) / np.abs(ip1)
        return diff < self.rtol

    def check_inner_product_bilinearity(self):
        """
        True if the inner product is bilinear. Based on check of first
        argument only, assumes the symmetry condition has been met.
        """
        a1 = np.random.randn()
        a2 = np.random.randn()
        x1 = self.space.random()
        x2 = self.space.random()
        x3 = self.space.random()
        x4 = self.space.add(self.space.multiply(a1, x1), self.space.multiply(a2, x2))
        ip1 = a1 * self.space.inner_product(x1, x3) + a2 * self.space.inner_product(
            x2, x3
        )
        ip2 = self.space.inner_product(x4, x3)
        diff = np.abs(ip2 - ip1) / np.abs(ip1)
        return diff < self.rtol

    def check_riesz_mapping_definition(self):
        """
        Checks isomorphism from dual to space is valid.
        """
        x1 = self.space.random()
        xp = self.space.dual.random()
        p1 = xp(x1)
        x2 = self.space.from_dual(xp)
        p2 = self.space.inner_product(x2, x1)
        diff = np.abs(p2 - p1) / np.abs(p1)
        return diff < self.rtol

    def check_inverse_riesz_mapping_definition(self):
        """
        True if riesz mapping and its inverse are compatible.
        """
        x1 = self.space.random()
        xp = self.space.to_dual(x1)
        x2 = self.space.from_dual(xp)
        x3 = self.space.subtract(x1, x2)
        diff = self.space.norm(x3) / self.space.norm(x1)
        return diff < self.rtol

    def check_riesz_mapping_linearity(self):
        """
        True if the riesz mapping is linear.
        """
        xp1 = self.space.dual.random()
        xp2 = self.space.dual.random()
        x1 = self.space.from_dual(xp1)
        x2 = self.space.from_dual(xp2)
        xp3 = xp1 + xp2
        x3 = self.space.from_dual(xp3)
        x4 = self.space.subtract(x3, self.space.add(x1, x2))
        diff = self.space.norm(x4) / self.space.norm(x3)
        return diff < self.rtol

    def check_inverse_riesz_mapping_linearity(self):
        """
        True if the riesz mapping is linear.
        """
        x1 = self.space.random()
        x2 = self.space.random()
        xp1 = self.space.to_dual(x1)
        xp2 = self.space.to_dual(x2)
        x3 = self.space.add(x1, x2)
        xp3 = self.space.to_dual(x3)
        xp4 = xp3 - xp1 - xp2
        diff = self.space.dual.norm(xp4) / self.space.dual.norm(xp3)
        return diff < self.rtol

    def check_to_component_linearity(self):
        """
        True if to_component mapping is linear.
        """
        a1 = np.random.randn()
        a2 = np.random.randn()
        x1 = self.space.random()
        x2 = self.space.random()
        x3 = self.space.add(self.space.multiply(a1, x1), self.space.multiply(a2, x2))
        c1 = self.space.to_components(x1)
        c2 = self.space.to_components(x2)
        c3 = self.space.to_components(x3)
        c4 = c3 - a1 * c1 - a2 * c2
        diff = np.linalg.norm(c4) / np.linalg.norm(c1)
        return diff < self.rtol

    def check_from_component_linearity(self):
        """
        True if from_component mapping is linear.
        """
        dim = self.space.dim
        a1 = np.random.randn()
        a2 = np.random.randn()
        c1 = np.random.randn(dim)
        c2 = np.random.randn(dim)
        c3 = a1 * c1 + a2 * c2
        x1 = self.space.from_components(c1)
        x2 = self.space.from_components(c2)
        x3 = self.space.from_components(c3)
        x4 = self.space.subtract(
            x3, self.space.add(self.space.multiply(a1, x1), self.space.multiply(a2, x2))
        )
        diff = self.space.norm(x4) / self.space.norm(x3)
        return diff < self.rtol

    def check_componet_mapping_compatibility(self):
        """
        True if to_component and from_component mappings are
        mutual inverses.
        """
        x1 = self.space.random()
        c1 = self.space.to_components(x1)
        x2 = self.space.from_components(c1)
        x3 = self.space.subtract(x1, x2)
        diff = self.space.norm(x3) / self.space.norm(x1)
        return diff < self.rtol
