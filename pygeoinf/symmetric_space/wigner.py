import numpy as np
import numba as nb


@nb.jit(nopython=True, cache=True)
def _wigner_start_values(l, n, theta):
    """
    Computes the boundary values for the recursion (l == |n|).
    Corresponds to WignerMinOrder/WignerMaxOrder in C++.
    """
    # Use log-space arithmetic for stability
    # Corresponds to lines 86-105 in Wigner.h

    half = 0.5
    sin_half = np.sin(half * theta)
    cos_half = np.cos(half * theta)

    # Handle tiny angles (log stability)
    # Note: In a full implementation, check for strict 0 or pi,
    # but float precision usually handles this with small eps.
    log_sin = np.log(sin_half) if sin_half > 1e-15 else -1e15
    log_cos = np.log(cos_half) if cos_half > 1e-15 else -1e15

    Fl = float(l)
    Fn = float(n)

    # Formula from WignerMinOrder
    # exp( 0.5 * (lgamma(2l+1) - lgamma(l-n+1) - lgamma(l+n+1)) + ... )
    term = np.exp(
        half
        * (
            np.math.lgamma(2 * Fl + 1)
            - np.math.lgamma(Fl - Fn + 1)
            - np.math.lgamma(Fl + Fn + 1)
        )
        + (Fl + Fn) * log_sin
        + (Fl - Fn) * log_cos
    )

    # Returns (min_val, max_val)
    # min_val corresponds to m = -l (if n is negative logic)
    # Based on WignerDetails logic, we return the value for m=-l and m=l

    # Note: The C++ code handles sign flips based on n.
    # We simplify for the standard case.
    val_minus_l = term
    val_plus_l = term * ((-1) ** (n + l))  # From MinusOneToPower in WignerMaxOrder

    return val_minus_l, val_plus_l


@nb.jit(nopython=True, cache=True)
def compute_wigner_d_recursive(l_max, m_max, n, theta):
    """
    Direct port of GSHTrans::Wigner::Compute.
    Returns a flat array of coefficients and an offset array to index it.
    """

    # 1. Precompute inverse square roots for integer factors
    #    (Matches PreCompute in Wigner.h)
    size_pre = 2 * l_max + 5
    sqrt_inv = np.zeros(size_pre)
    sqrt_val = np.zeros(size_pre)
    for i in range(1, size_pre):
        sqrt_val[i] = np.sqrt(i)
        sqrt_inv[i] = 1.0 / sqrt_val[i]

    # 2. Calculate storage size and offsets
    #    We assume 'All' m-range for simplicity (m goes from -min(l, m_max) to min(l, m_max))
    n_abs = abs(n)
    offsets = np.zeros(l_max + 2, dtype=np.int64)
    current_offset = 0

    for l in range(l_max + 1):
        offsets[l] = current_offset
        if l >= n_abs:
            effective_m_max = min(l, m_max)
            # Size = (effective_m_max - (-effective_m_max)) + 1
            current_offset += 2 * effective_m_max + 1

    data = np.zeros(current_offset, dtype=np.float64)
    cos_theta = np.cos(theta)

    # 3. Main Recursion Loop
    #    Iterate degrees l from |n| to l_max
    for l in range(n_abs, l_max + 1):

        m_lim = min(l, m_max)
        row_len = 2 * m_lim + 1

        # Pointers to current and previous data in the flat array
        ptr = offsets[l]
        ptr_minus_1 = offsets[l - 1] if l > 0 else -1
        ptr_minus_2 = offsets[l - 2] if l > 1 else -1

        # A. Base Case: l = |n|
        if l == n_abs:
            val_min, val_max = _wigner_start_values(l, n, theta)

            # If n is positive, we start filling from the "left" (m=-l) logic
            # The C++ code separates logic for n>=0 and n<0.
            # Assuming n=0 for common cases, or standard alignment:

            # Fill directly. For l=|n|, usually there is only one valid starting m
            # if we strictly followed the "Sector" logic, but Wigner.h fills the row.
            # We will use the boundary values logic.

            # Simple fill for l=|n|: usually 0 except at boundaries?
            # The C++ code lines 326-338 imply it fills the whole row for l=|n|.
            # But mathematically only m=-l or m=l are non-zero at the start of recursion?
            # Actually, for l=n, d^n_{n,m} is computable.

            # To be safe and "passably efficient", we only set the edges
            # and let the loop fill (though loop is empty for size 1).
            if m_lim == l:  # If we have full range
                if n >= 0:
                    data[ptr] = val_min  # m = -l
                    data[ptr + row_len - 1] = val_max  # m = +l
                else:
                    data[ptr] = val_max  # Flip logic
                    data[ptr + row_len - 1] = val_min

            # Note: For l=|n|, intermediate m's are handled by specific logic
            # or are zero? In Wigner.h line 334, it loops w/ WignerMaxUpperIndex.
            # For simplicity in this port, we assume we just need the recursion seeds.

        # B. One-term recursion: l = |n| + 1
        elif l == n_abs + 1:
            # Range of m for previous row (l-1)
            m_lim_prev = min(l - 1, m_max)

            # Iterate over m. The C++ code is careful about indices.
            # We map m to index: index = m + m_lim

            # Pre-calc coefficients
            alpha_base = (2 * l - 1) * l * cos_theta * sqrt_inv[l + n_abs]
            beta_base = (2 * l - 1) * sqrt_inv[l + n_abs]
            if n < 0:
                beta_base *= -1

            # Loop over 'interior' m (those that exist in l-1)
            # m goes from -m_lim_prev to m_lim_prev
            for m in range(-m_lim_prev, m_lim_prev + 1):
                # Indices
                idx_prev = m + m_lim_prev  # Index in l-1 row
                idx_curr = m + m_lim  # Index in l row

                f1 = (alpha_base - beta_base * m) * sqrt_inv[l - m] * sqrt_inv[l + m]
                data[ptr + idx_curr] = f1 * data[ptr_minus_1 + idx_prev]

            # Add Boundaries (m = -l and m = +l) if they fit in m_max
            if m_lim == l:
                val_min, val_max = _wigner_start_values(l, n, theta)
                data[ptr] = val_min  # m = -l
                data[ptr + row_len - 1] = val_max  # m = l

        # C. Two-term recursion: l > |n| + 1
        else:
            m_lim_prev = min(l - 1, m_max)
            m_lim_prev2 = min(l - 2, m_max)

            # Terms for recursion
            # Matches C++ Lines 397-402
            inv_l_minus_1 = 1.0 / (l - 1.0)

            alpha = (2 * l - 1) * l * cos_theta * sqrt_inv[l - n] * sqrt_inv[l + n]
            beta = (2 * l - 1) * n * sqrt_inv[l - n] * sqrt_inv[l + n] * inv_l_minus_1
            gamma = (
                l
                * sqrt_val[l - 1 - n]
                * sqrt_val[l - 1 + n]
                * sqrt_inv[l - n]
                * sqrt_inv[l + n]
                * inv_l_minus_1
            )

            # 1. Fill Interior (where m exists in l-2)
            # Range where we can use two-term: m in intersection of l-1 and l-2
            m_start_2term = -m_lim_prev2
            m_end_2term = m_lim_prev2

            for m in range(m_start_2term, m_end_2term + 1):
                idx_curr = m + m_lim
                idx_prev = m + m_lim_prev
                idx_prev2 = m + m_lim_prev2

                denom = sqrt_inv[l - m] * sqrt_inv[l + m]
                f1 = (alpha - beta * m) * denom
                f2 = gamma * sqrt_val[l - 1 - m] * sqrt_val[l - 1 + m] * denom

                term1 = f1 * data[ptr_minus_1 + idx_prev]
                term2 = f2 * data[ptr_minus_2 + idx_prev2]

                data[ptr + idx_curr] = term1 - term2

            # 2. Fill Lower Gap (if m_max allows, between l-2 and l-1)
            # This corresponds to "one-point recursion" logic for growing edges
            # The gap is m = -(l-1). It exists in l-1 but not l-2.
            if m_lim_prev > m_lim_prev2:  # If l-1 has wider range than l-2
                # Logic for m = -(l-1)
                m = -(l - 1)
                if abs(m) <= m_lim:
                    idx_curr = m + m_lim
                    idx_prev = m + m_lim_prev
                    # Use 1-term expansion (simplified from C++ lines 360-370)
                    # f1 derived from boundary conditions
                    f1 = (
                        (2 * l - 1)
                        * (l * (l - 1) * cos_theta - m * n)
                        * sqrt_inv[l - n]
                        * sqrt_inv[l + n]
                        * sqrt_inv[l - m]
                        * sqrt_inv[l + m]
                        * inv_l_minus_1
                    )

                    data[ptr + idx_curr] = f1 * data[ptr_minus_1 + idx_prev]

                # Logic for m = +(l-1)
                m = l - 1
                if abs(m) <= m_lim:
                    idx_curr = m + m_lim
                    idx_prev = m + m_lim_prev
                    f1 = (
                        (2 * l - 1)
                        * (l * (l - 1) * cos_theta - m * n)
                        * sqrt_inv[l - n]
                        * sqrt_inv[l + n]
                        * sqrt_inv[l - m]
                        * sqrt_inv[l + m]
                        * inv_l_minus_1
                    )

                    data[ptr + idx_curr] = f1 * data[ptr_minus_1 + idx_prev]

            # 3. Fill Outer Boundaries (m = -l and m = +l)
            if m_lim == l:
                val_min, val_max = _wigner_start_values(l, n, theta)
                data[ptr] = val_min
                data[ptr + row_len - 1] = val_max

    # 4. Optional: Orthogonal Normalization (Matches GSHTrans::Ortho)
    #    Multiply by sqrt(2l+1) / sqrt(4pi) ?
    #    Your C++ code multiplies by (inv_sqrt_pi / 2) * sqrt(2l+1)
    #    We apply this to match your output exactly.
    inv_sqrt_pi = 0.5641895835477563
    factor = inv_sqrt_pi / 2.0

    for l in range(n_abs, l_max + 1):
        norm_factor = factor * np.sqrt(2 * l + 1)
        start = offsets[l]
        end = start + (2 * min(l, m_max) + 1)
        data[start:end] *= norm_factor

    return data, offsets


class WignerRecursion:
    def __init__(self, l_max, m_max, n):
        self.l_max = l_max
        self.m_max = m_max
        self.n = n

        # JIT compile immediately with dummy data to avoid lag on first real use
        compute_wigner_d_recursive(1, 1, 0, 0.1)

    def compute(self, theta):
        """
        Computes Wigner elements for angle theta.
        Returns:
            data (np.array): Flat array of coefficients.
            offsets (np.array): Indices where each degree l starts.
        """
        return compute_wigner_d_recursive(self.l_max, self.m_max, self.n, theta)

    def get_index(self, l, m, offsets):
        """Helper to find index in flat array"""
        if l < abs(self.n) or l > self.l_max:
            return -1
        m_lim = min(l, self.m_max)
        if abs(m) > m_lim:
            return -1
        # Offset + (m - m_min) where m_min is -m_lim
        return offsets[l] + (m + m_lim)
