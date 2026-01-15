# pygeoinf/utils.py
from threadpoolctl import threadpool_limits


def set_serial_backend():
    """
    Limits the underlying linear algebra backends (BLAS, OpenMP, etc.)
    to use a single thread. This prevents oversubscription when
    using explicit multiprocessing.
    """
    # This limits threads for all supported libraries (MKL, OpenBLAS, etc.)
    # The return value is a context manager, but calling it directly
    # sets the global state until changed again.
    threadpool_limits(limits=1)
    print("Backend threading restricted to 1 thread.")
