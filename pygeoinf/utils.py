from threadpoolctl import threadpool_limits


def configure_threading(n_threads: int = 1):
    """
    Sets the maximum number of threads used by underlying linear algebra
    backends (MKL, OpenBLAS, etc.).

    Args:
        n_threads: The number of threads to allow.
                   Set to 1 for serial execution (safe for multiprocessing).
                   Set to -1 or None to use all available cores.
    """
    threadpool_limits(limits=n_threads)
    print(f"Backend threading restricted to {n_threads} thread(s).")
