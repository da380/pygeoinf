# File: scripts/colab_bootstrap.py
import sys
import importlib


def setup_env(package="pygeoinf", extra=None):
    """Dynamically installs the package and extra in Colab, or checks local install."""
    full_spec = f"{package}[{extra}]" if extra else package

    try:
        # Check if base is there
        importlib.import_module(package)
        # Check if extra is there (by checking the expected submodule)
        if extra:
            importlib.import_module(f"{package}.symmetric_space.{extra}")
        print(f"Environment ready: {full_spec} is available.")
        return
    except (ImportError, ModuleNotFoundError):
        pass

    # If missing, install only if in cloud
    if "google.colab" in sys.modules:
        print(f"Colab detected. Installing {full_spec}...")
        from IPython import get_ipython

        get_ipython().run_line_magic("pip", f'install -q "{full_spec}"')

        importlib.invalidate_caches()
        if package in sys.modules:
            importlib.reload(sys.modules[package])
        print("Installation complete.")
    else:
        print(
            f"ERROR: Missing dependencies. Locally, please run: pip install -e \".[{extra if extra else ''}]\""
        )
