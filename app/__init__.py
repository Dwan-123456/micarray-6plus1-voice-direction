import os

# The realtime workload uses many tiny 7x7 linear-algebra operations.  A large
# OpenBLAS pool only reserves memory and adds scheduling jitter, so configure it
# before importing runtime (and therefore NumPy/SciPy).
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from .runtime import ApplicationRuntime

__all__ = ["ApplicationRuntime"]
