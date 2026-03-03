"""Native backend package.

Prompt27 adds optional C++ correlation kernel backend on Windows.
If DLL is unavailable, runtime falls back to NumPy implementation.
"""

from .backend import NativeBackendStatus, load_native_corr_backend

__all__ = ["NativeBackendStatus", "load_native_corr_backend"]
