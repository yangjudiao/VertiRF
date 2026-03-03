from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NativeBackendStatus:
    available: bool
    dll_path: str
    message: str


class CorrNativeBackend:
    def __init__(self, dll_path: Path) -> None:
        self._dll = ctypes.WinDLL(str(dll_path))
        self._fn = self._dll.corr_same_double
        self._fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        self._fn.restype = None

    def corr_same(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
        b = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if a.ndim != 1 or b.ndim != 1 or a.size != b.size:
            raise ValueError("corr_same expects two 1D arrays of same length")
        out = np.zeros_like(a)
        self._fn(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(int(a.size)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return out


def _default_dll_path() -> Path:
    return Path(__file__).resolve().with_name("vertirf_native_corr.dll")


def load_native_corr_backend() -> tuple[CorrNativeBackend | None, NativeBackendStatus]:
    dll = _default_dll_path()
    if not dll.exists():
        return None, NativeBackendStatus(False, str(dll), "native DLL not found")

    try:
        backend = CorrNativeBackend(dll)
    except Exception as exc:  # noqa: BLE001
        return None, NativeBackendStatus(False, str(dll), f"failed to load DLL: {exc}")

    return backend, NativeBackendStatus(True, str(dll), "native corr backend loaded")
