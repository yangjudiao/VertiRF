from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilterSpec:
    filter_type: str = "gaussian"
    gauss_f0: float = math.pi
    low_hz: float = 0.1
    high_hz: float = 0.8
    corners: int = 4
    transition_hz: float = 0.05
    tukey_alpha: float = 0.3


def _safe_low_high(low_hz: float, high_hz: float, nyquist: float) -> tuple[float, float]:
    low = max(1e-6, float(low_hz))
    high = min(float(high_hz), max(low + 1e-6, nyquist - 1e-6))
    if high <= low:
        high = min(nyquist - 1e-6, low + 0.05)
    return low, high


def _gaussian_response(freq_hz: np.ndarray, gauss_f0: float) -> np.ndarray:
    w = 2.0 * np.pi * freq_hz
    f0 = max(1e-9, float(gauss_f0))
    return np.exp(-0.25 * (w / f0) ** 2)


def _butterworth_bandpass_response(freq_hz: np.ndarray, low_hz: float, high_hz: float, corners: int) -> np.ndarray:
    n = max(1, int(corners))
    f = np.asarray(freq_hz, dtype=np.float64)
    lo = np.zeros_like(f)
    hi = np.zeros_like(f)
    mask = f > 0.0
    f_safe = np.where(mask, f, 1.0)
    lo[mask] = 1.0 / np.sqrt(1.0 + (low_hz / f_safe[mask]) ** (2 * n))
    hi = 1.0 / np.sqrt(1.0 + (f_safe / high_hz) ** (2 * n))
    resp = lo * hi
    resp[~mask] = 0.0
    return np.clip(resp, 0.0, 1.0)


def _raised_cosine_edge(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out[x <= 0.0] = 0.0
    out[x >= 1.0] = 1.0
    mask = (x > 0.0) & (x < 1.0)
    out[mask] = 0.5 * (1.0 - np.cos(np.pi * x[mask]))
    return out


def _raised_cosine_bandpass_response(freq_hz: np.ndarray, low_hz: float, high_hz: float, transition_hz: float) -> np.ndarray:
    f = np.asarray(freq_hz, dtype=np.float64)
    t = max(1e-6, float(transition_hz))

    up = _raised_cosine_edge((f - (low_hz - t)) / t)
    down = 1.0 - _raised_cosine_edge((f - high_hz) / t)
    resp = up * down
    resp[f <= 0.0] = 0.0
    return np.clip(resp, 0.0, 1.0)


def _tukey_bandpass_response(freq_hz: np.ndarray, low_hz: float, high_hz: float, alpha: float) -> np.ndarray:
    f = np.asarray(freq_hz, dtype=np.float64)
    a = min(1.0, max(1e-6, float(alpha)))
    width = max(1e-6, high_hz - low_hz)
    taper = 0.5 * a * width

    left0 = low_hz
    left1 = low_hz + taper
    right0 = high_hz - taper
    right1 = high_hz

    resp = np.zeros_like(f)
    pass_mask = (f >= left1) & (f <= right0)
    resp[pass_mask] = 1.0

    left_mask = (f >= left0) & (f < left1)
    if np.any(left_mask):
        x = (f[left_mask] - left0) / max(1e-9, left1 - left0)
        resp[left_mask] = 0.5 * (1.0 - np.cos(np.pi * x))

    right_mask = (f > right0) & (f <= right1)
    if np.any(right_mask):
        x = (f[right_mask] - right0) / max(1e-9, right1 - right0)
        resp[right_mask] = 0.5 * (1.0 + np.cos(np.pi * x))

    resp[f <= 0.0] = 0.0
    return np.clip(resp, 0.0, 1.0)


def build_zero_phase_response(nfft: int, dt: float, spec: FilterSpec) -> np.ndarray:
    if nfft <= 0:
        raise ValueError("nfft must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    nyquist = 0.5 / dt
    freq = np.fft.rfftfreq(nfft, d=dt)
    low_hz, high_hz = _safe_low_high(spec.low_hz, spec.high_hz, nyquist)

    ftype = str(spec.filter_type).lower().strip()
    if ftype == "gaussian":
        amp = _gaussian_response(freq, gauss_f0=spec.gauss_f0)
    elif ftype == "butterworth_bandpass":
        amp = _butterworth_bandpass_response(freq, low_hz=low_hz, high_hz=high_hz, corners=spec.corners)
    elif ftype == "raised_cosine_bandpass":
        amp = _raised_cosine_bandpass_response(
            freq,
            low_hz=low_hz,
            high_hz=high_hz,
            transition_hz=max(1e-6, spec.transition_hz),
        )
    elif ftype == "tukey_bandpass":
        amp = _tukey_bandpass_response(freq, low_hz=low_hz, high_hz=high_hz, alpha=spec.tukey_alpha)
    else:
        raise ValueError(f"unsupported filter_type: {spec.filter_type}")

    return np.asarray(amp, dtype=np.float64)


def apply_zero_phase_filter(signal: np.ndarray, dt: float, spec: FilterSpec, nfft: int | None = None) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    n = int(x.size)
    nfft_ = int(nfft) if nfft is not None else n
    resp = build_zero_phase_response(nfft_, dt, spec)
    xf = np.fft.rfft(x, n=nfft_)
    yf = xf * resp
    y = np.fft.irfft(yf, n=nfft_)
    return y[:n]
