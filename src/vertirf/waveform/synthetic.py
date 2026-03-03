from __future__ import annotations

import math

import numpy as np


def ricker_wavelet(samples: int, dt: float, freq_hz: float = 0.8) -> np.ndarray:
    n = int(samples)
    t = (np.arange(n, dtype=np.float64) - 0.5 * (n - 1)) * dt
    a = (np.pi * float(freq_hz) * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    m = np.max(np.abs(w)) + 1e-12
    return w / m


def make_response(samples: int, allow_negative: bool = True, rng_seed: int = 0) -> np.ndarray:
    n = int(samples)
    rng = np.random.default_rng(int(rng_seed))
    r = np.zeros(n, dtype=np.float64)

    center = n // 2
    r[center] = 1.0
    r[center + 12] = -0.45
    r[center + 36] = 0.30

    if allow_negative:
        r[center - 20] = 0.55
        r[center - 40] = -0.25

    noise = 0.01 * rng.standard_normal(n)
    out = r + noise
    m = np.max(np.abs(out)) + 1e-12
    return out / m


def convolve_same(source: np.ndarray, response: np.ndarray) -> np.ndarray:
    s = np.asarray(source, dtype=np.float64)
    r = np.asarray(response, dtype=np.float64)
    n = int(r.size)
    n_conv = s.size + r.size - 1
    nfft = 1 << int((n_conv - 1).bit_length())
    y = np.fft.irfft(np.fft.rfft(s, nfft) * np.fft.rfft(r, nfft), nfft).real[:n_conv]
    start = (s.size - 1) // 2
    end = start + n
    return y[start:end]


def make_synthetic_batch(
    traces: int,
    samples: int,
    dt: float,
    wavelet_hz: float = 0.8,
    noise_std: float = 0.015,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(rng_seed))
    src = ricker_wavelet(samples=samples, dt=dt, freq_hz=wavelet_hz)
    responses = np.zeros((int(traces), int(samples)), dtype=np.float64)
    observed = np.zeros_like(responses)

    for i in range(int(traces)):
        resp = make_response(samples=samples, allow_negative=True, rng_seed=int(rng.integers(0, 2**31 - 1)))
        obs = convolve_same(src, resp)
        obs = obs + float(noise_std) * rng.standard_normal(obs.shape[0])
        responses[i] = resp
        observed[i] = obs

    return src, responses, observed


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    sx = np.std(x)
    sy = np.std(y)
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    return float(math.sqrt(np.mean((x - y) ** 2)))
