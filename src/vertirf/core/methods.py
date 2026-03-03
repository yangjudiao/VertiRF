from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

from vertirf.core.decon import DeconConfig, normalize_max_abs, run_batch_decon
from vertirf.filters.zero_phase import FilterSpec, apply_zero_phase_filter
from vertirf.native.backend import CorrNativeBackend, load_native_corr_backend

MethodName = Literal["decon", "corr", "stack"]
RunMode = Literal["baseline", "optimized"]


@dataclass(frozen=True)
class MethodConfig:
    method: MethodName = "decon"
    dt: float = 0.05
    tshift_sec: float = 0.0
    itmax: int = 400
    minderr: float = 1e-3
    allow_negative_impulse: bool = False
    filter_spec: FilterSpec = FilterSpec()

    # Corr-specific options.
    corr_smoothing_bandwidth_hz: float = 0.35
    corr_post_filter_type: str = "none"
    corr_post_gauss_f0: float = math.pi
    corr_post_low_hz: float = 0.1
    corr_post_high_hz: float = 0.8
    corr_post_corners: int = 4

    # Stack-specific options.
    stack_peak_window_start_sec: float = -2.0
    stack_peak_window_end_sec: float = 20.0


@dataclass(frozen=True)
class MethodPreparedState:
    method: MethodName
    n_samples: int
    dt: float

    source_filtered: np.ndarray
    source_energy: float

    corr_fft_kernel: np.ndarray
    nfft_conv: int
    corr_smoothing_response: np.ndarray


@lru_cache(maxsize=1)
def _native_corr_backend() -> CorrNativeBackend | None:
    backend, _ = load_native_corr_backend()
    return backend


def _to_decon_config(cfg: MethodConfig) -> DeconConfig:
    return DeconConfig(
        dt=float(cfg.dt),
        tshift_sec=float(cfg.tshift_sec),
        itmax=int(cfg.itmax),
        minderr=float(cfg.minderr),
        allow_negative_impulse=bool(cfg.allow_negative_impulse),
        filter_spec=cfg.filter_spec,
    )


def _same_fft_length(n: int) -> int:
    return 1 << int((2 * n - 2).bit_length())


def _correlate_same_fft(x: np.ndarray, kernel_fft: np.ndarray, nfft_conv: int) -> np.ndarray:
    n = int(x.size)
    full = np.fft.irfft(np.fft.rfft(x, nfft_conv) * kernel_fft, nfft_conv).real[: (2 * n - 1)]
    start = (n - 1) // 2
    return full[start : start + n]


def _corr_smoothing_response(n: int, dt: float, bandwidth_hz: float) -> np.ndarray:
    bw = float(bandwidth_hz)
    if bw <= 0.0:
        return np.ones((n // 2 + 1,), dtype=np.float64)
    f = np.fft.rfftfreq(n, d=float(dt))
    return np.exp(-0.5 * (f / max(1e-9, bw)) ** 2)


def _apply_corr_smoothing(x: np.ndarray, dt: float, bandwidth_hz: float) -> np.ndarray:
    n = int(x.size)
    resp = _corr_smoothing_response(n, dt, bandwidth_hz)
    return np.fft.irfft(np.fft.rfft(x, n) * resp, n)


def _corr_post_filter_spec(cfg: MethodConfig) -> FilterSpec | None:
    ftype = str(cfg.corr_post_filter_type).strip().lower()
    if ftype in {"", "none", "off", "false"}:
        return None
    return FilterSpec(
        filter_type=ftype,
        gauss_f0=float(cfg.corr_post_gauss_f0),
        low_hz=float(cfg.corr_post_low_hz),
        high_hz=float(cfg.corr_post_high_hz),
        corners=int(cfg.corr_post_corners),
        transition_hz=float(cfg.filter_spec.transition_hz),
        tukey_alpha=float(cfg.filter_spec.tukey_alpha),
    )


def prepare_method_state(source: np.ndarray, cfg: MethodConfig) -> MethodPreparedState:
    src = np.asarray(source, dtype=np.float64)
    n = int(src.size)
    src_f = apply_zero_phase_filter(src, cfg.dt, cfg.filter_spec)
    src_f, _ = normalize_max_abs(src_f)

    nfft_conv = _same_fft_length(n)
    corr_kernel = np.flip(src_f)
    corr_fft = np.fft.rfft(corr_kernel, nfft_conv)

    return MethodPreparedState(
        method=cfg.method,
        n_samples=n,
        dt=float(cfg.dt),
        source_filtered=src_f,
        source_energy=float(np.sum(src_f * src_f)) + 1e-12,
        corr_fft_kernel=corr_fft,
        nfft_conv=nfft_conv,
        corr_smoothing_response=_corr_smoothing_response(n, cfg.dt, cfg.corr_smoothing_bandwidth_hz),
    )


def _run_corr_single_baseline(obs: np.ndarray, cfg: MethodConfig, source_filtered: np.ndarray) -> np.ndarray:
    x = apply_zero_phase_filter(np.asarray(obs, dtype=np.float64), cfg.dt, cfg.filter_spec)
    cc = np.correlate(x, source_filtered, mode="same")
    cc = cc / (float(np.sum(source_filtered * source_filtered)) + 1e-12)
    cc = _apply_corr_smoothing(cc, cfg.dt, cfg.corr_smoothing_bandwidth_hz)

    post = _corr_post_filter_spec(cfg)
    if post is not None:
        cc = apply_zero_phase_filter(cc, cfg.dt, post)

    cc, _ = normalize_max_abs(cc)
    return cc


def _run_corr_single_optimized(
    obs: np.ndarray,
    cfg: MethodConfig,
    prepared: MethodPreparedState,
) -> np.ndarray:
    x = apply_zero_phase_filter(np.asarray(obs, dtype=np.float64), cfg.dt, cfg.filter_spec)
    native_backend = _native_corr_backend()
    if native_backend is not None:
        cc = native_backend.corr_same(x, prepared.source_filtered)
    else:
        cc = _correlate_same_fft(x, prepared.corr_fft_kernel, prepared.nfft_conv)
    cc = cc / prepared.source_energy

    if float(cfg.corr_smoothing_bandwidth_hz) > 0.0:
        n = cc.size
        cc = np.fft.irfft(np.fft.rfft(cc, n) * prepared.corr_smoothing_response, n)

    post = _corr_post_filter_spec(cfg)
    if post is not None:
        cc = apply_zero_phase_filter(cc, cfg.dt, post)

    cc, _ = normalize_max_abs(cc)
    return cc


def _stack_peak_window_indices(n: int, dt: float, t0: float, t1: float) -> np.ndarray:
    t = (np.arange(n, dtype=np.float64) - 0.5 * (n - 1)) * float(dt)
    mask = (t >= float(t0)) & (t <= float(t1))
    idx = np.where(mask)[0]
    if idx.size < 2:
        return np.arange(n, dtype=np.int64)
    return idx


def _run_stack_single(obs: np.ndarray, cfg: MethodConfig, peak_idx: np.ndarray) -> np.ndarray:
    x = apply_zero_phase_filter(np.asarray(obs, dtype=np.float64), cfg.dt, cfg.filter_spec)
    i = int(peak_idx[np.argmax(np.abs(x[peak_idx]))])

    sign = 1.0 if x[i] >= 0.0 else -1.0
    y = x * sign

    center = y.size // 2
    shift = center - i
    y = np.roll(y, shift)

    y, _ = normalize_max_abs(y)
    return y


def run_batch_method(
    observed: np.ndarray,
    source: np.ndarray,
    cfg: MethodConfig,
    mode: RunMode = "optimized",
    jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(observed, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("observed must be 2D: [n_traces, n_samples]")

    src = np.asarray(source, dtype=np.float64)
    if src.ndim != 1 or src.size != arr.shape[1]:
        raise ValueError("source must be 1D and same sample count as observed traces")

    method = str(cfg.method).lower().strip()
    if method == "decon":
        return run_batch_decon(arr, src, _to_decon_config(cfg), mode=mode, jobs=jobs)

    n_traces, n_samples = arr.shape
    out = np.zeros((n_traces, n_samples), dtype=np.float64)
    ok = np.zeros((n_traces,), dtype=bool)
    steps = np.zeros((n_traces,), dtype=np.int32)

    n_jobs = max(1, int(jobs))

    if method == "corr":
        prepared = prepare_method_state(src, cfg)

        def _job(i: int) -> tuple[int, np.ndarray]:
            if mode == "baseline":
                y = _run_corr_single_baseline(arr[i], cfg, prepared.source_filtered)
            else:
                y = _run_corr_single_optimized(arr[i], cfg, prepared)
            return i, y

    elif method == "stack":
        peak_idx = _stack_peak_window_indices(
            n=src.size,
            dt=cfg.dt,
            t0=cfg.stack_peak_window_start_sec,
            t1=cfg.stack_peak_window_end_sec,
        )

        def _job(i: int) -> tuple[int, np.ndarray]:
            y = _run_stack_single(arr[i], cfg, peak_idx)
            return i, y

    else:
        raise ValueError(f"unsupported method: {cfg.method}")

    if n_jobs == 1:
        for i in range(n_traces):
            idx, y = _job(i)
            out[idx] = y
            ok[idx] = bool(np.isfinite(y).all())
            steps[idx] = 1
        return out, ok, steps

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_job, i) for i in range(n_traces)]
        for fut in futures:
            idx, y = fut.result()
            out[idx] = y
            ok[idx] = bool(np.isfinite(y).all())
            steps[idx] = 1

    return out, ok, steps
