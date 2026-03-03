from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np

from vertirf.core.decon import DeconConfig, normalize_max_abs, run_batch_decon
from vertirf.filters.zero_phase import (
    FilterSpec,
    apply_zero_phase_filter,
    build_zero_phase_response,
)

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

    # Corr-specific options (prompt22-compatible semantics).
    corr_smoothing_bandwidth_hz: float = 0.2
    corr_divide_denom: bool = True
    corr_water_level: float = 1e-4
    corr_shift_sec: float = 0.0
    corr_post_filter_type: str = "none"
    corr_post_gauss_f0: float = math.pi
    corr_post_low_hz: float = 0.1
    corr_post_high_hz: float = 0.8
    corr_post_corners: int = 4
    corr_fft_switch_samples: int = 8192

    # Stack-specific options.
    stack_peak_window_start_sec: float = -2.0
    stack_peak_window_end_sec: float = 20.0
    stack_zero_index: int | None = None


@dataclass(frozen=True)
class MethodPreparedState:
    method: MethodName
    n_samples: int
    dt: float

    source_fft: np.ndarray
    source_fft_conj: np.ndarray
    corr_denom_water: np.ndarray
    corr_filter_response: np.ndarray
    corr_shift_samples: int


def _to_decon_config(cfg: MethodConfig) -> DeconConfig:
    return DeconConfig(
        dt=float(cfg.dt),
        tshift_sec=float(cfg.tshift_sec),
        itmax=int(cfg.itmax),
        minderr=float(cfg.minderr),
        allow_negative_impulse=bool(cfg.allow_negative_impulse),
        filter_spec=cfg.filter_spec,
    )


def _corr_smooth_spectrum_edge_comp(power: np.ndarray, win_len: int) -> np.ndarray:
    p = np.asarray(power, dtype=np.float64)
    if win_len <= 1:
        return p
    w = np.ones((int(win_len),), dtype=np.float64) / float(win_len)
    wei = np.convolve(np.ones((p.size,), dtype=np.float64), w, mode="same")
    smo = np.convolve(p, w, mode="same")
    return smo / np.maximum(wei, 1e-12)


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
    dt = float(cfg.dt)

    source_fft = np.fft.rfft(src, n=n)
    source_pow = np.abs(source_fft) ** 2

    df = 1.0 / max(float(n) * dt, 1e-12)
    half_bins = int(round(max(0.0, float(cfg.corr_smoothing_bandwidth_hz)) / max(df, 1e-12)))
    win_len = max(1, 2 * half_bins + 1)
    denom = _corr_smooth_spectrum_edge_comp(source_pow, win_len)
    denom = np.maximum(denom, 1e-12)
    denom_max = float(np.max(denom)) if denom.size else 0.0
    water_floor = max(1e-12, denom_max * max(0.0, float(cfg.corr_water_level)))
    denom_water = np.maximum(denom, water_floor)

    corr_filter_response = build_zero_phase_response(nfft=n, dt=dt, spec=cfg.filter_spec)
    corr_shift_samples = int(round(float(cfg.corr_shift_sec) / dt))

    return MethodPreparedState(
        method=cfg.method,
        n_samples=n,
        dt=dt,
        source_fft=np.asarray(source_fft, dtype=np.complex128),
        source_fft_conj=np.asarray(np.conj(source_fft), dtype=np.complex128),
        corr_denom_water=np.asarray(denom_water, dtype=np.float64),
        corr_filter_response=np.asarray(corr_filter_response, dtype=np.float64),
        corr_shift_samples=int(corr_shift_samples),
    )


def _normalize_rows_max_abs(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    amps = np.max(np.abs(arr), axis=1)
    valid = amps > 0.0
    out = arr.copy()
    out[valid] = out[valid] / amps[valid, None]
    out[~valid] = 0.0
    return out


def _run_corr_single_prompt22(
    observed_row: np.ndarray,
    cfg: MethodConfig,
    prepared: MethodPreparedState,
    post_filter_spec: FilterSpec | None,
) -> np.ndarray:
    row = np.asarray(observed_row, dtype=np.float64)
    d = np.fft.rfft(row, n=prepared.n_samples)
    rf_spec = d * prepared.source_fft_conj

    if bool(cfg.corr_divide_denom):
        rf_spec = rf_spec / prepared.corr_denom_water

    rf_spec = rf_spec * prepared.corr_filter_response
    rf = np.fft.irfft(rf_spec, n=prepared.n_samples).real

    if int(prepared.corr_shift_samples) != 0:
        rf = np.roll(rf, int(prepared.corr_shift_samples))

    if post_filter_spec is not None:
        rf = apply_zero_phase_filter(rf, cfg.dt, post_filter_spec)

    rf, _ = normalize_max_abs(rf)
    return rf


def _run_corr_rows_prompt22(
    observed: np.ndarray,
    cfg: MethodConfig,
    prepared: MethodPreparedState,
    jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(observed, dtype=np.float64)
    n_traces = int(arr.shape[0])
    n_samples = int(arr.shape[1])

    out = np.zeros((n_traces, n_samples), dtype=np.float64)
    ok = np.zeros((n_traces,), dtype=bool)
    steps = np.ones((n_traces,), dtype=np.int32)

    post = _corr_post_filter_spec(cfg)
    n_jobs = max(1, int(jobs))

    if n_jobs == 1 or n_traces < 2:
        for i in range(n_traces):
            y = _run_corr_single_prompt22(arr[i], cfg, prepared, post)
            out[i] = y
            ok[i] = bool(np.isfinite(y).all())
        return out, ok, steps

    def _job(i: int) -> tuple[int, np.ndarray]:
        y = _run_corr_single_prompt22(arr[i], cfg, prepared, post)
        return i, y

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_job, i) for i in range(n_traces)]
        for fut in futures:
            i, y = fut.result()
            out[i] = y
            ok[i] = bool(np.isfinite(y).all())

    return out, ok, steps


def _stack_peak_window_indices(
    n: int,
    dt: float,
    t0: float,
    t1: float,
    zero_index: int | None,
) -> np.ndarray:
    if zero_index is None:
        t = (np.arange(n, dtype=np.float64) - 0.5 * (n - 1)) * float(dt)
    else:
        t = (np.arange(n, dtype=np.float64) - float(int(zero_index))) * float(dt)
    mask = (t >= float(t0)) & (t <= float(t1))
    idx = np.where(mask)[0]
    if idx.size < 2:
        return np.arange(n, dtype=np.int64)
    return idx


def _run_stack_single_prompt22(
    observed_row: np.ndarray,
    filter_response: np.ndarray,
    n_samples: int,
    peak_idx: np.ndarray,
    target_index: int,
) -> np.ndarray:
    row = np.asarray(observed_row, dtype=np.float64)
    xf = np.fft.rfft(row, n=n_samples)
    x = np.fft.irfft(xf * filter_response, n=n_samples).real

    i = int(peak_idx[np.argmax(np.abs(x[peak_idx]))])
    peak_amp = float(x[i])

    y = np.roll(x, int(target_index - i))
    if peak_amp < 0.0:
        y = -y
    y, _ = normalize_max_abs(y)
    return y


def _run_stack_rows_prompt22(
    observed: np.ndarray,
    cfg: MethodConfig,
    jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(observed, dtype=np.float64)
    n_traces, n_samples = arr.shape
    out = np.zeros((n_traces, n_samples), dtype=np.float64)
    ok = np.zeros((n_traces,), dtype=bool)
    steps = np.ones((n_traces,), dtype=np.int32)

    target_index = int(cfg.stack_zero_index) if cfg.stack_zero_index is not None else int(n_samples // 2)
    if target_index < 0 or target_index >= n_samples:
        raise ValueError(f"stack_zero_index out of range: {target_index} for n_samples={n_samples}")
    peak_idx = _stack_peak_window_indices(
        n=n_samples,
        dt=cfg.dt,
        t0=cfg.stack_peak_window_start_sec,
        t1=cfg.stack_peak_window_end_sec,
        zero_index=cfg.stack_zero_index,
    )
    filter_response = build_zero_phase_response(nfft=n_samples, dt=float(cfg.dt), spec=cfg.filter_spec)
    n_jobs = max(1, int(jobs))

    if n_jobs == 1 or n_traces < 2:
        for i in range(n_traces):
            y = _run_stack_single_prompt22(arr[i], filter_response, n_samples, peak_idx, target_index)
            out[i] = y
            ok[i] = bool(np.isfinite(y).all())
        return out, ok, steps

    def _job(i: int) -> tuple[int, np.ndarray]:
        y = _run_stack_single_prompt22(arr[i], filter_response, n_samples, peak_idx, target_index)
        return i, y

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_job, i) for i in range(n_traces)]
        for fut in futures:
            i, y = fut.result()
            out[i] = y
            ok[i] = bool(np.isfinite(y).all())

    return out, ok, steps


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
        return run_batch_decon(arr, src, _to_decon_config(cfg), jobs=jobs)

    n_jobs = max(1, int(jobs))

    if method == "corr":
        prepared = prepare_method_state(src, cfg)
        return _run_corr_rows_prompt22(arr, cfg, prepared, jobs=n_jobs)

    if method == "stack":
        return _run_stack_rows_prompt22(arr, cfg, jobs=n_jobs)

    raise ValueError(f"unsupported method: {cfg.method}")
