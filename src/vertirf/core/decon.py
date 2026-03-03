from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from vertirf.filters.zero_phase import FilterSpec, apply_zero_phase_filter


@dataclass(frozen=True)
class DeconConfig:
    dt: float = 0.05
    tshift_sec: float = 0.0
    itmax: int = 400
    minderr: float = 1e-3
    allow_negative_impulse: bool = False
    filter_spec: FilterSpec = FilterSpec()


@dataclass
class DeconResult:
    rf: np.ndarray
    rms_history: np.ndarray
    iterations: int
    success: bool


@dataclass(frozen=True)
class _PreparedState:
    nt: int
    dt: float
    src_filt: np.ndarray
    denom: float
    corr_fft_kernel: np.ndarray
    conv_src_fft: np.ndarray
    nfft_conv: int


def next_pow_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << int((n - 1).bit_length())


def normalize_max_abs(x: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(x, dtype=np.float64)
    amp = float(np.max(np.abs(arr))) if arr.size else 0.0
    if (not math.isfinite(amp)) or amp <= 0.0:
        return arr, amp
    return arr / amp, amp


def _phase_shift_roll(x: np.ndarray, dt: float, tshift_sec: float) -> np.ndarray:
    shift = int(round(float(tshift_sec) / float(dt)))
    return np.roll(np.asarray(x, dtype=np.float64), shift)


def _convolve_same_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.convolve(a, b, mode="same")


def _convolve_same_fft(a: np.ndarray, b: np.ndarray, nfft_conv: int, b_fft: np.ndarray) -> np.ndarray:
    n = int(a.size)
    full = np.fft.irfft(np.fft.rfft(a, nfft_conv) * b_fft, nfft_conv).real[: (2 * n - 1)]
    start = (n - 1) // 2
    return full[start : start + n]


def _correlate_same_direct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.correlate(a, b, mode="same")


def _correlate_same_fft(a: np.ndarray, corr_fft_kernel: np.ndarray, nfft_conv: int) -> np.ndarray:
    n = int(a.size)
    full = np.fft.irfft(np.fft.rfft(a, nfft_conv) * corr_fft_kernel, nfft_conv).real[: (2 * n - 1)]
    start = (n - 1) // 2
    return full[start : start + n]


def _pick_window_index(corr_same: np.ndarray, allow_negative_impulse: bool) -> int:
    n = int(corr_same.size)
    center = n // 2
    if allow_negative_impulse:
        return int(np.argmax(np.abs(corr_same)))
    return int(center + np.argmax(np.abs(corr_same[center:])))


def _prepare_state(source: np.ndarray, cfg: DeconConfig) -> _PreparedState:
    src = np.asarray(source, dtype=np.float64)
    nt = int(src.size)
    src_filt = apply_zero_phase_filter(src, cfg.dt, cfg.filter_spec)
    src_filt, _ = normalize_max_abs(src_filt)

    denom = float(np.sum(src_filt * src_filt)) + 1e-12
    nfft_conv = next_pow_2(2 * nt - 1)

    corr_kernel = np.flip(src_filt)
    corr_fft_kernel = np.fft.rfft(corr_kernel, nfft_conv)
    conv_src_fft = np.fft.rfft(src_filt, nfft_conv)

    return _PreparedState(
        nt=nt,
        dt=float(cfg.dt),
        src_filt=src_filt,
        denom=denom,
        corr_fft_kernel=corr_fft_kernel,
        conv_src_fft=conv_src_fft,
        nfft_conv=nfft_conv,
    )


def _decon_core_baseline(observed: np.ndarray, prepared: _PreparedState, cfg: DeconConfig) -> DeconResult:
    u = np.asarray(observed, dtype=np.float64)
    if u.size != prepared.nt:
        raise ValueError("observed length must match source length")

    obs = apply_zero_phase_filter(u, cfg.dt, cfg.filter_spec)
    residual = obs.copy()
    impulse = np.zeros(prepared.nt, dtype=np.float64)
    power_obs = float(np.sum(obs * obs)) + 1e-12
    rms = np.zeros(max(1, int(cfg.itmax)), dtype=np.float64)

    prev_sumsq = 1.0
    d_error = 100.0 * power_obs + float(cfg.minderr)
    it = 0

    while abs(d_error) > float(cfg.minderr) and it < int(cfg.itmax):
        corr_same = _correlate_same_direct(residual, prepared.src_filt)
        idx = _pick_window_index(corr_same, allow_negative_impulse=bool(cfg.allow_negative_impulse))

        amp = corr_same[idx] / prepared.denom
        impulse[idx] += amp

        pred = _convolve_same_direct(impulse, prepared.src_filt)
        residual = obs - pred

        sumsq = float(np.sum(residual * residual) / power_obs)
        if not math.isfinite(sumsq):
            break

        rms[it] = sumsq
        d_error = 100.0 * (prev_sumsq - sumsq)
        prev_sumsq = sumsq
        it += 1

    rf = _phase_shift_roll(impulse, dt=cfg.dt, tshift_sec=cfg.tshift_sec)
    rf, amp = normalize_max_abs(rf)
    return DeconResult(rf=rf, rms_history=rms[: max(0, it)], iterations=it, success=bool(amp > 0.0 and math.isfinite(amp)))


def _decon_core_optimized(observed: np.ndarray, prepared: _PreparedState, cfg: DeconConfig) -> DeconResult:
    u = np.asarray(observed, dtype=np.float64)
    if u.size != prepared.nt:
        raise ValueError("observed length must match source length")

    obs = apply_zero_phase_filter(u, cfg.dt, cfg.filter_spec)
    residual = obs.copy()
    impulse = np.zeros(prepared.nt, dtype=np.float64)
    power_obs = float(np.sum(obs * obs)) + 1e-12
    rms = np.zeros(max(1, int(cfg.itmax)), dtype=np.float64)

    prev_sumsq = 1.0
    d_error = 100.0 * power_obs + float(cfg.minderr)
    it = 0

    while abs(d_error) > float(cfg.minderr) and it < int(cfg.itmax):
        corr_same = _correlate_same_fft(residual, prepared.corr_fft_kernel, prepared.nfft_conv)
        idx = _pick_window_index(corr_same, allow_negative_impulse=bool(cfg.allow_negative_impulse))

        amp = corr_same[idx] / prepared.denom
        impulse[idx] += amp

        pred = _convolve_same_fft(impulse, prepared.src_filt, prepared.nfft_conv, prepared.conv_src_fft)
        residual = obs - pred

        sumsq = float(np.sum(residual * residual) / power_obs)
        if not math.isfinite(sumsq):
            break

        rms[it] = sumsq
        d_error = 100.0 * (prev_sumsq - sumsq)
        prev_sumsq = sumsq
        it += 1

    rf = _phase_shift_roll(impulse, dt=cfg.dt, tshift_sec=cfg.tshift_sec)
    rf, amp = normalize_max_abs(rf)
    return DeconResult(rf=rf, rms_history=rms[: max(0, it)], iterations=it, success=bool(amp > 0.0 and math.isfinite(amp)))


def deconit_baseline(observed: np.ndarray, source: np.ndarray, cfg: DeconConfig) -> DeconResult:
    prepared = _prepare_state(source, cfg)
    return _decon_core_baseline(observed, prepared, cfg)


def deconit_optimized(observed: np.ndarray, prepared: _PreparedState, cfg: DeconConfig) -> DeconResult:
    return _decon_core_optimized(observed, prepared, cfg)


def run_batch_decon(
    observed: np.ndarray,
    source: np.ndarray,
    cfg: DeconConfig,
    mode: Literal["baseline", "optimized"] = "optimized",
    jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(observed, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("observed must be 2D: [n_traces, n_samples]")

    n_traces, n_samples = arr.shape
    src = np.asarray(source, dtype=np.float64)
    if src.ndim != 1 or src.size != n_samples:
        raise ValueError("source must be 1D and same sample count as observed traces")

    out = np.zeros_like(arr, dtype=np.float64)
    ok = np.zeros((n_traces,), dtype=bool)
    iters = np.zeros((n_traces,), dtype=np.int32)

    if mode == "baseline":
        for i in range(n_traces):
            res = deconit_baseline(arr[i], src, cfg)
            out[i] = res.rf
            ok[i] = res.success
            iters[i] = int(res.iterations)
        return out, ok, iters

    prepared = _prepare_state(src, cfg)

    def _job(i: int) -> tuple[int, DeconResult]:
        return i, deconit_optimized(arr[i], prepared, cfg)

    n_jobs = max(1, int(jobs))
    if n_jobs == 1:
        for i in range(n_traces):
            idx, res = _job(i)
            out[idx] = res.rf
            ok[idx] = res.success
            iters[idx] = int(res.iterations)
        return out, ok, iters

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_job, i) for i in range(n_traces)]
        for fut in futures:
            idx, res = fut.result()
            out[idx] = res.rf
            ok[idx] = res.success
            iters[idx] = int(res.iterations)

    return out, ok, iters
