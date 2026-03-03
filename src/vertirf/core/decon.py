from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

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


def _correlate_same_fft(a: np.ndarray, corr_fft_kernel: np.ndarray, nfft_conv: int) -> np.ndarray:
    n = int(a.size)
    full = np.fft.irfft(np.fft.rfft(a, nfft_conv) * corr_fft_kernel, nfft_conv).real[: (2 * n - 1)]
    start = (n - 1) // 2
    return full[start : start + n]


def _accumulate_shifted_same_inplace(dst: np.ndarray, src: np.ndarray, shift: int, scale: float) -> None:
    n = int(dst.size)
    s = int(shift)
    if s >= 0:
        n_tail = n - s
        if n_tail > 0:
            dst[s:] += float(scale) * src[:n_tail]
        return

    n_head = n + s
    if n_head > 0:
        dst[:n_head] += float(scale) * src[-s:]


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

    return _PreparedState(
        nt=nt,
        dt=float(cfg.dt),
        src_filt=src_filt,
        denom=denom,
        corr_fft_kernel=corr_fft_kernel,
        nfft_conv=nfft_conv,
    )


def _decon_core_fast(observed: np.ndarray, prepared: _PreparedState, cfg: DeconConfig) -> DeconResult:
    u = np.asarray(observed, dtype=np.float64)
    if u.size != prepared.nt:
        raise ValueError("observed length must match source length")

    obs = apply_zero_phase_filter(u, cfg.dt, cfg.filter_spec)
    residual = obs.copy()
    impulse = np.zeros(prepared.nt, dtype=np.float64)
    power_obs = float(np.sum(obs * obs)) + 1e-12
    rms = np.zeros(max(1, int(cfg.itmax)), dtype=np.float64)

    center = (prepared.nt - 1) // 2
    prev_sumsq = 1.0
    d_error = 100.0 * power_obs + float(cfg.minderr)
    it = 0

    while abs(d_error) > float(cfg.minderr) and it < int(cfg.itmax):
        corr_same = _correlate_same_fft(residual, prepared.corr_fft_kernel, prepared.nfft_conv)
        idx = _pick_window_index(corr_same, allow_negative_impulse=bool(cfg.allow_negative_impulse))

        amp = corr_same[idx] / prepared.denom
        impulse[idx] += amp

        shift = int(idx) - int(center)
        _accumulate_shifted_same_inplace(residual, prepared.src_filt, shift=shift, scale=-float(amp))

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


def run_batch_decon(
    observed: np.ndarray,
    source: np.ndarray,
    cfg: DeconConfig,
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

    prepared = _prepare_state(src, cfg)

    def _job(i: int) -> tuple[int, DeconResult]:
        return i, _decon_core_fast(arr[i], prepared, cfg)

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
