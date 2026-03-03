from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from vertirf.filters.zero_phase import FilterSpec, build_zero_phase_response


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
    nfft: int
    decon_filter_full: np.ndarray
    source_fft: np.ndarray
    source_filt_fft: np.ndarray
    source_filt_energy: float
    pred_kernel: np.ndarray
    maxlag: int


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


def _phase_shift_prompt22(x: np.ndarray, nfft: int, dt: float, tshift_sec: float) -> np.ndarray:
    xf = np.fft.fft(np.asarray(x, dtype=np.float64), nfft)
    shift_i = int(float(tshift_sec) / float(dt))
    phase = 2.0 * np.pi * np.arange(1, nfft + 1, dtype=np.float64) * shift_i / float(nfft)
    xf = xf * (np.cos(phase) - 1j * np.sin(phase))
    y = np.fft.ifft(xf, nfft) / np.cos(2.0 * np.pi * shift_i / float(nfft))
    return y.real


def _build_decon_filter_full_fft(nfft: int, dt: float, spec: FilterSpec) -> np.ndarray:
    amp = build_zero_phase_response(nfft=nfft, dt=float(dt), spec=spec)
    half = int(0.5 * nfft + 1)
    out = np.zeros((nfft,), dtype=np.float64)
    out[:half] = amp / float(dt)
    if nfft > 2:
        out[half:] = np.flip(out[1 : half - 1])
    return out


def _gfilter(x: np.ndarray, nfft: int, filt: np.ndarray, dt: float) -> np.ndarray:
    xf = np.fft.fft(np.asarray(x, dtype=np.float64), nfft)
    xf = xf * np.asarray(filt, dtype=np.complex128) * float(dt)
    return np.fft.ifft(xf, nfft).real


def _correl_with_prepared_fft(r: np.ndarray, source_filt_fft: np.ndarray, nfft: int) -> np.ndarray:
    return np.fft.ifft(np.fft.fft(r, nfft) * np.conj(source_filt_fft), nfft).real


def _accumulate_shifted_circular_inplace(dst: np.ndarray, src: np.ndarray, shift: int, scale: float) -> None:
    n = int(dst.size)
    s = int(shift) % n
    if s == 0:
        dst += float(scale) * src
        return

    n_tail = n - s
    if n_tail > 0:
        dst[s:] += float(scale) * src[:n_tail]
    if s > 0:
        dst[:s] += float(scale) * src[n_tail:]


def _pick_window_index(corr: np.ndarray, allow_negative_impulse: bool, maxlag: int) -> int:
    if bool(allow_negative_impulse):
        return int(np.argmax(np.abs(corr)))
    hi = max(1, int(maxlag) - 1)
    return int(np.argmax(np.abs(corr[:hi])))


def _prepare_state(source: np.ndarray, cfg: DeconConfig) -> _PreparedState:
    src = np.asarray(source, dtype=np.float64)
    nt = int(src.size)
    nfft = next_pow_2(nt)

    src0 = np.zeros((nfft,), dtype=np.float64)
    src0[:nt] = src

    decon_f = _build_decon_filter_full_fft(nfft=nfft, dt=float(cfg.dt), spec=cfg.filter_spec)
    src_filt = _gfilter(src0, nfft, decon_f, cfg.dt)
    src_filt_fft = np.fft.fft(src_filt, nfft)
    src_filt_energy = float(np.sum(src_filt**2))

    src_fft = np.fft.fft(src0, nfft)
    unit_impulse = np.zeros((nfft,), dtype=np.float64)
    unit_impulse[0] = 1.0
    pred_kernel = _gfilter(unit_impulse, nfft, decon_f, cfg.dt)
    pred_kernel = _gfilter(pred_kernel, nfft, src_fft, cfg.dt)

    return _PreparedState(
        nt=nt,
        dt=float(cfg.dt),
        nfft=nfft,
        decon_filter_full=decon_f,
        source_fft=src_fft,
        source_filt_fft=src_filt_fft,
        source_filt_energy=src_filt_energy,
        pred_kernel=pred_kernel,
        maxlag=int(0.5 * nfft),
    )


def _decon_core_fast(observed: np.ndarray, prepared: _PreparedState, cfg: DeconConfig) -> DeconResult:
    u = np.asarray(observed, dtype=np.float64)
    if u.size != prepared.nt:
        raise ValueError("observed length must match source length")

    if prepared.source_filt_energy <= 0.0 or (not math.isfinite(prepared.source_filt_energy)):
        raise RuntimeError("invalid source power in decon")

    u0 = np.zeros((prepared.nfft,), dtype=np.float64)
    u0[: prepared.nt] = u
    u_flt = _gfilter(u0, prepared.nfft, prepared.decon_filter_full, prepared.dt)

    residual = u_flt.copy()
    impulse = np.zeros((prepared.nfft,), dtype=np.float64)
    power_u = float(np.sum(u_flt**2))
    if power_u <= 0.0:
        raise RuntimeError("invalid source power in decon")

    rms = np.zeros(max(1, int(cfg.itmax)), dtype=np.float64)
    prev_sumsq = 1.0
    d_error = 100.0 * power_u + float(cfg.minderr)
    it = 0

    while abs(d_error) > float(cfg.minderr) and it < int(cfg.itmax):
        rw = _correl_with_prepared_fft(residual, prepared.source_filt_fft, prepared.nfft)
        rw = rw / prepared.source_filt_energy
        idx = _pick_window_index(
            rw,
            allow_negative_impulse=bool(cfg.allow_negative_impulse),
            maxlag=int(prepared.maxlag),
        )

        amp = float(rw[idx] / prepared.dt)
        impulse[idx] += amp
        _accumulate_shifted_circular_inplace(residual, prepared.pred_kernel, shift=int(idx), scale=-amp)

        sumsq = float(np.sum(residual * residual) / power_u)
        if not math.isfinite(sumsq):
            break

        rms[it] = sumsq
        d_error = 100.0 * (prev_sumsq - sumsq)
        prev_sumsq = sumsq
        it += 1

    rf = _gfilter(impulse, prepared.nfft, prepared.decon_filter_full, prepared.dt)
    rf = _phase_shift_prompt22(rf, prepared.nfft, prepared.dt, cfg.tshift_sec)
    rf = rf[: prepared.nt]
    rf, amp = normalize_max_abs(rf)
    return DeconResult(
        rf=rf,
        rms_history=rms[: max(0, it - 1)],
        iterations=it,
        success=bool(amp > 0.0 and math.isfinite(amp)),
    )


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
            try:
                idx, res = _job(i)
                out[idx] = res.rf
                ok[idx] = res.success
                iters[idx] = int(res.iterations)
            except Exception:
                ok[i] = False
                iters[i] = 0
        return out, ok, iters

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_job, i) for i in range(n_traces)]
        for i, fut in enumerate(futures):
            try:
                idx, res = fut.result()
                out[idx] = res.rf
                ok[idx] = res.success
                iters[idx] = int(res.iterations)
            except Exception:
                ok[i] = False
                iters[i] = 0

    return out, ok, iters
