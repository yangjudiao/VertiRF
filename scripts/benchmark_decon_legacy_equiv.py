from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vertirf.core.decon import DeconConfig, run_batch_decon
from vertirf.filters.zero_phase import FilterSpec, build_zero_phase_response
from vertirf.waveform.synthetic import make_synthetic_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark legacy-compatible decon reference vs optimized single engine")
    p.add_argument("--out", type=Path, default=Path("benchmark_decon_legacy_equiv.json"))
    p.add_argument("--traces", type=int, default=120)
    p.add_argument("--samples", type=int, default=2001)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--itmax", type=int, default=1200)
    p.add_argument("--minderr", type=float, default=1e-3)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--allow-negative-impulse", action="store_true")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--tshift-sec", type=float, default=10.0)
    return p.parse_args()


def _next_pow_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << int((n - 1).bit_length())


def _build_decon_filter_full(nfft: int, dt: float, spec: FilterSpec) -> np.ndarray:
    amp = build_zero_phase_response(nfft=nfft, dt=dt, spec=spec)
    half = int(0.5 * nfft + 1)
    out = np.zeros((nfft,), dtype=np.float64)
    out[:half] = amp / float(dt)
    if nfft > 2:
        out[half:] = np.flip(out[1 : half - 1])
    return out


def _gfilter(x: np.ndarray, nfft: int, filt: np.ndarray, dt: float) -> np.ndarray:
    xf = np.fft.fft(np.asarray(x, dtype=np.float64), nfft)
    xf = xf * np.asarray(filt) * float(dt)
    return np.fft.ifft(xf, nfft).real


def _correl(r: np.ndarray, w: np.ndarray, nfft: int) -> np.ndarray:
    return np.fft.ifft(np.fft.fft(r, nfft) * np.conj(np.fft.fft(w, nfft)), nfft).real


def _phase_shift_legacy(x: np.ndarray, nfft: int, dt: float, tshift: float) -> np.ndarray:
    xf = np.fft.fft(x, nfft)
    shift_i = int(float(tshift) / float(dt))
    p = 2.0 * np.pi * np.arange(1, nfft + 1, dtype=np.float64) * shift_i / float(nfft)
    xf = xf * (np.cos(p) - 1j * np.sin(p))
    y = np.fft.ifft(xf, nfft) / np.cos(2.0 * np.pi * shift_i / float(nfft))
    return y.real


def _normalize_max_abs(x: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(x, dtype=np.float64)
    amp = float(np.max(np.abs(arr))) if arr.size else 0.0
    if amp <= 0.0:
        return arr, amp
    return arr / amp, amp


def _legacy_decon_single(observed: np.ndarray, source: np.ndarray, cfg: DeconConfig) -> tuple[np.ndarray, int, bool]:
    nt = int(observed.size)
    nfft = _next_pow_2(nt)

    p0 = np.zeros((nfft,), dtype=np.float64)
    u0 = np.zeros((nfft,), dtype=np.float64)
    w0 = np.zeros((nfft,), dtype=np.float64)
    u0[:nt] = observed
    w0[:nt] = source

    decon_f = _build_decon_filter_full(nfft=nfft, dt=cfg.dt, spec=cfg.filter_spec)
    u_flt = _gfilter(u0, nfft, decon_f, cfg.dt)
    w_flt = _gfilter(w0, nfft, decon_f, cfg.dt)
    wf = np.fft.fft(w0, nfft)

    r_flt = u_flt
    power_u = float(np.sum(u_flt**2))
    if power_u <= 0.0:
        return np.zeros((nt,), dtype=np.float64), 0, False

    it = 0
    sumsq_i = 1.0
    d_error = 100.0 * power_u + float(cfg.minderr)
    maxlag = int(0.5 * nfft)
    denom = float(np.sum(w_flt**2))

    while abs(d_error) > float(cfg.minderr) and it < int(cfg.itmax):
        rw = _correl(r_flt, w_flt, nfft)
        rw = rw / denom
        if bool(cfg.allow_negative_impulse):
            i1 = int(np.argmax(np.abs(rw)))
        else:
            i1 = int(np.argmax(np.abs(rw[0 : int(maxlag) - 1])))
        amp = float(rw[i1] / cfg.dt)
        p0[i1] = p0[i1] + amp

        p_flt = _gfilter(p0, nfft, decon_f, cfg.dt)
        p_flt = _gfilter(p_flt, nfft, wf, cfg.dt)
        r_flt = u_flt - p_flt

        sumsq = float(np.sum(r_flt**2) / power_u)
        d_error = 100.0 * (sumsq_i - sumsq)
        sumsq_i = sumsq
        it += 1

    p_flt = _gfilter(p0, nfft, decon_f, cfg.dt)
    p_flt = _phase_shift_legacy(p_flt, nfft, cfg.dt, cfg.tshift_sec)
    rf, amp = _normalize_max_abs(p_flt[:nt])
    return rf, it, bool(amp > 0.0)


def _run_legacy_batch(obs: np.ndarray, src: np.ndarray, cfg: DeconConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.zeros_like(obs, dtype=np.float64)
    ok = np.zeros((obs.shape[0],), dtype=bool)
    iters = np.zeros((obs.shape[0],), dtype=np.int32)
    for i in range(obs.shape[0]):
        rf, it, succ = _legacy_decon_single(obs[i], src, cfg)
        out[i] = rf
        ok[i] = succ
        iters[i] = int(it)
    return out, ok, iters


def main() -> int:
    args = parse_args()

    src, _, obs = make_synthetic_batch(
        traces=int(args.traces),
        samples=int(args.samples),
        dt=float(args.dt),
        rng_seed=int(args.seed),
    )

    cfg = DeconConfig(
        dt=float(args.dt),
        tshift_sec=float(args.tshift_sec),
        itmax=int(args.itmax),
        minderr=float(args.minderr),
        allow_negative_impulse=bool(args.allow_negative_impulse),
        filter_spec=FilterSpec(
            filter_type="butterworth_bandpass",
            gauss_f0=np.pi,
            low_hz=0.1,
            high_hz=0.8,
            corners=4,
            transition_hz=0.05,
            tukey_alpha=0.3,
        ),
    )

    legacy_times = []
    fast_times = []
    legacy_last = None
    fast_last = None

    for _ in range(max(1, int(args.repeat))):
        t0 = time.perf_counter()
        legacy_last = _run_legacy_batch(obs, src, cfg)
        legacy_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        fast_last = run_batch_decon(obs, src, cfg, jobs=max(1, int(args.jobs)))
        fast_times.append(time.perf_counter() - t0)

    legacy_rec, legacy_ok, legacy_it = legacy_last
    fast_rec, fast_ok, fast_it = fast_last

    diff = fast_rec - legacy_rec
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    flat_cc = float(np.corrcoef(fast_rec.ravel(), legacy_rec.ravel())[0, 1])

    iter_equal = int(np.count_nonzero(fast_it == legacy_it))

    legacy_mean = float(np.mean(np.asarray(legacy_times, dtype=np.float64)))
    fast_mean = float(np.mean(np.asarray(fast_times, dtype=np.float64)))
    speedup = legacy_mean / max(1e-12, fast_mean)

    summary = {
        "benchmark": {
            "traces": int(args.traces),
            "samples": int(args.samples),
            "dt": float(args.dt),
            "itmax": int(args.itmax),
            "minderr": float(args.minderr),
            "repeat": int(max(1, int(args.repeat))),
            "jobs": int(max(1, int(args.jobs))),
            "seed": int(args.seed),
            "allow_negative_impulse": bool(args.allow_negative_impulse),
            "tshift_sec": float(args.tshift_sec),
        },
        "legacy_reference": {
            "elapsed_sec": legacy_times,
            "elapsed_sec_mean": legacy_mean,
            "success_count": int(np.count_nonzero(legacy_ok)),
            "mean_iterations": float(np.mean(legacy_it.astype(np.float64))),
        },
        "optimized_single_engine": {
            "elapsed_sec": fast_times,
            "elapsed_sec_mean": fast_mean,
            "success_count": int(np.count_nonzero(fast_ok)),
            "mean_iterations": float(np.mean(fast_it.astype(np.float64))),
        },
        "consistency": {
            "mae": mae,
            "max_abs": max_abs,
            "rmse": rmse,
            "flatten_corrcoef": flat_cc,
            "iteration_equal_count": iter_equal,
            "trace_count": int(obs.shape[0]),
        },
        "speedup_vs_legacy": float(speedup),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
