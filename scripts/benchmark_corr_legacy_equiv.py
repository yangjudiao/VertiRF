from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vertirf.core.decon import normalize_max_abs
from vertirf.core.methods import MethodConfig, run_batch_method
from vertirf.filters.zero_phase import FilterSpec, build_zero_phase_response
from vertirf.waveform.synthetic import make_synthetic_batch


def _parse_bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark legacy-compatible corr reference vs optimized single corr engine")
    p.add_argument("--out", type=Path, default=Path("benchmark_corr_legacy_equiv.json"))
    p.add_argument("--traces", type=int, default=160)
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--noise-std", type=float, default=0.015)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--smooth-hz", type=float, default=0.2)
    p.add_argument("--water-level", type=float, default=1e-4)
    p.add_argument("--shift-sec", type=float, default=0.0)
    p.add_argument("--divide-denom", type=_parse_bool, default=True)
    return p.parse_args()


def _smooth_edge(power: np.ndarray, win_len: int) -> np.ndarray:
    p = np.asarray(power, dtype=np.float64)
    if win_len <= 1:
        return p
    w = np.ones((int(win_len),), dtype=np.float64) / float(win_len)
    wei = np.convolve(np.ones((p.size,), dtype=np.float64), w, mode="same")
    smo = np.convolve(p, w, mode="same")
    return smo / np.maximum(wei, 1e-12)


def _legacy_corr_batch(observed: np.ndarray, source: np.ndarray, cfg: MethodConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(observed, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    n = int(src.size)

    b = np.fft.rfft(src, n=n)
    bpow = np.abs(b) ** 2
    df = 1.0 / max(float(n) * float(cfg.dt), 1e-12)
    half_bins = int(round(max(0.0, float(cfg.corr_smoothing_bandwidth_hz)) / max(df, 1e-12)))
    win_len = max(1, 2 * half_bins + 1)
    denom = _smooth_edge(bpow, win_len)
    denom = np.maximum(denom, 1e-12)
    denom_max = float(np.max(denom)) if denom.size else 0.0
    water_floor = max(1e-12, denom_max * max(0.0, float(cfg.corr_water_level)))
    denom_water = np.maximum(denom, water_floor)

    filt = build_zero_phase_response(nfft=n, dt=float(cfg.dt), spec=cfg.filter_spec)
    shift_samples = int(round(float(cfg.corr_shift_sec) / float(cfg.dt)))

    out = np.zeros_like(arr, dtype=np.float64)
    ok = np.ones((arr.shape[0],), dtype=bool)
    steps = np.ones((arr.shape[0],), dtype=np.int32)
    for i in range(arr.shape[0]):
        d = np.fft.rfft(arr[i], n=n)
        rf_spec = d * np.conj(b)
        if bool(cfg.corr_divide_denom):
            rf_spec = rf_spec / denom_water
        rf_spec = rf_spec * filt
        rf = np.fft.irfft(rf_spec, n=n).real
        rf = np.roll(rf, shift_samples)
        rf, _ = normalize_max_abs(rf)
        out[i] = rf
    return out, ok, steps


def _run_timed(fn, observed: np.ndarray, source: np.ndarray, cfg: MethodConfig, jobs: int = 1):
    t0 = time.perf_counter()
    rec, ok, steps = fn(observed, source, cfg) if fn is _legacy_corr_batch else fn(observed, source, cfg, mode="optimized", jobs=jobs)
    elapsed = time.perf_counter() - t0
    return elapsed, rec, ok, steps


def main() -> int:
    args = parse_args()
    src, _, obs = make_synthetic_batch(
        traces=int(args.traces),
        samples=int(args.samples),
        dt=float(args.dt),
        noise_std=float(args.noise_std),
        rng_seed=int(args.seed),
    )

    cfg = MethodConfig(
        method="corr",
        dt=float(args.dt),
        filter_spec=FilterSpec(
            filter_type="butterworth_bandpass",
            gauss_f0=float(np.pi),
            low_hz=0.1,
            high_hz=0.8,
            corners=4,
            transition_hz=0.05,
            tukey_alpha=0.3,
        ),
        corr_smoothing_bandwidth_hz=float(args.smooth_hz),
        corr_divide_denom=bool(args.divide_denom),
        corr_water_level=float(args.water_level),
        corr_shift_sec=float(args.shift_sec),
        corr_post_filter_type="none",
    )

    legacy_times = []
    fast_times = []
    legacy_last = None
    fast_last = None

    for _ in range(max(1, int(args.repeat))):
        t, rec, ok, steps = _run_timed(_legacy_corr_batch, obs, src, cfg)
        legacy_times.append(t)
        legacy_last = (rec, ok, steps)

        t, rec, ok, steps = _run_timed(run_batch_method, obs, src, cfg, jobs=max(1, int(args.jobs)))
        fast_times.append(t)
        fast_last = (rec, ok, steps)

    rec_l, ok_l, steps_l = legacy_last
    rec_f, ok_f, steps_f = fast_last

    diff = rec_f - rec_l
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    flat_cc = float(np.corrcoef(rec_f.ravel(), rec_l.ravel())[0, 1])

    legacy_mean = float(np.mean(np.asarray(legacy_times, dtype=np.float64)))
    fast_mean = float(np.mean(np.asarray(fast_times, dtype=np.float64)))
    speedup = legacy_mean / max(1e-12, fast_mean)

    summary = {
        "benchmark": {
            "traces": int(args.traces),
            "samples": int(args.samples),
            "dt": float(args.dt),
            "repeat": int(max(1, int(args.repeat))),
            "jobs": int(max(1, int(args.jobs))),
            "seed": int(args.seed),
            "noise_std": float(args.noise_std),
            "corr_smoothing_bandwidth_hz": float(args.smooth_hz),
            "corr_divide_denom": bool(args.divide_denom),
            "corr_water_level": float(args.water_level),
            "corr_shift_sec": float(args.shift_sec),
        },
        "legacy_reference": {
            "elapsed_sec": legacy_times,
            "elapsed_sec_mean": legacy_mean,
            "success_count": int(np.count_nonzero(ok_l)),
            "mean_steps": float(np.mean(steps_l.astype(np.float64))),
        },
        "optimized_single_engine": {
            "elapsed_sec": fast_times,
            "elapsed_sec_mean": fast_mean,
            "success_count": int(np.count_nonzero(ok_f)),
            "mean_steps": float(np.mean(steps_f.astype(np.float64))),
        },
        "consistency": {
            "mae": mae,
            "max_abs": max_abs,
            "rmse": rmse,
            "flatten_corrcoef": flat_cc,
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
