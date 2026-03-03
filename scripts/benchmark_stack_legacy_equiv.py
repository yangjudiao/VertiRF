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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark legacy-compatible stack reference vs optimized single stack engine")
    p.add_argument("--out", type=Path, default=Path("benchmark_stack_legacy_equiv.json"))
    p.add_argument("--traces", type=int, default=160)
    p.add_argument("--samples", type=int, default=2001)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--noise-std", type=float, default=0.015)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--stack-peak-window-start-sec", type=float, default=-10.0)
    p.add_argument("--stack-peak-window-end-sec", type=float, default=40.0)
    p.add_argument("--stack-zero-index", type=int, default=200)
    return p.parse_args()


def _legacy_stack_batch(observed: np.ndarray, cfg: MethodConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(observed, dtype=np.float64)
    n_traces, n_samples = arr.shape

    target = int(cfg.stack_zero_index) if cfg.stack_zero_index is not None else int(n_samples // 2)
    if cfg.stack_zero_index is None:
        t = (np.arange(n_samples, dtype=np.float64) - 0.5 * (n_samples - 1)) * float(cfg.dt)
    else:
        t = (np.arange(n_samples, dtype=np.float64) - float(target)) * float(cfg.dt)
    idx = np.where((t >= float(cfg.stack_peak_window_start_sec)) & (t <= float(cfg.stack_peak_window_end_sec)))[0]
    if idx.size < 2:
        idx = np.arange(n_samples, dtype=np.int64)

    out = np.zeros_like(arr, dtype=np.float64)
    ok = np.ones((n_traces,), dtype=bool)
    steps = np.ones((n_traces,), dtype=np.int32)
    for i in range(n_traces):
        row = arr[i]
        filt = build_zero_phase_response(nfft=n_samples, dt=float(cfg.dt), spec=cfg.filter_spec)
        y = np.fft.irfft(np.fft.rfft(row, n=n_samples) * filt, n=n_samples).real
        local = int(np.argmax(np.abs(y[idx])))
        peak_idx = int(idx[local])
        peak_amp = float(y[peak_idx])
        shift = int(target - peak_idx)
        ys = np.roll(y, shift)
        if peak_amp < 0.0:
            ys = -ys
        ys, _ = normalize_max_abs(ys)
        out[i] = ys
    return out, ok, steps


def _run_timed_legacy(observed: np.ndarray, cfg: MethodConfig) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    t0 = time.perf_counter()
    out = _legacy_stack_batch(observed, cfg)
    return time.perf_counter() - t0, out


def _run_timed_fast(
    observed: np.ndarray,
    source: np.ndarray,
    cfg: MethodConfig,
    jobs: int,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    t0 = time.perf_counter()
    out = run_batch_method(observed, source, cfg, mode="optimized", jobs=max(1, int(jobs)))
    return time.perf_counter() - t0, out


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
        method="stack",
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
        stack_peak_window_start_sec=float(args.stack_peak_window_start_sec),
        stack_peak_window_end_sec=float(args.stack_peak_window_end_sec),
        stack_zero_index=int(args.stack_zero_index),
    )

    legacy_times: list[float] = []
    fast_times: list[float] = []
    legacy_last: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    fast_last: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for _ in range(max(1, int(args.repeat))):
        t, out = _run_timed_legacy(obs, cfg)
        legacy_times.append(float(t))
        legacy_last = out

        t, out = _run_timed_fast(obs, src, cfg, jobs=max(1, int(args.jobs)))
        fast_times.append(float(t))
        fast_last = out

    legacy_rec, legacy_ok, legacy_steps = legacy_last
    fast_rec, fast_ok, fast_steps = fast_last

    diff = fast_rec - legacy_rec
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    flat_cc = float(np.corrcoef(fast_rec.ravel(), legacy_rec.ravel())[0, 1])

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
            "stack_peak_window_start_sec": float(args.stack_peak_window_start_sec),
            "stack_peak_window_end_sec": float(args.stack_peak_window_end_sec),
            "stack_zero_index": int(args.stack_zero_index),
        },
        "legacy_reference": {
            "elapsed_sec": legacy_times,
            "elapsed_sec_mean": legacy_mean,
            "success_count": int(np.count_nonzero(legacy_ok)),
            "mean_steps": float(np.mean(legacy_steps.astype(np.float64))),
        },
        "optimized_single_engine": {
            "elapsed_sec": fast_times,
            "elapsed_sec_mean": fast_mean,
            "success_count": int(np.count_nonzero(fast_ok)),
            "mean_steps": float(np.mean(fast_steps.astype(np.float64))),
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
