from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vertirf.core.methods import MethodConfig, run_batch_method
from vertirf.filters.zero_phase import FilterSpec
from vertirf.waveform.synthetic import make_synthetic_batch

METHODS = ("decon", "corr", "stack")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark serial vs parallel speedup for decon/corr/stack")
    p.add_argument("--out", type=Path, default=Path("method_parallel_benchmark_summary.json"))
    p.add_argument("--traces", type=int, default=96)
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--noise-std", type=float, default=0.015)
    p.add_argument("--filter-type", choices=["gaussian", "butterworth_bandpass", "raised_cosine_bandpass", "tukey_bandpass"], default="butterworth_bandpass")
    p.add_argument("--low-hz", type=float, default=0.1)
    p.add_argument("--high-hz", type=float, default=0.8)
    p.add_argument("--itmax", type=int, default=260)
    p.add_argument("--minderr", type=float, default=1e-3)
    return p.parse_args()


def _base_cfg(args: argparse.Namespace, method: str) -> MethodConfig:
    return MethodConfig(
        method=method,
        dt=float(args.dt),
        tshift_sec=0.0,
        itmax=int(args.itmax),
        minderr=float(args.minderr),
        allow_negative_impulse=True,
        filter_spec=FilterSpec(
            filter_type=str(args.filter_type),
            gauss_f0=float(np.pi),
            low_hz=float(args.low_hz),
            high_hz=float(args.high_hz),
            corners=4,
            transition_hz=0.05,
            tukey_alpha=0.3,
        ),
        corr_smoothing_bandwidth_hz=0.25,
        corr_post_filter_type="gaussian",
        stack_peak_window_start_sec=-2.0,
        stack_peak_window_end_sec=20.0,
    )


def _run_timed(observed: np.ndarray, source: np.ndarray, cfg: MethodConfig, jobs: int) -> dict[str, float]:
    t0 = time.perf_counter()
    _, ok, steps = run_batch_method(observed, source, cfg, mode="optimized", jobs=jobs)
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_sec": float(elapsed),
        "success_count": int(np.count_nonzero(ok)),
        "success_rate": float(np.mean(ok.astype(np.float64))),
        "mean_steps": float(np.mean(steps.astype(np.float64))),
    }


def main() -> int:
    args = parse_args()

    src, _, obs = make_synthetic_batch(
        traces=int(args.traces),
        samples=int(args.samples),
        dt=float(args.dt),
        noise_std=float(args.noise_std),
        rng_seed=int(args.seed),
    )

    repeat = max(1, int(args.repeat))
    jobs_parallel = max(1, int(args.jobs))

    methods: dict[str, dict] = {}
    for method in METHODS:
        cfg = _base_cfg(args, method)

        serial_runs = []
        parallel_runs = []
        for _ in range(repeat):
            serial_runs.append(_run_timed(obs, src, cfg, jobs=1))
            parallel_runs.append(_run_timed(obs, src, cfg, jobs=jobs_parallel))

        serial_mean = float(np.mean([x["elapsed_sec"] for x in serial_runs]))
        parallel_mean = float(np.mean([x["elapsed_sec"] for x in parallel_runs]))
        speedup = serial_mean / max(1e-12, parallel_mean)

        methods[method] = {
            "serial": {
                "elapsed_sec_mean": serial_mean,
                "runs": serial_runs,
            },
            "parallel": {
                "elapsed_sec_mean": parallel_mean,
                "runs": parallel_runs,
                "jobs": jobs_parallel,
            },
            "parallel_speedup_vs_serial": float(speedup),
        }

    summary = {
        "benchmark": {
            "traces": int(args.traces),
            "samples": int(args.samples),
            "dt": float(args.dt),
            "repeat": int(repeat),
            "jobs_parallel": int(jobs_parallel),
            "seed": int(args.seed),
            "noise_std": float(args.noise_std),
            "filter_type": str(args.filter_type),
            "low_hz": float(args.low_hz),
            "high_hz": float(args.high_hz),
        },
        "methods": methods,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
