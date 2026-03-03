from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vertirf.core.decon import DeconConfig, run_batch_decon
from vertirf.filters.zero_phase import FilterSpec
from vertirf.waveform.synthetic import make_synthetic_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark VertiRF baseline vs optimized")
    p.add_argument("--out", type=Path, default=Path("benchmark_summary.json"))
    p.add_argument("--traces", type=int, default=128)
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--itmax", type=int, default=280)
    p.add_argument("--minderr", type=float, default=1e-3)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--filter-type", choices=["gaussian", "butterworth_bandpass", "raised_cosine_bandpass", "tukey_bandpass"], default="butterworth_bandpass")
    p.add_argument("--allow-negative-impulse", action="store_true")
    p.add_argument("--min-speedup", type=float, default=1.5)
    return p.parse_args()


def _run(mode: str, obs: np.ndarray, src: np.ndarray, cfg: DeconConfig, jobs: int) -> tuple[float, dict[str, float]]:
    t0 = time.perf_counter()
    rec, ok, iters = run_batch_decon(obs, src, cfg, mode=mode, jobs=jobs)
    elapsed = time.perf_counter() - t0
    return elapsed, {
        "success_count": int(np.count_nonzero(ok)),
        "success_rate": float(np.mean(ok.astype(np.float64))),
        "mean_iterations": float(np.mean(iters.astype(np.float64))),
        "mean_abs_amplitude": float(np.mean(np.abs(rec))),
    }


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
        tshift_sec=0.0,
        itmax=int(args.itmax),
        minderr=float(args.minderr),
        allow_negative_impulse=bool(args.allow_negative_impulse),
        filter_spec=FilterSpec(
            filter_type=str(args.filter_type),
            gauss_f0=np.pi,
            low_hz=0.1,
            high_hz=0.8,
            corners=4,
            transition_hz=0.05,
            tukey_alpha=0.3,
        ),
    )

    baseline_times = []
    baseline_stats = []
    optimized_times = []
    optimized_stats = []

    for _ in range(max(1, int(args.repeat))):
        t_b, s_b = _run("baseline", obs, src, cfg, jobs=1)
        baseline_times.append(t_b)
        baseline_stats.append(s_b)

        t_o, s_o = _run("optimized", obs, src, cfg, jobs=max(1, int(args.jobs)))
        optimized_times.append(t_o)
        optimized_stats.append(s_o)

    baseline_mean = float(np.mean(np.asarray(baseline_times, dtype=np.float64)))
    optimized_mean = float(np.mean(np.asarray(optimized_times, dtype=np.float64)))
    speedup = baseline_mean / max(1e-12, optimized_mean)

    summary = {
        "benchmark": {
            "traces": int(args.traces),
            "samples": int(args.samples),
            "repeat": int(max(1, int(args.repeat))),
            "jobs_optimized": int(max(1, int(args.jobs))),
            "filter_type": str(args.filter_type),
            "allow_negative_impulse": bool(args.allow_negative_impulse),
            "itmax": int(args.itmax),
            "minderr": float(args.minderr),
            "seed": int(args.seed),
        },
        "baseline": {
            "elapsed_sec": baseline_times,
            "elapsed_sec_mean": baseline_mean,
            "stats": baseline_stats,
        },
        "optimized": {
            "elapsed_sec": optimized_times,
            "elapsed_sec_mean": optimized_mean,
            "stats": optimized_stats,
        },
        "optimized_vs_baseline_speedup": float(speedup),
        "min_speedup_target": float(args.min_speedup),
        "speedup_passed": bool(speedup >= float(args.min_speedup)),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    return 0 if speedup >= float(args.min_speedup) else 2


if __name__ == "__main__":
    raise SystemExit(main())
