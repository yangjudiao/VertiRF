from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vertirf.core.decon import DeconConfig, run_batch_decon
from vertirf.filters.zero_phase import FilterSpec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run reproducible engineering benchmark on a fixed dataset")
    p.add_argument("--dataset", default="data/engineering_benchmark/engineering_dataset.npz")
    p.add_argument("--out", default="data/engineering_benchmark/repro_report.json")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--itmax", type=int, default=260)
    p.add_argument("--minderr", type=float, default=1e-3)
    p.add_argument("--allow-negative-impulse", action="store_true")
    p.add_argument(
        "--filter-type",
        choices=["gaussian", "butterworth_bandpass", "raised_cosine_bandpass", "tukey_bandpass"],
        default="butterworth_bandpass",
    )
    p.add_argument("--low-hz", type=float, default=0.1)
    p.add_argument("--high-hz", type=float, default=0.8)
    p.add_argument("--corners", type=int, default=4)
    p.add_argument("--transition-hz", type=float, default=0.05)
    p.add_argument("--tukey-alpha", type=float, default=0.3)
    p.add_argument("--gauss-f0", type=float, default=float(np.pi))
    return p.parse_args()


def run_once(observed: np.ndarray, source: np.ndarray, cfg: DeconConfig, mode: str, jobs: int) -> dict:
    t0 = time.perf_counter()
    recovered, ok, iters = run_batch_decon(observed, source, cfg, mode=mode, jobs=jobs)
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_sec": float(elapsed),
        "success_count": int(np.count_nonzero(ok)),
        "success_rate": float(np.mean(ok.astype(np.float64))),
        "mean_iterations": float(np.mean(iters.astype(np.float64))),
        "mean_abs_amplitude": float(np.mean(np.abs(recovered))),
        "recovered": recovered,
    }


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise RuntimeError(f"dataset not found: {dataset_path}")

    with np.load(dataset_path, allow_pickle=False) as d:
        source = np.asarray(d["source_wavelet"], dtype=np.float64)
        observed = np.asarray(d["observed_traces"], dtype=np.float64)
        dt_sec = float(d["dt_sec"])

    cfg = DeconConfig(
        dt=dt_sec,
        tshift_sec=0.0,
        itmax=int(args.itmax),
        minderr=float(args.minderr),
        allow_negative_impulse=bool(args.allow_negative_impulse),
        filter_spec=FilterSpec(
            filter_type=str(args.filter_type),
            gauss_f0=float(args.gauss_f0),
            low_hz=float(args.low_hz),
            high_hz=float(args.high_hz),
            corners=int(args.corners),
            transition_hz=float(args.transition_hz),
            tukey_alpha=float(args.tukey_alpha),
        ),
    )

    n_repeat = max(1, int(args.repeat))
    baseline_runs: list[dict] = []
    optimized_runs: list[dict] = []

    for _ in range(n_repeat):
        baseline_runs.append(run_once(observed, source, cfg, mode="baseline", jobs=1))
        optimized_runs.append(run_once(observed, source, cfg, mode="optimized", jobs=max(1, int(args.jobs))))

    baseline_elapsed = float(np.mean([x["elapsed_sec"] for x in baseline_runs]))
    optimized_elapsed = float(np.mean([x["elapsed_sec"] for x in optimized_runs]))
    speedup = baseline_elapsed / max(1e-12, optimized_elapsed)

    b0 = baseline_runs[0]["recovered"]
    o0 = optimized_runs[0]["recovered"]
    consistency_mae = float(np.mean(np.abs(b0 - o0)))

    for x in baseline_runs:
        x.pop("recovered", None)
    for x in optimized_runs:
        x.pop("recovered", None)

    report = {
        "dataset": str(dataset_path),
        "out": str(out_path),
        "dataset_shape": {
            "traces": int(observed.shape[0]),
            "samples": int(observed.shape[1]),
            "dt_sec": float(dt_sec),
        },
        "config": {
            "filter_type": str(args.filter_type),
            "allow_negative_impulse": bool(args.allow_negative_impulse),
            "jobs": int(args.jobs),
            "repeat": int(n_repeat),
            "itmax": int(args.itmax),
            "minderr": float(args.minderr),
        },
        "baseline": {
            "elapsed_sec_mean": baseline_elapsed,
            "runs": baseline_runs,
        },
        "optimized": {
            "elapsed_sec_mean": optimized_elapsed,
            "runs": optimized_runs,
        },
        "optimized_vs_baseline_speedup": float(speedup),
        "baseline_optimized_consistency_mae": consistency_mae,
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
