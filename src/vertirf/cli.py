from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from vertirf.core.decon import DeconConfig, run_batch_decon
from vertirf.filters.zero_phase import FilterSpec
from vertirf.waveform.synthetic import corrcoef, make_synthetic_batch, rmse


def _parse_bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {v}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VertiRF CLI")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mode", choices=["baseline", "optimized"], default="optimized")
    common.add_argument("--filter-type", choices=["gaussian", "butterworth_bandpass", "raised_cosine_bandpass", "tukey_bandpass"], default="gaussian")
    common.add_argument("--allow-negative-impulse", type=_parse_bool, default=False)
    common.add_argument("--gauss-f0", type=float, default=np.pi)
    common.add_argument("--low-hz", type=float, default=0.1)
    common.add_argument("--high-hz", type=float, default=0.8)
    common.add_argument("--corners", type=int, default=4)
    common.add_argument("--transition-hz", type=float, default=0.05)
    common.add_argument("--tukey-alpha", type=float, default=0.3)
    common.add_argument("--jobs", type=int, default=1)
    common.add_argument("--dt", type=float, default=0.05)
    common.add_argument("--itmax", type=int, default=400)
    common.add_argument("--minderr", type=float, default=1e-3)
    common.add_argument("--tshift-sec", type=float, default=0.0)

    run_syn = sub.add_parser("run-synthetic", parents=[common], help="Run synthetic decon batch")
    run_syn.add_argument("--traces", type=int, default=48)
    run_syn.add_argument("--samples", type=int, default=1024)
    run_syn.add_argument("--wavelet-hz", type=float, default=0.8)
    run_syn.add_argument("--noise-std", type=float, default=0.015)
    run_syn.add_argument("--seed", type=int, default=0)
    run_syn.add_argument("--out", type=Path, default=None)

    bench = sub.add_parser("benchmark", parents=[common], help="Benchmark baseline vs optimized")
    bench.add_argument("--traces", type=int, default=96)
    bench.add_argument("--samples", type=int, default=1024)
    bench.add_argument("--wavelet-hz", type=float, default=0.8)
    bench.add_argument("--noise-std", type=float, default=0.015)
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--repeat", type=int, default=2)
    bench.add_argument("--out", type=Path, default=Path("benchmark_summary.json"))

    return p


def _build_filter_spec(args: argparse.Namespace) -> FilterSpec:
    return FilterSpec(
        filter_type=str(args.filter_type),
        gauss_f0=float(args.gauss_f0),
        low_hz=float(args.low_hz),
        high_hz=float(args.high_hz),
        corners=int(args.corners),
        transition_hz=float(args.transition_hz),
        tukey_alpha=float(args.tukey_alpha),
    )


def _build_decon_cfg(args: argparse.Namespace) -> DeconConfig:
    return DeconConfig(
        dt=float(args.dt),
        tshift_sec=float(args.tshift_sec),
        itmax=int(args.itmax),
        minderr=float(args.minderr),
        allow_negative_impulse=bool(args.allow_negative_impulse),
        filter_spec=_build_filter_spec(args),
    )


def _evaluate(recovered: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    ccs = []
    rmses = []
    for i in range(recovered.shape[0]):
        ccs.append(corrcoef(recovered[i], truth[i]))
        rmses.append(rmse(recovered[i], truth[i]))
    c = np.asarray(ccs, dtype=np.float64)
    r = np.asarray(rmses, dtype=np.float64)
    return {
        "mean_corrcoef": float(np.nanmean(c)),
        "median_corrcoef": float(np.nanmedian(c)),
        "mean_rmse": float(np.nanmean(r)),
        "median_rmse": float(np.nanmedian(r)),
    }


def _run_once(args: argparse.Namespace, mode: str, jobs: int) -> dict:
    src, resp_true, obs = make_synthetic_batch(
        traces=int(args.traces),
        samples=int(args.samples),
        dt=float(args.dt),
        wavelet_hz=float(args.wavelet_hz),
        noise_std=float(args.noise_std),
        rng_seed=int(args.seed),
    )
    cfg = _build_decon_cfg(args)

    t0 = time.perf_counter()
    rec, ok, iters = run_batch_decon(obs, src, cfg, mode=mode, jobs=int(jobs))
    elapsed = time.perf_counter() - t0

    metrics = _evaluate(rec, resp_true)
    out = {
        "mode": mode,
        "jobs": int(jobs),
        "elapsed_sec": float(elapsed),
        "traces": int(obs.shape[0]),
        "samples": int(obs.shape[1]),
        "success_count": int(np.count_nonzero(ok)),
        "success_rate": float(np.mean(ok.astype(np.float64))),
        "mean_iterations": float(np.mean(iters.astype(np.float64))),
    }
    out.update(metrics)
    return out


def cmd_run_synthetic(args: argparse.Namespace) -> int:
    res = _run_once(args, mode=str(args.mode), jobs=int(args.jobs))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    repeats = max(1, int(args.repeat))

    baseline_runs = []
    optimized_runs = []
    for _ in range(repeats):
        baseline_runs.append(_run_once(args, mode="baseline", jobs=1))
        optimized_runs.append(_run_once(args, mode="optimized", jobs=max(1, int(args.jobs))))

    baseline_elapsed = float(np.mean([x["elapsed_sec"] for x in baseline_runs]))
    optimized_elapsed = float(np.mean([x["elapsed_sec"] for x in optimized_runs]))
    speedup = baseline_elapsed / max(1e-12, optimized_elapsed)

    summary = {
        "baseline": {
            "elapsed_sec_mean": baseline_elapsed,
            "runs": baseline_runs,
        },
        "optimized": {
            "elapsed_sec_mean": optimized_elapsed,
            "runs": optimized_runs,
        },
        "optimized_vs_baseline_speedup": float(speedup),
        "benchmark_config": {
            "traces": int(args.traces),
            "samples": int(args.samples),
            "repeat": int(repeats),
            "jobs_optimized": int(max(1, int(args.jobs))),
            "filter_type": str(args.filter_type),
            "allow_negative_impulse": bool(args.allow_negative_impulse),
            "itmax": int(args.itmax),
            "minderr": float(args.minderr),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-synthetic":
        return cmd_run_synthetic(args)
    if args.command == "benchmark":
        return cmd_benchmark(args)

    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
