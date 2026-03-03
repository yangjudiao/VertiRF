from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vertirf.core.methods import MethodConfig, run_batch_method
from vertirf.filters.zero_phase import FilterSpec
from vertirf.waveform.synthetic import corrcoef, make_synthetic_batch, rmse

FILTER_TYPES = ["gaussian", "butterworth_bandpass", "raised_cosine_bandpass", "tukey_bandpass"]
POST_FILTER_TYPES = ["none"] + FILTER_TYPES
METHODS = ["decon", "corr", "stack"]


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
    common.add_argument("--method", choices=METHODS, default="decon")
    common.add_argument("--mode", choices=["baseline", "optimized"], default="optimized")
    common.add_argument("--filter-type", choices=FILTER_TYPES, default="gaussian")
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

    # Corr-specific
    common.add_argument("--corr-smoothing-bandwidth-hz", type=float, default=0.35)
    common.add_argument("--corr-divide-denom", type=_parse_bool, default=True)
    common.add_argument("--corr-water-level", type=float, default=1e-4)
    common.add_argument("--corr-shift-sec", type=float, default=0.0)
    common.add_argument("--corr-post-filter-type", choices=POST_FILTER_TYPES, default="none")
    common.add_argument("--corr-post-gauss-f0", type=float, default=np.pi)
    common.add_argument("--corr-post-low-hz", type=float, default=0.1)
    common.add_argument("--corr-post-high-hz", type=float, default=0.8)
    common.add_argument("--corr-post-corners", type=int, default=4)
    common.add_argument("--corr-fft-switch-samples", type=int, default=8192)

    # Stack-specific
    common.add_argument("--stack-peak-window-start-sec", type=float, default=-2.0)
    common.add_argument("--stack-peak-window-end-sec", type=float, default=20.0)

    run_syn = sub.add_parser("run-synthetic", parents=[common], help="Run synthetic method batch")
    run_syn.add_argument("--traces", type=int, default=48)
    run_syn.add_argument("--samples", type=int, default=1024)
    run_syn.add_argument("--wavelet-hz", type=float, default=0.8)
    run_syn.add_argument("--noise-std", type=float, default=0.015)
    run_syn.add_argument("--seed", type=int, default=0)
    run_syn.add_argument("--out", type=Path, default=None)

    bench = sub.add_parser("benchmark", parents=[common], help="Benchmark serial vs parallel for one method")
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


def _build_method_cfg(args: argparse.Namespace) -> MethodConfig:
    return MethodConfig(
        method=str(args.method),
        dt=float(args.dt),
        tshift_sec=float(args.tshift_sec),
        itmax=int(args.itmax),
        minderr=float(args.minderr),
        allow_negative_impulse=bool(args.allow_negative_impulse),
        filter_spec=_build_filter_spec(args),
        corr_smoothing_bandwidth_hz=float(args.corr_smoothing_bandwidth_hz),
        corr_divide_denom=bool(args.corr_divide_denom),
        corr_water_level=float(args.corr_water_level),
        corr_shift_sec=float(args.corr_shift_sec),
        corr_post_filter_type=str(args.corr_post_filter_type),
        corr_post_gauss_f0=float(args.corr_post_gauss_f0),
        corr_post_low_hz=float(args.corr_post_low_hz),
        corr_post_high_hz=float(args.corr_post_high_hz),
        corr_post_corners=int(args.corr_post_corners),
        corr_fft_switch_samples=int(args.corr_fft_switch_samples),
        stack_peak_window_start_sec=float(args.stack_peak_window_start_sec),
        stack_peak_window_end_sec=float(args.stack_peak_window_end_sec),
    )


def _evaluate(recovered: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    ccs = []
    rmses = []
    for i in range(recovered.shape[0]):
        ccs.append(corrcoef(recovered[i], truth[i]))
        rmses.append(rmse(recovered[i], truth[i]))

    c = np.asarray(ccs, dtype=np.float64)
    r = np.asarray(rmses, dtype=np.float64)

    avg_rec = np.mean(recovered, axis=0)
    avg_true = np.mean(truth, axis=0)
    stack_cc = corrcoef(avg_rec, avg_true)

    peak_off = np.argmax(np.abs(recovered), axis=1) - np.argmax(np.abs(truth), axis=1)
    peak_off = np.asarray(peak_off, dtype=np.float64)

    return {
        "mean_corrcoef": float(np.nanmean(c)),
        "median_corrcoef": float(np.nanmedian(c)),
        "mean_rmse": float(np.nanmean(r)),
        "median_rmse": float(np.nanmedian(r)),
        "stack_profile_corrcoef": float(stack_cc),
        "peak_offset_mean_samples": float(np.mean(peak_off)),
        "peak_offset_std_samples": float(np.std(peak_off)),
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
    cfg = _build_method_cfg(args)
    effective_mode = "optimized" if str(args.method) in {"decon", "corr"} else str(mode)

    t0 = time.perf_counter()
    rec, ok, steps = run_batch_method(obs, src, cfg, mode=effective_mode, jobs=int(jobs))
    elapsed = time.perf_counter() - t0

    metrics = _evaluate(rec, resp_true)
    out = {
        "method": str(args.method),
        "mode": effective_mode,
        "requested_mode": str(mode),
        "jobs": int(jobs),
        "elapsed_sec": float(elapsed),
        "traces": int(obs.shape[0]),
        "samples": int(obs.shape[1]),
        "success_count": int(np.count_nonzero(ok)),
        "success_rate": float(np.mean(ok.astype(np.float64))),
        "mean_steps": float(np.mean(steps.astype(np.float64))),
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

    if str(args.method) in {"decon", "corr"}:
        serial_runs = []
        parallel_runs = []
        for _ in range(repeats):
            serial_runs.append(_run_once(args, mode="optimized", jobs=1))
            parallel_runs.append(_run_once(args, mode="optimized", jobs=max(1, int(args.jobs))))

        serial_elapsed = float(np.mean([x["elapsed_sec"] for x in serial_runs]))
        parallel_elapsed = float(np.mean([x["elapsed_sec"] for x in parallel_runs]))
        speedup = serial_elapsed / max(1e-12, parallel_elapsed)

        summary = {
            "method": str(args.method),
            "engine": "single_fast_version",
            "serial": {
                "elapsed_sec_mean": serial_elapsed,
                "runs": serial_runs,
            },
            "parallel": {
                "elapsed_sec_mean": parallel_elapsed,
                "runs": parallel_runs,
                "jobs": int(max(1, int(args.jobs))),
            },
            "parallel_speedup_vs_serial": float(speedup),
            "benchmark_config": {
                "traces": int(args.traces),
                "samples": int(args.samples),
                "repeat": int(repeats),
                "jobs_parallel": int(max(1, int(args.jobs))),
                "filter_type": str(args.filter_type),
                "allow_negative_impulse": bool(args.allow_negative_impulse),
                "itmax": int(args.itmax),
                "minderr": float(args.minderr),
            },
        }
    else:
        baseline_runs = []
        optimized_runs = []
        for _ in range(repeats):
            baseline_runs.append(_run_once(args, mode="baseline", jobs=1))
            optimized_runs.append(_run_once(args, mode="optimized", jobs=max(1, int(args.jobs))))

        baseline_elapsed = float(np.mean([x["elapsed_sec"] for x in baseline_runs]))
        optimized_elapsed = float(np.mean([x["elapsed_sec"] for x in optimized_runs]))
        speedup = baseline_elapsed / max(1e-12, optimized_elapsed)

        summary = {
            "method": str(args.method),
            "baseline": {
                "elapsed_sec_mean": baseline_elapsed,
                "runs": baseline_runs,
            },
            "optimized": {
                "elapsed_sec_mean": optimized_elapsed,
                "runs": optimized_runs,
            },
            "parallel_speedup_vs_serial": float(speedup),
            "benchmark_config": {
                "traces": int(args.traces),
                "samples": int(args.samples),
                "repeat": int(repeats),
                "jobs_parallel": int(max(1, int(args.jobs))),
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
