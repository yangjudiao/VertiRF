from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from vertirf.core.methods import MethodConfig, run_batch_method
from vertirf.filters.zero_phase import FilterSpec
from vertirf.waveform.synthetic import make_synthetic_batch


def _build_cfg(params: dict[str, Any]) -> tuple[MethodConfig, str, int, int, int, float]:
    filter_spec = FilterSpec(
        filter_type=str(params.get("filter_type", "gaussian")),
        gauss_f0=float(params.get("gauss_f0", np.pi)),
        low_hz=float(params.get("low_hz", 0.1)),
        high_hz=float(params.get("high_hz", 0.8)),
        corners=int(params.get("corners", 4)),
        transition_hz=float(params.get("transition_hz", 0.05)),
        tukey_alpha=float(params.get("tukey_alpha", 0.3)),
    )
    cfg = MethodConfig(
        method=str(params.get("method", "decon")),
        dt=float(params.get("dt", 0.05)),
        tshift_sec=float(params.get("tshift_sec", 0.0)),
        itmax=int(params.get("itmax", 400)),
        minderr=float(params.get("minderr", 1e-3)),
        allow_negative_impulse=bool(params.get("allow_negative_impulse", False)),
        filter_spec=filter_spec,
        corr_smoothing_bandwidth_hz=float(params.get("corr_smoothing_bandwidth_hz", 0.35)),
        corr_post_filter_type=str(params.get("corr_post_filter_type", "none")),
        corr_post_gauss_f0=float(params.get("corr_post_gauss_f0", np.pi)),
        corr_post_low_hz=float(params.get("corr_post_low_hz", 0.1)),
        corr_post_high_hz=float(params.get("corr_post_high_hz", 0.8)),
        corr_post_corners=int(params.get("corr_post_corners", 4)),
        stack_peak_window_start_sec=float(params.get("stack_peak_window_start_sec", -2.0)),
        stack_peak_window_end_sec=float(params.get("stack_peak_window_end_sec", 20.0)),
    )
    mode = str(params.get("mode", "optimized"))
    traces = int(params.get("traces", 32))
    samples = int(params.get("samples", 1024))
    jobs = int(params.get("jobs", 1))
    noise_std = float(params.get("noise_std", 0.015))
    return cfg, mode, traces, samples, jobs, noise_std


def _run_synthetic(params: dict[str, Any]) -> dict[str, Any]:
    cfg, mode, traces, samples, jobs, noise_std = _build_cfg(params)
    effective_mode = "optimized" if str(cfg.method) == "decon" else str(mode)
    src, truth, obs = make_synthetic_batch(
        traces=traces,
        samples=samples,
        dt=cfg.dt,
        wavelet_hz=float(params.get("wavelet_hz", 0.8)),
        noise_std=noise_std,
        rng_seed=int(params.get("seed", 0)),
    )
    rec, ok, steps = run_batch_method(obs, src, cfg, mode=effective_mode, jobs=jobs)
    return {
        "method": cfg.method,
        "mode": effective_mode,
        "requested_mode": mode,
        "traces": traces,
        "samples": samples,
        "jobs": jobs,
        "success_count": int(np.count_nonzero(ok)),
        "success_rate": float(np.mean(ok.astype(np.float64))),
        "mean_steps": float(np.mean(steps.astype(np.float64))),
        "mean_abs_error": float(np.mean(np.abs(rec - truth))),
    }


def dispatch(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    p = params or {}
    if method == "ping":
        return {"pong": True, "service": "vertirf-agent"}
    if method == "run_decon_synthetic":
        return _run_synthetic({**p, "method": "decon"})
    if method == "run_method_synthetic":
        return _run_synthetic(p)
    raise ValueError(f"unknown method: {method}")


def _rpc_loop() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = str(req.get("method", ""))
            params = req.get("params", {})
            result = dispatch(method, params)
            resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as exc:  # noqa: BLE001
            resp = {
                "jsonrpc": "2.0",
                "id": req.get("id") if isinstance(req, dict) else None,
                "error": {"code": -32000, "message": str(exc)},
            }
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    return 0


def _self_test() -> int:
    ping = dispatch("ping", {})
    run_decon = dispatch(
        "run_method_synthetic",
        {
            "method": "decon",
            "mode": "optimized",
            "filter_type": "butterworth_bandpass",
            "allow_negative_impulse": True,
            "traces": 8,
            "samples": 512,
            "jobs": 2,
            "itmax": 200,
        },
    )
    run_corr = dispatch(
        "run_method_synthetic",
        {
            "method": "corr",
            "mode": "optimized",
            "filter_type": "butterworth_bandpass",
            "corr_smoothing_bandwidth_hz": 0.25,
            "corr_post_filter_type": "gaussian",
            "traces": 8,
            "samples": 512,
            "jobs": 2,
            "itmax": 120,
        },
    )
    run_stack = dispatch(
        "run_method_synthetic",
        {
            "method": "stack",
            "mode": "optimized",
            "filter_type": "butterworth_bandpass",
            "stack_peak_window_start_sec": -2,
            "stack_peak_window_end_sec": 20,
            "traces": 8,
            "samples": 512,
            "jobs": 2,
            "itmax": 120,
        },
    )

    ok = bool(ping.get("pong")) and min(
        run_decon.get("success_count", 0),
        run_corr.get("success_count", 0),
        run_stack.get("success_count", 0),
    ) > 0

    out = {
        "self_test_passed": ok,
        "ping": ping,
        "run_decon": run_decon,
        "run_corr": run_corr,
        "run_stack": run_stack,
    }
    print(json.dumps(out, indent=2))
    return 0 if ok else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VertiRF MCP-style JSON-RPC agent server")
    p.add_argument("--self-test", action="store_true", help="run internal self-test and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return _self_test()
    return _rpc_loop()


if __name__ == "__main__":
    raise SystemExit(main())
