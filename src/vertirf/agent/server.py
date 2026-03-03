from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from vertirf.core.decon import DeconConfig, run_batch_decon
from vertirf.filters.zero_phase import FilterSpec
from vertirf.waveform.synthetic import make_synthetic_batch


def _build_cfg(params: dict[str, Any]) -> tuple[DeconConfig, str, int, int, int, float]:
    filter_spec = FilterSpec(
        filter_type=str(params.get("filter_type", "gaussian")),
        gauss_f0=float(params.get("gauss_f0", np.pi)),
        low_hz=float(params.get("low_hz", 0.1)),
        high_hz=float(params.get("high_hz", 0.8)),
        corners=int(params.get("corners", 4)),
        transition_hz=float(params.get("transition_hz", 0.05)),
        tukey_alpha=float(params.get("tukey_alpha", 0.3)),
    )
    cfg = DeconConfig(
        dt=float(params.get("dt", 0.05)),
        tshift_sec=float(params.get("tshift_sec", 0.0)),
        itmax=int(params.get("itmax", 400)),
        minderr=float(params.get("minderr", 1e-3)),
        allow_negative_impulse=bool(params.get("allow_negative_impulse", False)),
        filter_spec=filter_spec,
    )
    mode = str(params.get("mode", "optimized"))
    traces = int(params.get("traces", 32))
    samples = int(params.get("samples", 1024))
    jobs = int(params.get("jobs", 1))
    noise_std = float(params.get("noise_std", 0.015))
    return cfg, mode, traces, samples, jobs, noise_std


def _run_synthetic(params: dict[str, Any]) -> dict[str, Any]:
    cfg, mode, traces, samples, jobs, noise_std = _build_cfg(params)
    src, truth, obs = make_synthetic_batch(
        traces=traces,
        samples=samples,
        dt=cfg.dt,
        wavelet_hz=float(params.get("wavelet_hz", 0.8)),
        noise_std=noise_std,
        rng_seed=int(params.get("seed", 0)),
    )
    rec, ok, iters = run_batch_decon(obs, src, cfg, mode=mode, jobs=jobs)
    return {
        "mode": mode,
        "traces": traces,
        "samples": samples,
        "jobs": jobs,
        "success_count": int(np.count_nonzero(ok)),
        "success_rate": float(np.mean(ok.astype(np.float64))),
        "mean_iterations": float(np.mean(iters.astype(np.float64))),
        "mean_abs_error": float(np.mean(np.abs(rec - truth))),
    }


def dispatch(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    p = params or {}
    if method == "ping":
        return {"pong": True, "service": "vertirf-agent"}
    if method == "run_decon_synthetic":
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
    run = dispatch(
        "run_decon_synthetic",
        {
            "mode": "optimized",
            "filter_type": "butterworth_bandpass",
            "allow_negative_impulse": True,
            "traces": 8,
            "samples": 512,
            "jobs": 2,
            "itmax": 200,
        },
    )
    ok = bool(ping.get("pong")) and run.get("success_count", 0) > 0
    out = {"self_test_passed": ok, "ping": ping, "run": run}
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
