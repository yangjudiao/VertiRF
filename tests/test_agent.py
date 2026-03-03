from __future__ import annotations

from vertirf.agent.server import dispatch


def test_agent_ping() -> None:
    out = dispatch("ping", {})
    assert out["pong"] is True


def test_agent_run_decon_synthetic() -> None:
    out = dispatch(
        "run_decon_synthetic",
        {
            "mode": "optimized",
            "filter_type": "tukey_bandpass",
            "allow_negative_impulse": True,
            "traces": 6,
            "samples": 512,
            "jobs": 2,
            "itmax": 160,
        },
    )
    assert out["success_count"] > 0
    assert out["success_rate"] > 0.5


def test_agent_run_corr_stack_methods() -> None:
    corr = dispatch(
        "run_method_synthetic",
        {
            "method": "corr",
            "mode": "optimized",
            "filter_type": "butterworth_bandpass",
            "corr_smoothing_bandwidth_hz": 0.25,
            "corr_post_filter_type": "gaussian",
            "traces": 6,
            "samples": 512,
            "jobs": 2,
        },
    )
    stack = dispatch(
        "run_method_synthetic",
        {
            "method": "stack",
            "mode": "optimized",
            "filter_type": "butterworth_bandpass",
            "stack_peak_window_start_sec": -1.0,
            "stack_peak_window_end_sec": 3.0,
            "traces": 6,
            "samples": 512,
            "jobs": 2,
        },
    )

    assert corr["success_count"] == 6
    assert stack["success_count"] == 6
