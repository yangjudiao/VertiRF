from __future__ import annotations

from vertirf.agent.server import dispatch


def test_agent_ping() -> None:
    out = dispatch("ping", {})
    assert out["pong"] is True


def test_agent_run_synthetic() -> None:
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
