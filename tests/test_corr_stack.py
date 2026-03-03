from __future__ import annotations

import numpy as np

from vertirf.core.methods import MethodConfig, run_batch_method
from vertirf.filters.zero_phase import FilterSpec
from vertirf.waveform.synthetic import convolve_same, corrcoef, make_synthetic_batch, ricker_wavelet


def _cfg(method: str) -> MethodConfig:
    return MethodConfig(
        method=method,
        dt=0.05,
        itmax=220,
        minderr=1e-3,
        allow_negative_impulse=True,
        filter_spec=FilterSpec(
            filter_type="butterworth_bandpass",
            gauss_f0=np.pi,
            low_hz=0.1,
            high_hz=0.8,
            corners=4,
            transition_hz=0.05,
            tukey_alpha=0.3,
        ),
        corr_smoothing_bandwidth_hz=0.25,
        corr_post_filter_type="gaussian",
        corr_post_gauss_f0=np.pi,
        stack_peak_window_start_sec=-1.0,
        stack_peak_window_end_sec=3.0,
    )


def test_corr_synthetic_recovery_has_consistent_peak_offsets() -> None:
    src, truth, obs = make_synthetic_batch(
        traces=12,
        samples=512,
        dt=0.05,
        noise_std=0.004,
        rng_seed=27,
    )
    cfg = _cfg("corr")
    rec, ok, _ = run_batch_method(obs, src, cfg, mode="optimized", jobs=3)

    assert int(np.count_nonzero(ok)) >= 10

    offsets = np.argmax(np.abs(rec), axis=1) - np.argmax(np.abs(truth), axis=1)
    offsets = np.asarray(offsets, dtype=np.float64)
    assert float(np.std(offsets)) <= 3.0
    assert float(np.max(np.abs(offsets))) <= 32.0

    c = corrcoef(np.mean(rec, axis=0), np.mean(truth, axis=0))
    assert float(c) > 0.10


def test_stack_peak_window_aligns_shifted_traces() -> None:
    n = 512
    dt = 0.05
    center = n // 2

    src = ricker_wavelet(samples=n, dt=dt, freq_hz=0.9)
    resp = np.zeros(n, dtype=np.float64)
    resp[center + 20] = 1.0
    base = convolve_same(src, resp)

    rng = np.random.default_rng(42)
    traces = []
    for _ in range(10):
        s = int(rng.integers(-3, 4))
        tr = np.roll(base, s) + 0.01 * rng.standard_normal(n)
        traces.append(tr)
    obs = np.asarray(traces, dtype=np.float64)

    cfg = MethodConfig(
        method="stack",
        dt=dt,
        filter_spec=FilterSpec(filter_type="gaussian", gauss_f0=np.pi),
        stack_peak_window_start_sec=0.5,
        stack_peak_window_end_sec=1.8,
    )

    rec, ok, _ = run_batch_method(obs, src, cfg, mode="optimized", jobs=2)
    assert int(np.count_nonzero(ok)) == obs.shape[0]

    peaks = np.argmax(np.abs(rec), axis=1)
    max_dev = int(np.max(np.abs(peaks - center)))
    assert max_dev <= 1


def test_corr_stack_parallel_matches_serial() -> None:
    src, _, obs = make_synthetic_batch(
        traces=14,
        samples=512,
        dt=0.05,
        noise_std=0.01,
        rng_seed=9,
    )

    for method in ("corr", "stack"):
        cfg = _cfg(method)
        r0, ok0, _ = run_batch_method(obs, src, cfg, mode="optimized", jobs=1)
        r1, ok1, _ = run_batch_method(obs, src, cfg, mode="optimized", jobs=4)

        assert int(np.count_nonzero(ok0)) == obs.shape[0]
        assert int(np.count_nonzero(ok1)) == obs.shape[0]
        diff = float(np.mean(np.abs(r0 - r1)))
        assert diff < 1e-8


def test_corr_mode_flag_is_converged_to_single_engine() -> None:
    src, _, obs = make_synthetic_batch(
        traces=10,
        samples=512,
        dt=0.05,
        noise_std=0.008,
        rng_seed=21,
    )
    cfg = _cfg("corr")
    r_base, ok_base, _ = run_batch_method(obs, src, cfg, mode="baseline", jobs=2)
    r_opt, ok_opt, _ = run_batch_method(obs, src, cfg, mode="optimized", jobs=2)

    assert int(np.count_nonzero(ok_base)) == obs.shape[0]
    assert int(np.count_nonzero(ok_opt)) == obs.shape[0]
    diff = float(np.mean(np.abs(r_base - r_opt)))
    assert diff < 1e-12
