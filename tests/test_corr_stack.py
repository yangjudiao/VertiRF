from __future__ import annotations

import numpy as np

from vertirf.core.decon import normalize_max_abs
from vertirf.core.methods import MethodConfig, run_batch_method
from vertirf.filters.zero_phase import FilterSpec, build_zero_phase_response
from vertirf.waveform.synthetic import convolve_same, make_synthetic_batch, ricker_wavelet


def _corr_cfg() -> MethodConfig:
    return MethodConfig(
        method="corr",
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
        corr_divide_denom=True,
        corr_water_level=1e-4,
        corr_shift_sec=0.0,
        corr_post_filter_type="none",
    )


def _stack_cfg(*, zero_index: int | None = None) -> MethodConfig:
    return MethodConfig(
        method="stack",
        dt=0.05,
        filter_spec=FilterSpec(filter_type="gaussian", gauss_f0=np.pi),
        stack_peak_window_start_sec=0.5,
        stack_peak_window_end_sec=1.8,
        stack_zero_index=zero_index,
    )


def _smooth_edge(power: np.ndarray, win_len: int) -> np.ndarray:
    p = np.asarray(power, dtype=np.float64)
    if win_len <= 1:
        return p
    w = np.ones((int(win_len),), dtype=np.float64) / float(win_len)
    wei = np.convolve(np.ones((p.size,), dtype=np.float64), w, mode="same")
    smo = np.convolve(p, w, mode="same")
    return smo / np.maximum(wei, 1e-12)


def _legacy_corr_batch(observed: np.ndarray, source: np.ndarray, cfg: MethodConfig) -> np.ndarray:
    arr = np.asarray(observed, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    n = int(src.size)

    b = np.fft.rfft(src, n=n)
    bpow = np.abs(b) ** 2
    df = 1.0 / max(float(n) * float(cfg.dt), 1e-12)
    half_bins = int(round(max(0.0, float(cfg.corr_smoothing_bandwidth_hz)) / max(df, 1e-12)))
    win_len = max(1, 2 * half_bins + 1)
    denom = _smooth_edge(bpow, win_len)
    denom = np.maximum(denom, 1e-12)
    denom_max = float(np.max(denom)) if denom.size else 0.0
    water_floor = max(1e-12, denom_max * max(0.0, float(cfg.corr_water_level)))
    denom_water = np.maximum(denom, water_floor)

    filt = build_zero_phase_response(nfft=n, dt=float(cfg.dt), spec=cfg.filter_spec)
    shift_samples = int(round(float(cfg.corr_shift_sec) / float(cfg.dt)))

    out = np.zeros_like(arr, dtype=np.float64)
    for i in range(arr.shape[0]):
        d = np.fft.rfft(arr[i], n=n)
        rf_spec = d * np.conj(b)
        if bool(cfg.corr_divide_denom):
            rf_spec = rf_spec / denom_water
        rf_spec = rf_spec * filt
        rf = np.fft.irfft(rf_spec, n=n).real
        rf = np.roll(rf, shift_samples)
        rf, _ = normalize_max_abs(rf)
        out[i] = rf
    return out


def _legacy_stack_batch(observed: np.ndarray, cfg: MethodConfig) -> np.ndarray:
    arr = np.asarray(observed, dtype=np.float64)
    n = int(arr.shape[1])
    filt = build_zero_phase_response(nfft=n, dt=float(cfg.dt), spec=cfg.filter_spec)
    f = np.fft.rfft(arr, n=n, axis=1)
    y = np.fft.irfft(f * filt[None, :], n=n, axis=1).real

    if cfg.stack_zero_index is None:
        t = (np.arange(n, dtype=np.float64) - 0.5 * (n - 1)) * float(cfg.dt)
        target = int(n // 2)
    else:
        target = int(cfg.stack_zero_index)
        t = (np.arange(n, dtype=np.float64) - float(target)) * float(cfg.dt)

    idx = np.where((t >= float(cfg.stack_peak_window_start_sec)) & (t <= float(cfg.stack_peak_window_end_sec)))[0]
    if idx.size < 2:
        idx = np.arange(n, dtype=np.int64)

    out = np.zeros_like(y, dtype=np.float64)
    for i in range(y.shape[0]):
        row = y[i]
        local = int(np.argmax(np.abs(row[idx])))
        peak_idx = int(idx[local])
        peak_amp = float(row[peak_idx])
        shift = int(target - peak_idx)
        ys = np.roll(row, shift)
        if peak_amp < 0.0:
            ys = -ys
        ys, _ = normalize_max_abs(ys)
        out[i] = ys
    return out


def test_corr_matches_legacy_reference_strict() -> None:
    src, _, obs = make_synthetic_batch(
        traces=12,
        samples=512,
        dt=0.05,
        noise_std=0.006,
        rng_seed=27,
    )
    cfg = _corr_cfg()

    rec, ok, _ = run_batch_method(obs, src, cfg, mode="optimized", jobs=1)
    ref = _legacy_corr_batch(obs, src, cfg)

    assert int(np.count_nonzero(ok)) == obs.shape[0]
    diff = rec - ref
    assert float(np.mean(np.abs(diff))) < 1e-12
    assert float(np.max(np.abs(diff))) < 1e-10


def test_corr_parallel_matches_serial_strict() -> None:
    src, _, obs = make_synthetic_batch(
        traces=30,
        samples=2048,
        dt=0.05,
        noise_std=0.01,
        rng_seed=9,
    )
    cfg = _corr_cfg()

    r0, ok0, _ = run_batch_method(obs, src, cfg, mode="optimized", jobs=1)
    r1, ok1, _ = run_batch_method(obs, src, cfg, mode="optimized", jobs=4)

    assert int(np.count_nonzero(ok0)) == obs.shape[0]
    assert int(np.count_nonzero(ok1)) == obs.shape[0]
    diff = float(np.mean(np.abs(r0 - r1)))
    assert diff < 1e-12


def test_corr_mode_flag_is_converged_to_single_engine() -> None:
    src, _, obs = make_synthetic_batch(
        traces=10,
        samples=512,
        dt=0.05,
        noise_std=0.008,
        rng_seed=21,
    )
    cfg = _corr_cfg()
    r_base, ok_base, _ = run_batch_method(obs, src, cfg, mode="baseline", jobs=2)
    r_opt, ok_opt, _ = run_batch_method(obs, src, cfg, mode="optimized", jobs=2)

    assert int(np.count_nonzero(ok_base)) == obs.shape[0]
    assert int(np.count_nonzero(ok_opt)) == obs.shape[0]
    diff = float(np.mean(np.abs(r_base - r_opt)))
    assert diff < 1e-12


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

    cfg = _stack_cfg()
    rec, ok, _ = run_batch_method(obs, src, cfg, mode="optimized", jobs=2)
    assert int(np.count_nonzero(ok)) == obs.shape[0]

    peaks = np.argmax(np.abs(rec), axis=1)
    max_dev = int(np.max(np.abs(peaks - center)))
    assert max_dev <= 1


def test_stack_matches_legacy_reference_with_zero_index_strict() -> None:
    n = 2001
    dt = 0.05
    zero_index = 200
    target = zero_index + 120

    src = ricker_wavelet(samples=n, dt=dt, freq_hz=0.9)
    resp = np.zeros(n, dtype=np.float64)
    resp[target] = 1.0
    base = convolve_same(src, resp)

    rng = np.random.default_rng(20260304)
    traces = []
    for _ in range(24):
        s = int(rng.integers(-6, 7))
        tr = np.roll(base, s) + 0.01 * rng.standard_normal(n)
        traces.append(tr)
    obs = np.asarray(traces, dtype=np.float64)

    cfg = MethodConfig(
        method="stack",
        dt=dt,
        filter_spec=FilterSpec(
            filter_type="butterworth_bandpass",
            gauss_f0=np.pi,
            low_hz=0.1,
            high_hz=0.8,
            corners=4,
        ),
        stack_peak_window_start_sec=-10.0,
        stack_peak_window_end_sec=40.0,
        stack_zero_index=zero_index,
    )

    rec, ok, _ = run_batch_method(obs, src, cfg, mode="optimized", jobs=4)
    ref = _legacy_stack_batch(obs, cfg)

    assert int(np.count_nonzero(ok)) == obs.shape[0]
    diff = rec - ref
    assert float(np.mean(np.abs(diff))) < 1e-12
    assert float(np.max(np.abs(diff))) < 1e-10

    peaks = np.argmax(np.abs(rec), axis=1)
    max_dev = int(np.max(np.abs(peaks - zero_index)))
    assert max_dev <= 1


def test_stack_parallel_matches_serial_strict() -> None:
    src, _, obs = make_synthetic_batch(
        traces=40,
        samples=2001,
        dt=0.05,
        noise_std=0.015,
        rng_seed=31,
    )
    cfg = MethodConfig(
        method="stack",
        dt=0.05,
        filter_spec=FilterSpec(filter_type="butterworth_bandpass", low_hz=0.1, high_hz=0.8, corners=4),
        stack_peak_window_start_sec=-10.0,
        stack_peak_window_end_sec=40.0,
        stack_zero_index=200,
    )

    r0, ok0, _ = run_batch_method(obs, src, cfg, mode="optimized", jobs=1)
    r1, ok1, _ = run_batch_method(obs, src, cfg, mode="optimized", jobs=4)

    assert int(np.count_nonzero(ok0)) == obs.shape[0]
    assert int(np.count_nonzero(ok1)) == obs.shape[0]
    diff = float(np.mean(np.abs(r0 - r1)))
    assert diff < 1e-12
