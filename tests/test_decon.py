from __future__ import annotations

import numpy as np

from vertirf.core.decon import DeconConfig, run_batch_decon
from vertirf.filters.zero_phase import FilterSpec, apply_zero_phase_filter
from vertirf.waveform.synthetic import convolve_same, corrcoef, make_synthetic_batch, ricker_wavelet


def _cfg(filter_type: str, allow_negative: bool) -> DeconConfig:
    return DeconConfig(
        dt=0.05,
        tshift_sec=0.0,
        itmax=280,
        minderr=1e-3,
        allow_negative_impulse=allow_negative,
        filter_spec=FilterSpec(
            filter_type=filter_type,
            gauss_f0=np.pi,
            low_hz=0.1,
            high_hz=0.8,
            corners=4,
            transition_hz=0.05,
            tukey_alpha=0.3,
        ),
    )


def test_closed_loop_recovery_across_filters() -> None:
    src, truth, obs = make_synthetic_batch(traces=6, samples=512, dt=0.05, noise_std=0.005, rng_seed=3)

    for ft in ["gaussian", "butterworth_bandpass", "raised_cosine_bandpass", "tukey_bandpass"]:
        cfg = _cfg(ft, allow_negative=True)
        rec, ok, _ = run_batch_decon(obs, src, cfg, jobs=2)
        assert int(np.count_nonzero(ok)) >= 5

        src_f = apply_zero_phase_filter(src, cfg.dt, cfg.filter_spec)
        recon_cc: list[float] = []
        peak_offsets: list[int] = []
        for i in range(rec.shape[0]):
            obs_f = apply_zero_phase_filter(obs[i], cfg.dt, cfg.filter_spec)
            recon = convolve_same(src_f, rec[i])
            recon_cc.append(corrcoef(recon, obs_f))
            peak_offsets.append(int(np.argmax(np.abs(rec[i]))) - int(np.argmax(np.abs(truth[i]))))

        assert float(np.nanmean(np.asarray(recon_cc, dtype=np.float64))) > 0.90
        offsets = np.asarray(peak_offsets, dtype=np.float64)
        assert float(np.std(offsets)) <= 1.0
        assert float(np.max(np.abs(offsets))) <= 32.0


def test_allow_negative_impulse_switch_changes_precenter_recovery() -> None:
    n = 256
    dt = 0.05
    src = ricker_wavelet(samples=n, dt=dt, freq_hz=0.9)

    response = np.zeros(n, dtype=np.float64)
    center = n // 2
    response[center - 35] = 1.0
    obs = convolve_same(src, response)

    cfg_pos = _cfg("gaussian", allow_negative=False)
    cfg_full = _cfg("gaussian", allow_negative=True)

    rf_pos, _, _ = run_batch_decon(obs[None, :], src, cfg_pos, jobs=1)
    rf_full, _, _ = run_batch_decon(obs[None, :], src, cfg_full, jobs=1)

    pre_pos = float(np.max(np.abs(rf_pos[0, :center])))
    pre_full = float(np.max(np.abs(rf_full[0, :center])))
    assert pre_full > pre_pos * 1.2


def test_serial_and_parallel_are_numerically_close() -> None:
    src, _, obs = make_synthetic_batch(traces=10, samples=512, dt=0.05, noise_std=0.01, rng_seed=9)
    cfg = _cfg("butterworth_bandpass", allow_negative=True)

    r0, ok0, _ = run_batch_decon(obs, src, cfg, jobs=1)
    r1, ok1, _ = run_batch_decon(obs, src, cfg, jobs=3)

    assert int(np.count_nonzero(ok0)) >= 8
    assert int(np.count_nonzero(ok1)) >= 8
    diff = float(np.mean(np.abs(r0 - r1)))
    assert diff < 1e-12
