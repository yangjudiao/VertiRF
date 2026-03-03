from __future__ import annotations

from dataclasses import replace

import numpy as np

from vertirf.core.decon import DeconConfig, normalize_max_abs, run_batch_decon
from vertirf.filters.zero_phase import FilterSpec, build_zero_phase_response
from vertirf.waveform.synthetic import make_synthetic_batch


def _cfg(filter_type: str, allow_negative: bool, tshift_sec: float = 10.0) -> DeconConfig:
    return DeconConfig(
        dt=0.05,
        tshift_sec=float(tshift_sec),
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


def _next_pow_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << int((n - 1).bit_length())


def _build_decon_filter_full(nfft: int, dt: float, spec: FilterSpec) -> np.ndarray:
    amp = build_zero_phase_response(nfft=nfft, dt=dt, spec=spec)
    half = int(0.5 * nfft + 1)
    out = np.zeros((nfft,), dtype=np.float64)
    out[:half] = amp / float(dt)
    if nfft > 2:
        out[half:] = np.flip(out[1 : half - 1])
    return out


def _gfilter(x: np.ndarray, nfft: int, filt: np.ndarray, dt: float) -> np.ndarray:
    xf = np.fft.fft(np.asarray(x, dtype=np.float64), nfft)
    xf = xf * np.asarray(filt) * float(dt)
    return np.fft.ifft(xf, nfft).real


def _correl(r: np.ndarray, w: np.ndarray, nfft: int) -> np.ndarray:
    return np.fft.ifft(np.fft.fft(r, nfft) * np.conj(np.fft.fft(w, nfft)), nfft).real


def _phase_shift_legacy(x: np.ndarray, nfft: int, dt: float, tshift: float) -> np.ndarray:
    xf = np.fft.fft(x, nfft)
    shift_i = int(float(tshift) / float(dt))
    p = 2.0 * np.pi * np.arange(1, nfft + 1, dtype=np.float64) * shift_i / float(nfft)
    xf = xf * (np.cos(p) - 1j * np.sin(p))
    y = np.fft.ifft(xf, nfft) / np.cos(2.0 * np.pi * shift_i / float(nfft))
    return y.real


def _legacy_decon_single(observed: np.ndarray, source: np.ndarray, cfg: DeconConfig) -> tuple[np.ndarray, int]:
    nt = int(observed.size)
    nfft = _next_pow_2(nt)

    p0 = np.zeros((nfft,), dtype=np.float64)
    u0 = np.zeros((nfft,), dtype=np.float64)
    w0 = np.zeros((nfft,), dtype=np.float64)
    u0[:nt] = observed
    w0[:nt] = source

    decon_f = _build_decon_filter_full(nfft=nfft, dt=cfg.dt, spec=cfg.filter_spec)
    u_flt = _gfilter(u0, nfft, decon_f, cfg.dt)
    w_flt = _gfilter(w0, nfft, decon_f, cfg.dt)
    wf = np.fft.fft(w0, nfft)

    r_flt = u_flt
    power_u = float(np.sum(u_flt**2))
    if power_u <= 0.0:
        raise RuntimeError("invalid source power in decon")

    it = 0
    sumsq_i = 1.0
    d_error = 100.0 * power_u + float(cfg.minderr)
    maxlag = int(0.5 * nfft)
    denom = float(np.sum(w_flt**2))

    while abs(d_error) > float(cfg.minderr) and it < int(cfg.itmax):
        rw = _correl(r_flt, w_flt, nfft)
        rw = rw / denom
        if bool(cfg.allow_negative_impulse):
            i1 = int(np.argmax(np.abs(rw)))
        else:
            i1 = int(np.argmax(np.abs(rw[0 : int(maxlag) - 1])))
        amp = float(rw[i1] / cfg.dt)
        p0[i1] = p0[i1] + amp

        p_flt = _gfilter(p0, nfft, decon_f, cfg.dt)
        p_flt = _gfilter(p_flt, nfft, wf, cfg.dt)
        r_flt = u_flt - p_flt

        sumsq = float(np.sum(r_flt**2) / power_u)
        d_error = 100.0 * (sumsq_i - sumsq)
        sumsq_i = sumsq
        it += 1

    p_flt = _gfilter(p0, nfft, decon_f, cfg.dt)
    p_flt = _phase_shift_legacy(p_flt, nfft, cfg.dt, cfg.tshift_sec)
    rf = p_flt[:nt]
    rf, _ = normalize_max_abs(rf)
    return rf, it


def test_decon_matches_legacy_reference_strict() -> None:
    src, _, obs = make_synthetic_batch(traces=6, samples=512, dt=0.05, noise_std=0.01, rng_seed=11)
    cfg = _cfg("butterworth_bandpass", allow_negative=False, tshift_sec=10.0)

    rec, ok, iters = run_batch_decon(obs, src, cfg, jobs=1)
    assert int(np.count_nonzero(ok)) >= 5

    maes: list[float] = []
    maxs: list[float] = []
    ref_iters: list[int] = []
    for i in range(obs.shape[0]):
        rf_ref, it_ref = _legacy_decon_single(obs[i], src, cfg)
        ref_iters.append(int(it_ref))

        diff = rec[i] - rf_ref
        maes.append(float(np.mean(np.abs(diff))))
        maxs.append(float(np.max(np.abs(diff))))

    assert float(np.mean(np.asarray(maes, dtype=np.float64))) < 1e-12
    assert float(np.max(np.asarray(maxs, dtype=np.float64))) < 1e-10
    assert np.array_equal(iters.astype(np.int64), np.asarray(ref_iters, dtype=np.int64))


def test_allow_negative_impulse_expands_search_domain() -> None:
    n = 256
    src = np.random.default_rng(0).normal(size=n)
    obs = np.roll(src, 190)

    cfg_pos = replace(_cfg("gaussian", allow_negative=False, tshift_sec=0.0), itmax=1)
    cfg_full = replace(_cfg("gaussian", allow_negative=True, tshift_sec=0.0), itmax=1)

    rf_pos, ok_pos, _ = run_batch_decon(obs[None, :], src, cfg_pos, jobs=1)
    rf_full, ok_full, _ = run_batch_decon(obs[None, :], src, cfg_full, jobs=1)

    assert bool(ok_pos[0])
    assert bool(ok_full[0])

    idx_pos = int(np.argmax(np.abs(rf_pos[0])))
    idx_full = int(np.argmax(np.abs(rf_full[0])))
    assert idx_pos < (n // 2)
    assert idx_full > idx_pos + 20


def test_serial_and_parallel_are_numerically_close() -> None:
    src, _, obs = make_synthetic_batch(traces=10, samples=512, dt=0.05, noise_std=0.01, rng_seed=9)
    cfg = _cfg("butterworth_bandpass", allow_negative=True, tshift_sec=10.0)

    r0, ok0, _ = run_batch_decon(obs, src, cfg, jobs=1)
    r1, ok1, _ = run_batch_decon(obs, src, cfg, jobs=3)

    assert int(np.count_nonzero(ok0)) >= 8
    assert int(np.count_nonzero(ok1)) >= 8
    diff = float(np.mean(np.abs(r0 - r1)))
    assert diff < 1e-12
