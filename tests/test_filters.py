from __future__ import annotations

import numpy as np

from vertirf.filters.zero_phase import FilterSpec, build_zero_phase_response


def test_all_filter_types_return_finite_response() -> None:
    types = ["gaussian", "butterworth_bandpass", "raised_cosine_bandpass", "tukey_bandpass"]
    for ft in types:
        spec = FilterSpec(filter_type=ft, gauss_f0=np.pi, low_hz=0.1, high_hz=0.8, corners=4)
        resp = build_zero_phase_response(nfft=1024, dt=0.05, spec=spec)
        assert resp.shape == (513,)
        assert np.isfinite(resp).all()
        assert float(resp.min()) >= 0.0
        assert float(resp.max()) <= 1.05
