from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vertirf.core.decon import DeconConfig, run_batch_decon
from vertirf.filters.zero_phase import FilterSpec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate real-case wiggle plot for VertiRF project intro"
    )
    p.add_argument(
        "--input-dir",
        default=r"D:\works_2\seismic_data_retrieval_1\data\prompt19\p14_like_lowpass_t200\convolved_npz",
    )
    p.add_argument("--event-id", default="")
    p.add_argument("--component", choices=["z", "r"], default="z")
    p.add_argument("--stations", type=int, default=20)
    p.add_argument("--time-start-sec", type=float, default=-10.0)
    p.add_argument("--time-end-sec", type=float, default=90.0)
    p.add_argument("--itmax", type=int, default=260)
    p.add_argument("--minderr", type=float, default=1e-3)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--filter-type", default="butterworth_bandpass")
    p.add_argument("--low-hz", type=float, default=0.1)
    p.add_argument("--high-hz", type=float, default=0.8)
    p.add_argument("--corners", type=int, default=4)
    p.add_argument("--allow-negative-impulse", action="store_true")
    p.add_argument(
        "--rf-display-gain",
        type=float,
        default=12.0,
        help="Display gain for recovered RF panel only",
    )
    p.add_argument(
        "--rf-smooth-samples",
        type=int,
        default=9,
        help="Odd moving-average length for RF display only (0/1 disables)",
    )
    p.add_argument("--out", default="assets/real_case_wiggle.png")
    p.add_argument("--meta-out", default="assets/real_case_wiggle.json")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _select_event_file(input_dir: Path, event_id: str) -> Path:
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"no event npz found under {input_dir}")
    if event_id:
        matched = [x for x in files if event_id in x.stem]
        if not matched:
            raise RuntimeError(f"event_id not found in filenames: {event_id}")
        return matched[0]
    return files[0]


def _window_indices(time_sec: np.ndarray, t0: float, t1: float) -> np.ndarray:
    mask = (time_sec >= float(t0)) & (time_sec <= float(t1))
    idx = np.where(mask)[0]
    if idx.size < 10:
        raise RuntimeError("time window too small after filtering")
    return idx


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    den = np.max(np.abs(arr), axis=1, keepdims=True) + 1e-12
    return arr / den


def _smooth_rows(x: np.ndarray, win: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    w = int(win)
    if w <= 1:
        return arr
    if w % 2 == 0:
        w += 1
    ker = np.ones((w,), dtype=np.float64) / float(w)
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        out[i] = np.convolve(arr[i], ker, mode="same")
    return out


def _plot_wiggle(
    ax: plt.Axes,
    mat: np.ndarray,
    t: np.ndarray,
    labels: list[str],
    title: str,
    color: str,
    gain: float,
) -> None:
    n = mat.shape[0]
    spacing = 1.6
    disp = np.asarray(mat, dtype=np.float64) * float(gain)

    for i in range(n):
        y0 = i * spacing
        y = disp[i] + y0
        ax.plot(t, y, color=color, lw=0.7)
        ax.fill_between(
            t,
            y0,
            y,
            where=y >= y0,
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    tick_idx = np.arange(0, n, max(1, n // 6), dtype=int)
    ax.set_yticks(tick_idx * spacing)
    ax.set_yticklabels([labels[i] for i in tick_idx], fontsize=8)
    ax.set_ylabel("Station")
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle="--")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    event_file = _select_event_file(input_dir, args.event_id)

    with np.load(event_file, allow_pickle=False) as d:
        event_id = str(d["event_id"]) if "event_id" in d else event_file.stem
        station_codes = np.asarray(d["station_codes"], dtype="U32")
        time_sec = np.asarray(d["time_sec"], dtype=np.float64)
        comp = np.asarray(
            d["z_component" if args.component == "z" else "r_component"],
            dtype=np.float64,
        )
        source = np.asarray(d["wavelet_amp"], dtype=np.float64)
        dt = float(d["dt_sec"]) if "dt_sec" in d else float(np.median(np.diff(time_sec)))

    n_samples = min(source.size, comp.shape[0])
    source = source[:n_samples]
    source = source / (np.max(np.abs(source)) + 1e-12)

    idx_t = _window_indices(time_sec[:n_samples], args.time_start_sec, args.time_end_sec)

    n_stations = min(int(args.stations), comp.shape[1])
    station_idx = np.arange(n_stations, dtype=int)

    observed = comp[:n_samples, station_idx].T.astype(np.float64)
    observed = _normalize_rows(observed)

    cfg = DeconConfig(
        dt=dt,
        tshift_sec=0.0,
        itmax=int(args.itmax),
        minderr=float(args.minderr),
        allow_negative_impulse=bool(args.allow_negative_impulse),
        filter_spec=FilterSpec(
            filter_type=str(args.filter_type),
            low_hz=float(args.low_hz),
            high_hz=float(args.high_hz),
            corners=int(args.corners),
            gauss_f0=float(np.pi),
        ),
    )

    recovered, ok, iters = run_batch_decon(
        observed,
        source,
        cfg,
        jobs=max(1, int(args.jobs)),
    )

    recovered = _normalize_rows(recovered)
    recovered_disp = _smooth_rows(recovered, int(args.rf_smooth_samples))
    recovered_disp = _normalize_rows(recovered_disp)

    t_rel = time_sec[:n_samples][idx_t]
    obs_plot = observed[:, idx_t]
    rec_plot = recovered_disp[:, idx_t]
    labels = [str(station_codes[i]) for i in station_idx]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.0, 8.2),
        dpi=max(100, int(args.dpi)),
        sharex=True,
    )
    _plot_wiggle(
        axes[0],
        obs_plot,
        t_rel,
        labels,
        "Observed Seismograms (Real Case)",
        "#1d4ed8",
        gain=1.0,
    )
    _plot_wiggle(
        axes[1],
        rec_plot,
        t_rel,
        labels,
        f"Recovered RF by VertiRF Decon (display gain x{float(args.rf_display_gain):g})",
        "#0f766e",
        gain=float(args.rf_display_gain),
    )

    for ax in axes:
        ax.axvline(0.0, color="#111827", ls="--", lw=1.0)
        ax.set_xlabel("Time (s)")

    fig.suptitle(
        (
            "VertiRF Real Decon Case | "
            f"event={event_id} | stations={n_stations} | filter={args.filter_type} | "
            f"window=[{float(args.time_start_sec):g}, {float(args.time_end_sec):g}]s"
        ),
        fontsize=10.8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    # Extract peak time distribution for visibility diagnostics.
    rec_peaks_idx = np.argmax(np.abs(recovered), axis=1)
    rec_peak_times = time_sec[:n_samples][rec_peaks_idx]

    meta = {
        "event_file": str(event_file),
        "event_id": event_id,
        "component": str(args.component),
        "stations_used": int(n_stations),
        "time_window_sec": [float(args.time_start_sec), float(args.time_end_sec)],
        "filter_type": str(args.filter_type),
        "allow_negative_impulse": bool(args.allow_negative_impulse),
        "itmax": int(args.itmax),
        "minderr": float(args.minderr),
        "jobs": int(args.jobs),
        "rf_display_gain": float(args.rf_display_gain),
        "rf_smooth_samples": int(args.rf_smooth_samples),
        "success_count": int(np.count_nonzero(ok)),
        "mean_iterations": float(np.mean(iters.astype(np.float64))),
        "rf_peak_time_sec_min": float(np.min(rec_peak_times)),
        "rf_peak_time_sec_max": float(np.max(rec_peak_times)),
        "output_png": str(out_png),
    }

    meta_path = Path(args.meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
