from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vertirf.core.methods import MethodConfig, run_batch_method
from vertirf.filters.zero_phase import FilterSpec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate real-case decon/corr/stack wiggle figure")
    p.add_argument(
        "--input-dir",
        default=r"D:\works_2\seismic_data_retrieval_1\data\prompt19\p14_like_lowpass_t200\convolved_npz",
    )
    p.add_argument("--event-id", default="")
    p.add_argument("--component", choices=["z", "r"], default="z")
    p.add_argument("--stations", type=int, default=20)
    p.add_argument("--time-start-sec", type=float, default=-10.0)
    p.add_argument("--time-end-sec", type=float, default=90.0)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--itmax", type=int, default=260)
    p.add_argument("--minderr", type=float, default=1e-3)
    p.add_argument("--filter-type", default="butterworth_bandpass")
    p.add_argument("--low-hz", type=float, default=0.1)
    p.add_argument("--high-hz", type=float, default=0.8)
    p.add_argument("--corners", type=int, default=4)
    p.add_argument("--allow-negative-impulse", action="store_true")

    p.add_argument("--corr-smoothing-bandwidth-hz", type=float, default=0.25)
    p.add_argument("--corr-post-filter-type", default="gaussian")
    p.add_argument("--stack-peak-window-start-sec", type=float, default=-2.0)
    p.add_argument("--stack-peak-window-end-sec", type=float, default=20.0)

    p.add_argument("--decon-display-gain", type=float, default=10.0)
    p.add_argument("--corr-display-gain", type=float, default=6.0)
    p.add_argument("--stack-display-gain", type=float, default=3.0)
    p.add_argument("--out", default="assets/real_case_three_methods_wiggle.png")
    p.add_argument("--meta-out", default="assets/real_case_three_methods_wiggle.json")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _select_event_file(input_dir: Path, event_id: str) -> Path:
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"no event npz found under {input_dir}")
    if event_id:
        m = [x for x in files if event_id in x.stem]
        if not m:
            raise RuntimeError(f"event_id not found in filenames: {event_id}")
        return m[0]
    return files[0]


def _window_indices(time_sec: np.ndarray, t0: float, t1: float) -> np.ndarray:
    idx = np.where((time_sec >= float(t0)) & (time_sec <= float(t1)))[0]
    if idx.size < 10:
        raise RuntimeError("time window too small after filtering")
    return idx


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    den = np.max(np.abs(arr), axis=1, keepdims=True) + 1e-12
    return arr / den


def _plot_panel(
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
        y = y0 + disp[i]
        ax.plot(t, y, color=color, lw=0.7)
        ax.fill_between(t, y0, y, where=(y >= y0), color=color, alpha=0.10, linewidth=0)

    tick_idx = np.arange(0, n, max(1, n // 6), dtype=int)
    ax.set_yticks(tick_idx * spacing)
    ax.set_yticklabels([labels[i] for i in tick_idx], fontsize=8)
    ax.set_ylabel("Station")
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle="--")


def _build_cfg(method: str, dt: float, args: argparse.Namespace) -> MethodConfig:
    return MethodConfig(
        method=method,
        dt=float(dt),
        tshift_sec=0.0,
        itmax=int(args.itmax),
        minderr=float(args.minderr),
        allow_negative_impulse=bool(args.allow_negative_impulse),
        filter_spec=FilterSpec(
            filter_type=str(args.filter_type),
            gauss_f0=float(np.pi),
            low_hz=float(args.low_hz),
            high_hz=float(args.high_hz),
            corners=int(args.corners),
            transition_hz=0.05,
            tukey_alpha=0.3,
        ),
        corr_smoothing_bandwidth_hz=float(args.corr_smoothing_bandwidth_hz),
        corr_post_filter_type=str(args.corr_post_filter_type),
        corr_post_gauss_f0=float(np.pi),
        corr_post_low_hz=float(args.low_hz),
        corr_post_high_hz=float(args.high_hz),
        corr_post_corners=int(args.corners),
        stack_peak_window_start_sec=float(args.stack_peak_window_start_sec),
        stack_peak_window_end_sec=float(args.stack_peak_window_end_sec),
    )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    event_file = _select_event_file(input_dir, args.event_id)

    with np.load(event_file, allow_pickle=False) as d:
        event_id = str(d["event_id"]) if "event_id" in d else event_file.stem
        station_codes = np.asarray(d["station_codes"], dtype="U32")
        time_sec = np.asarray(d["time_sec"], dtype=np.float64)
        comp = np.asarray(d["z_component" if args.component == "z" else "r_component"], dtype=np.float64)
        source = np.asarray(d["wavelet_amp"], dtype=np.float64)
        dt = float(d["dt_sec"]) if "dt_sec" in d else float(np.median(np.diff(time_sec)))

    n_samples = min(source.size, comp.shape[0])
    source = source[:n_samples]
    source = source / (np.max(np.abs(source)) + 1e-12)

    n_stations = min(int(args.stations), comp.shape[1])
    station_idx = np.arange(n_stations, dtype=int)
    observed = comp[:n_samples, station_idx].T.astype(np.float64)
    observed = _normalize_rows(observed)

    idx_t = _window_indices(time_sec[:n_samples], args.time_start_sec, args.time_end_sec)
    t = time_sec[:n_samples][idx_t]
    labels = [str(station_codes[i]) for i in station_idx]

    methods = ["decon", "corr", "stack"]
    rows: dict[str, np.ndarray] = {}
    status: dict[str, dict] = {}

    for m in methods:
        cfg = _build_cfg(m, dt, args)
        out, ok, steps = run_batch_method(observed, source, cfg, mode="optimized", jobs=max(1, int(args.jobs)))
        rows[m] = _normalize_rows(out)[:, idx_t]
        status[m] = {
            "success_count": int(np.count_nonzero(ok)),
            "mean_steps": float(np.mean(steps.astype(np.float64))),
        }

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 8.4), dpi=max(100, int(args.dpi)), sharex=True)
    _plot_panel(
        axes[0],
        rows["decon"],
        t,
        labels,
        f"Decon (gain x{float(args.decon_display_gain):g})",
        "#0f766e",
        float(args.decon_display_gain),
    )
    _plot_panel(
        axes[1],
        rows["corr"],
        t,
        labels,
        f"Corr (gain x{float(args.corr_display_gain):g})",
        "#1d4ed8",
        float(args.corr_display_gain),
    )
    _plot_panel(
        axes[2],
        rows["stack"],
        t,
        labels,
        f"Stack (gain x{float(args.stack_display_gain):g})",
        "#b45309",
        float(args.stack_display_gain),
    )

    for ax in axes:
        ax.axvline(0.0, color="#111827", ls="--", lw=1.0)
        ax.set_xlabel("Time (s)")

    fig.suptitle(
        (
            "VertiRF Real Case: Decon vs Corr vs Stack | "
            f"event={event_id} | stations={n_stations} | "
            f"window=[{float(args.time_start_sec):g}, {float(args.time_end_sec):g}]s"
        ),
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    meta = {
        "event_file": str(event_file),
        "event_id": event_id,
        "component": str(args.component),
        "stations_used": int(n_stations),
        "time_window_sec": [float(args.time_start_sec), float(args.time_end_sec)],
        "filter_type": str(args.filter_type),
        "corr_smoothing_bandwidth_hz": float(args.corr_smoothing_bandwidth_hz),
        "stack_peak_window_sec": [
            float(args.stack_peak_window_start_sec),
            float(args.stack_peak_window_end_sec),
        ],
        "jobs": int(args.jobs),
        "method_status": status,
        "output_png": str(out_png),
    }

    meta_path = Path(args.meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
