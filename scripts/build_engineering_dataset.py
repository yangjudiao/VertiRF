from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build engineering benchmark dataset from existing convolved event NPZ files")
    p.add_argument(
        "--input-dir",
        default=r"D:\works_2\seismic_data_retrieval_1\data\prompt19\p14_like_lowpass_t200\convolved_npz",
    )
    p.add_argument("--out", default="data/engineering_benchmark/engineering_dataset.npz")
    p.add_argument("--events", type=int, default=12)
    p.add_argument("--stations", type=int, default=20)
    p.add_argument("--component", choices=["z", "r"], default="z")
    p.add_argument("--seed", type=int, default=20260303)
    p.add_argument("--max-wavelet-samples", type=int, default=4096)
    p.add_argument("--window-start-sec", type=float, default=-10.0)
    p.add_argument("--window-end-sec", type=float, default=90.0)
    return p.parse_args()


def choose_time_window_indices(time_sec: np.ndarray, start_sec: float, end_sec: float, target_len: int) -> np.ndarray:
    mask = (time_sec >= float(start_sec)) & (time_sec <= float(end_sec))
    idx = np.where(mask)[0]
    if idx.size < target_len:
        idx = np.arange(min(target_len, time_sec.size), dtype=np.int64)
    if idx.size > target_len:
        idx = idx[:target_len]
    return idx


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"no npz files found under {input_dir}")

    rng = np.random.default_rng(int(args.seed))
    n_events = min(int(args.events), len(files))
    event_indices = np.sort(rng.choice(len(files), size=n_events, replace=False))

    selected_files = [files[i] for i in event_indices]

    traces: list[np.ndarray] = []
    station_codes_all: list[str] = []
    event_ids_all: list[str] = []

    source_wavelet = None
    dt = None
    n_used_samples = None

    for f in selected_files:
        with np.load(f, allow_pickle=False) as d:
            event_id = str(d["event_id"]) if "event_id" in d else f.stem
            stations = np.asarray(d["station_codes"], dtype="U32")
            mat = np.asarray(d["z_component" if args.component == "z" else "r_component"], dtype=np.float64)
            t = np.asarray(d["time_sec"], dtype=np.float64)

            w = np.asarray(d["wavelet_amp"], dtype=np.float64)
            if dt is None:
                dt = float(d["dt_sec"]) if "dt_sec" in d else float(np.median(np.diff(t)))
                if int(args.max_wavelet_samples) > 0:
                    use_w = min(int(args.max_wavelet_samples), int(w.size))
                else:
                    use_w = int(w.size)
                source_wavelet = w[:use_w].copy()
                source_wavelet /= np.max(np.abs(source_wavelet)) + 1e-12
                n_used_samples = int(source_wavelet.size)

            assert source_wavelet is not None
            assert n_used_samples is not None

            time_idx = choose_time_window_indices(
                t,
                start_sec=float(args.window_start_sec),
                end_sec=float(args.window_end_sec),
                target_len=n_used_samples,
            )

            n_sta = min(int(args.stations), mat.shape[1])
            sta_sel = np.sort(rng.choice(mat.shape[1], size=n_sta, replace=False))

            for sidx in sta_sel:
                tr = mat[time_idx, sidx].astype(np.float64)
                tr = tr / (np.max(np.abs(tr)) + 1e-12)
                traces.append(tr)
                station_codes_all.append(str(stations[sidx]))
                event_ids_all.append(event_id)

    if not traces:
        raise RuntimeError("no traces selected for dataset")

    observed = np.asarray(traces, dtype=np.float64)

    np.savez_compressed(
        out_path,
        source_wavelet=np.asarray(source_wavelet, dtype=np.float64),
        observed_traces=observed,
        dt_sec=np.asarray(float(dt), dtype=np.float64),
        component=np.asarray(str(args.component), dtype="U4"),
        station_codes=np.asarray(station_codes_all, dtype="U32"),
        event_ids=np.asarray(event_ids_all, dtype="U64"),
        input_dir=np.asarray(str(input_dir), dtype="U256"),
        selected_event_files=np.asarray([str(x) for x in selected_files], dtype="U512"),
    )

    summary = {
        "dataset_npz": str(out_path),
        "input_dir": str(input_dir),
        "component": str(args.component),
        "events_selected": int(len(selected_files)),
        "stations_per_event": int(args.stations),
        "traces_total": int(observed.shape[0]),
        "samples_per_trace": int(observed.shape[1]),
        "dt_sec": float(dt),
        "window_start_sec": float(args.window_start_sec),
        "window_end_sec": float(args.window_end_sec),
        "seed": int(args.seed),
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
