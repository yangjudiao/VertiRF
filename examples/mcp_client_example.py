from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def request(proc: subprocess.Popen[str], payload: dict) -> dict:
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("agent server closed stdout")
    return json.loads(line)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "-m", "vertirf.agent.server"]

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    assert proc.stdin is not None
    assert proc.stdout is not None

    try:
        ping_resp = request(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "ping",
                "params": {},
            },
        )
        print("PING RESPONSE")
        print(json.dumps(ping_resp, indent=2))

        for rid, method_name in [(2, "decon"), (3, "corr"), (4, "stack")]:
            run_resp = request(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "method": "run_method_synthetic",
                    "params": {
                        "method": method_name,
                        "mode": "optimized",
                        "filter_type": "butterworth_bandpass",
                        "allow_negative_impulse": True,
                        "traces": 8,
                        "samples": 512,
                        "jobs": 2,
                        "itmax": 200,
                        "corr_smoothing_bandwidth_hz": 0.25,
                        "corr_post_filter_type": "gaussian",
                        "stack_peak_window_start_sec": -2,
                        "stack_peak_window_end_sec": 20,
                    },
                },
            )
            print(f"RUN RESPONSE [{method_name}]")
            print(json.dumps(run_resp, indent=2))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
