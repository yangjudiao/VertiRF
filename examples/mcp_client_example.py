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

        run_resp = request(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "run_decon_synthetic",
                "params": {
                    "mode": "optimized",
                    "filter_type": "tukey_bandpass",
                    "allow_negative_impulse": True,
                    "traces": 10,
                    "samples": 512,
                    "jobs": 2,
                    "itmax": 200,
                },
            },
        )
        print("RUN RESPONSE")
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
