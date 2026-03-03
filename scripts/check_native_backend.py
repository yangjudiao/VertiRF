from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check and optionally build native (C/C++) backend")
    p.add_argument("--out", type=Path, default=Path("assets/native_backend_status.json"))
    p.add_argument("--attempt-build", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    tools = {
        "cl": shutil.which("cl"),
        "g++": shutil.which("g++"),
        "gcc": shutil.which("gcc"),
        "cmake": shutil.which("cmake"),
    }
    compiler_available = bool(tools["cl"] or tools["g++"] or tools["gcc"])

    status = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "toolchain": tools,
        "native_backend_available": False,
        "target_language": "C/C++",
        "build_attempted": False,
        "build_succeeded": False,
        "action": "",
        "notes": [],
    }

    if compiler_available and bool(args.attempt_build):
        cmd = [
            sys.executable,
            "scripts/build_native_corr_backend.py",
            "--status-out",
            str(args.out),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        status["build_attempted"] = True
        status["build_succeeded"] = proc.returncode == 0
        status["native_backend_available"] = proc.returncode == 0
        status["action"] = "native_corr_backend_built" if proc.returncode == 0 else "native_build_failed_fallback"
        status["notes"].append("build_native_corr_backend.py executed")
        status["notes"].append(f"returncode={proc.returncode}")
        status["notes"].append(proc.stdout[-600:])
        if proc.stderr:
            status["notes"].append(proc.stderr[-600:])
    elif compiler_available:
        status["action"] = "toolchain_detected_build_skipped"
        status["notes"].append("Compiler found but build step skipped")
    else:
        status["action"] = "toolchain_missing_use_numpy_parallel_fallback"
        status["notes"].append("No C/C++ compiler found in PATH")

    status["notes"].append("Runtime fallback remains optimized NumPy + parallel threads")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if status["native_backend_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
