from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build native C++ corr backend DLL (Windows/cl)")
    p.add_argument("--src", type=Path, default=Path("src/vertirf/native/corr_kernel.cpp"))
    p.add_argument("--out", type=Path, default=Path("src/vertirf/native/vertirf_native_corr.dll"))
    p.add_argument("--status-out", type=Path, default=Path("assets/native_backend_status.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cl = shutil.which("cl")

    status = {
        "toolchain": {
            "cl": cl,
            "g++": shutil.which("g++"),
            "gcc": shutil.which("gcc"),
            "cmake": shutil.which("cmake"),
        },
        "build_attempted": True,
        "build_succeeded": False,
        "dll_path": str(args.out),
        "message": "",
    }

    if cl is None:
        status["message"] = "cl compiler not found"
        args.status_out.parent.mkdir(parents=True, exist_ok=True)
        args.status_out.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2))
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cl",
        "/nologo",
        "/O2",
        "/LD",
        "/EHsc",
        str(args.src),
        f"/Fe:{args.out}",
    ]

    vsdev = Path(
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
    )
    if vsdev.exists():
        cl_cmd = " ".join(
            [f"\"{c}\"" if (" " in c and not c.startswith("/")) else c for c in cmd]
        )
        cmdline = f"call \"{vsdev}\" -arch=x64 >nul && {cl_cmd}"
        proc = subprocess.run(
            cmdline,
            capture_output=True,
            text=False,
            shell=True,
            env=os.environ.copy(),
        )
    else:
        proc = subprocess.run(cmd, capture_output=True, text=False, shell=False)

    status["returncode"] = int(proc.returncode)
    stdout_text = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
    stderr_text = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
    status["stdout_tail"] = stdout_text[-1200:]
    status["stderr_tail"] = stderr_text[-1200:]

    if proc.returncode == 0 and args.out.exists():
        status["build_succeeded"] = True
        status["message"] = "native corr backend DLL built"
    else:
        status["message"] = "native corr backend build failed"

    args.status_out.parent.mkdir(parents=True, exist_ok=True)
    args.status_out.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if status["build_succeeded"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
