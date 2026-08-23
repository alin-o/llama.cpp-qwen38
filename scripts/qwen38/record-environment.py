#!/usr/bin/env python3
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gpu_line = command(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    )
    name, driver, memory_mib, compute_capability = [item.strip() for item in gpu_line.split(",")]
    report = {
        "recorded_at": time.time(),
        "host": platform.uname()._asdict(),
        "gpu": {
            "name": name,
            "driver": driver,
            "memory_mib": int(memory_mib),
            "compute_capability": compute_capability,
        },
        "toolchain": {
            "cuda_build_image": "nvidia/cuda:13.1.0-devel-ubuntu24.04",
            "cuda_runtime_image": "nvidia/cuda:13.1.0-runtime-ubuntu24.04",
            "cuda_architectures": "89",
            "cmake_flags": [
                "GGML_CUDA=ON",
                "BUILD_SHARED_LIBS=OFF",
                "GGML_STATIC=ON",
                "GGML_CUDA_FA=ON",
                "GGML_CUDA_FA_ALL_QUANTS=1",
            ],
        },
        "images": {
            "upstream": {
                "name": "qwen38-upstream:b10502",
                "id": command("docker", "image", "inspect", "--format", "{{.Id}}", "qwen38-upstream:b10502"),
            },
            "fork": {
                "name": "qwen38-turboquant:d7179ba3c",
                "id": command("docker", "image", "inspect", "--format", "{{.Id}}", "qwen38-turboquant:d7179ba3c"),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
